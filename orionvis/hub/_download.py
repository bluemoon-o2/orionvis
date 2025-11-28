import logging
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
import torch
import warnings
import zipfile
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
from types import ModuleType
from typing import Any, Optional, Union, Literal, TypeGuard, Iterable, Callable, List, Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from torch.serialization import MAP_LOCATION

from ..utils import sys_path
from .._version import __version__

__all__ = ["download_url_to_file", "get_dir", "set_dir", "docs", "entrypoints", "load", "load_state_dict_from_url"]

logger = logging.getLogger(__name__)


class EnvVar(Enum):
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

VAR_DEPENDENCY: str = "dependencies"
ORIONVIS_GATEWAY: str = "gateway.py"
TrustRepoType = Literal[True, False, "check"]
_TRUSTED_REPO_OWNERS: set[str] = {
    "bluemoon-o2",
    "pytorch",
    "facebookresearch",
}

DEFAULT_CACHE_DIR: str = os.path.expanduser("~/.cache/orionvis")
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
_hub_dir: Optional[str] = None

DEFAULT_RETRY_DELAY : float = 1.0
MAX_RETRY_DELAY : float = 10.0
READ_DATA_CHUNK: int = 128 * 1024
HASH_REGEX: re.Pattern = re.compile(r"-([a-f0-9]*)\.")
USER_AGENT = "OrionVis/{} (Python/{}.{}; {}; hash={})".format(
    __version__,
    sys.version_info.major,
    sys.version_info.minor,
    sys.platform,
    hashlib.sha256(sys.platform.encode()).hexdigest()[:8]
)


class UntrustedRepoError(Exception):
    """Raised when the repository is not trusted and user declines to trust it."""
    pass


class DenyVisitError(Exception):
    """Raised when the repository denies user permission to visit it."""
    pass


def unit_scale(size: Union[int, float]) -> str:
    if not isinstance(size, (int, float)):
        raise TypeError(f"Invalid downloaded size: {size}")
    if size < 0:
        raise ValueError("Invalid downloaded size: negative value")
    if size == 0:
        return "0.00 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    count = 0
    while size > 1024 and count < len(units) - 1:
        size /= 1024
        count += 1
    return f"{size:.2f} {units[count]}"


def get_dir() -> str:
    """Get the OrionVis Hub cache directory used for storing downloaded models & weights."""
    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(DEFAULT_CACHE_DIR, "hub")


def set_dir(d: Union[str, os.PathLike]) -> None:
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    _hub_dir = os.path.expanduser(d)
    os.makedirs(_hub_dir, exist_ok=True)


def is_non_string_iterable(obj: Any) -> TypeGuard[Iterable[Any]]:
    """Check if obj is an iterable."""
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _verify_string(model: str) -> None:
    """Verify that the model name is a non-empty string."""
    if not model.strip():
        raise ValueError(
            "model must be a non-empty string function name.\n"
            "Please specify a valid function name."
        )


@lru_cache(maxsize=128)
def _check_module_exists(module_name: str) -> bool:
    """Lazily check if a module exists."""
    if not isinstance(module_name, str) or not module_name.strip():
        return False
    import importlib.util
    return importlib.util.find_spec(module_name.strip()) is not None


def _import_module(name: str, path: str) -> ModuleType:
    import importlib.util
    from importlib.abc import Loader

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(module)
    return module


def _remove_all(path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.exists():
        return
    try:
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except PermissionError as e:
        raise RuntimeError(f"Permission denied to remove {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to remove {path}") from e


def _git_archive_link(repo_owner: str, repo_name: str, ref: str) -> str:
    # See https://docs.github.com/en/rest/repos?apiVersion=2022-11-28
    return f"https://github.com/{repo_owner}/{repo_name}/zipball/{ref}"


def _response_request(request: Request, timeout: int = 3) -> str:
    with urlopen(request, timeout=timeout) as r:
        return r.read().decode(r.headers.get_content_charset("utf-8"))


def _parse_repo_info(github: str) -> tuple[str, str, str]:
    """Parse repo info from GitHub URL.

    Args:
        github (str): GitHub URL in the format of 'owner/repo' or 'owner/repo:ref'.

    Returns:
        tuple[str, str, str]: A tuple containing repo owner, repo name, and ref.
    """
    github = github.strip()
    if ":" in github:
        repo_info, ref = github.split(":", 1)
        ref = ref.strip() or None  # blank ref is None
    else:
        repo_info, ref = github.strip(), None

    repo_parts = repo_info.split("/")
    if len(repo_parts) != 2 or not all(part.strip() for part in repo_parts):
        raise ValueError(
            f"Invalid GitHub repo format: {github}\n"
            "Expected format: 'owner/repo' or 'owner/repo:ref'"
        )
    repo_owner, repo_name = [part.strip() for part in repo_parts]

    if ref is None:
        default_branches = ("main", "master")
        try:
            repo_tree_url = f"https://github.com/{repo_owner}/{repo_name}/tree/"
            for candidate in default_branches:
                with urlopen(repo_tree_url + candidate, timeout=3):
                    ref = candidate
                    break

        except HTTPError as e:
            if e.code == 404:
                ref = "master" if default_branches[0] == "main" else None
            elif e.code == 403:
                raise DenyVisitError(
                    f"Permission denied when checking branch for {repo_owner}/{repo_name} "
                    "(If the repo is private, set GITHUB_TOKEN in environment variables)"
                ) from e
            elif e.code >= 500:
                raise RuntimeError(
                    f"GitHub server error (HTTP {e.code}) when checking branch for {repo_owner}/{repo_name}"
                ) from e
            else:
                raise RuntimeError(f"Failed to check branch for {repo_owner}/{repo_name} (HTTP {e.code})") from e

        except URLError as e:
            cache_dir = Path(get_dir())
            for candidate in default_branches:
                cache_path = cache_dir / f"{repo_owner}_{repo_name}_{candidate}"
                if cache_path.exists():
                    ref = candidate
                    break

            if ref is None:
                raise RuntimeError(
                    "No internet connection and repo not found in cache.\n"
                    f"Cache directory: {cache_dir}\n"
                    f"Checked all default branches: {', '.join(default_branches)}"
                ) from e

    return repo_owner, repo_name, ref


def _validate_original_repo(repo_owner: str, repo_name: str, ref: str) -> None:
    # Use urlopen to avoid depending on local git.
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = EnvVar.GITHUB_TOKEN.value
    if token is not None:
        headers["Authorization"] = f"token {token}"

    api_paths = (
        f"repos/{repo_owner}/{repo_name}/branches",
        f"repos/{repo_owner}/{repo_name}/tags"
    )

    for path in api_paths:
        url_prefix = "https://api.github.com/" + path
        page = 0
        while True:
            page += 1
            url = f"{url_prefix}?per_page=100&page={page}&sha={ref[:7]}"
            try:
                response = json.loads(_response_request(Request(url, headers=headers)))
            except HTTPError as e:
                if e.code != 403 or "Authorization" not in headers:
                    raise RuntimeError(
                        f"GitHub API request failed (HTTP {e.code}): {url}\n"
                        f"Reason: {e.reason}"
                    ) from e
                # Retry without token in case it had insufficient permissions.
                del headers["Authorization"]
                response = json.loads(_response_request(Request(url, headers=headers)))
            except URLError as e:
                raise RuntimeError(f"Network error when accessing {url}: {e}") from e
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON response from {url}: {e}") from e
            # Empty response means no more data to process
            if not isinstance(response, list) or not response:
                break
            for item in response:
                item_name = item.get("name")
                # branch: commit.sha, tag: object.sha(commit/tag/tree)
                commit_sha = item.get("commit", {}).get("sha") or item.get("object", {}).get("sha")
                if not commit_sha:
                    continue
                if item_name == ref or commit_sha.startswith(ref):
                    return

    raise ValueError(
        f"Cannot find '{ref}' in https://github.com/{repo_owner}/{repo_name} (branches/tags/commits).\n"
        "Possible reasons:\n"
        "1. The ref does not exist in the original repo;\n"
        "2. The ref comes from a forked repo (call hub.load() with the forked repo directly);\n"
        "3. Insufficient permissions (add a valid GITHUB_TOKEN environment variable)."
    )


def _get_cache_or_reload(
        github: str,
        force_reload: bool = False,
        trust_repo: TrustRepoType = "check",
        verbose: bool = True,
        skip_validation: bool = False,
) -> str:
    """
    Load a GitHub repository to the local cache.

    Args:
        github: GitHub repository address (format: owner/repo or owner/repo:ref)
        force_reload: Whether to force reload the repository (ignore cache), default False
        trust_repo: Repository trust validation mode (refer to _check_repo_is_trusted definition), default "check"
        verbose: Whether to display verbose logs (download progress, cache prompts), default True
        skip_validation: Whether to skip original repository validation (avoid errors when ref is ambiguous), default False

    Returns:
        str: Local cache directory path (absolute path)

    Raises:
        RuntimeError: Download/extract/delete failed, permission denied, etc.
        HTTPError: GitHub API/download request failed
        ValueError: Repository parsing failed, trust validation failed
    """
    hub_dir = Path(get_dir()).resolve()
    repo_owner, repo_name, ref = _parse_repo_info(github)

    # normalize ref (replace '/' with '_')
    normalized_ref = ref.replace("/", "_") if ref else ""
    owner_name_branch = f"{repo_owner}_{repo_name}_{normalized_ref}"
    repo_dir = hub_dir / owner_name_branch

    _check_repo_is_trusted(
        repo_owner=repo_owner,
        repo_name=repo_name,
        trust_repo=trust_repo,
    )

    use_cache = not force_reload and repo_dir.is_dir() and any(repo_dir.iterdir())
    if use_cache:
        if verbose:
            logger.info(f"Using cache found: {repo_dir}")
        return str(repo_dir)

    if not skip_validation:
        _validate_original_repo(repo_owner, repo_name, ref)

    cached_zip = hub_dir / f"{normalized_ref}.zip"
    _remove_all(cached_zip)

    download_url = _git_archive_link(repo_owner, repo_name, ref)
    if verbose:
        logger.info(f"Downloading {cached_zip} from: {download_url}")

    try:
        download_url_to_file(download_url, str(cached_zip), progress=verbose)
    except HTTPError as err:
        if err.code == 300:
            warnings.warn(
                f"Ref '{ref}' has both branch and tag ambiguity in GitHub!\n"
                "OrionVis Hub will default to handling it as a branch.\n"
                "To specify explicitly:\n"
                f"  - Branch: pass ref='refs/heads/{ref}'\n"
                f"  - Tag: pass ref='refs/tags/{ref}'\n"
                "Specify explicitly to avoid ambiguity."
            )
            disambiguated_ref = f"refs/heads/{ref}"
            download_url = _git_archive_link(repo_owner, repo_name, disambiguated_ref)
            if verbose:
                logger.info(f"Downloading {cached_zip} from disambiguated branch: {download_url}")
            download_url_to_file(download_url, str(cached_zip), progress=verbose)
        else:
            raise RuntimeError(
                f"Download failed (HTTP {err.code}): {download_url}\n"
                f"Reason: {err.reason}"
            ) from err

    try:
        with zipfile.ZipFile(cached_zip, "r") as zipf:
            zip_entries = zipf.infolist()
            if not zip_entries:
                raise RuntimeError(f"Downloaded zip file is empty: {cached_zip}")

            root_dirs = set(entry.filename.split("/")[0] for entry in zip_entries if not entry.is_dir())
            extracted_root = root_dirs.pop() if root_dirs else zip_entries[0].filename.split("/")[0]
            extracted_repo_dir = hub_dir / extracted_root

            _remove_all(extracted_repo_dir)
            zipf.extractall(hub_dir)
            if verbose:
                logger.info(f"Extracted to: {extracted_repo_dir}")
    except zipfile.BadZipFile as e:
        _remove_all(cached_zip)
        raise RuntimeError(f"zip file is corrupted, extraction failed: {cached_zip}") from e
    except Exception as e:
        _remove_all(cached_zip)
        raise RuntimeError(f"Extraction failed: {cached_zip} with error: {e}") from e

    _remove_all(cached_zip)
    _remove_all(repo_dir)

    try:
        shutil.move(str(extracted_repo_dir), str(repo_dir))
    except shutil.Error as e:
        raise RuntimeError(f"Failed to rename extracted directory: {extracted_repo_dir} to {repo_dir}: {e}") from e

    if verbose:
        logger.info(f"Cache successfully saved to: {repo_dir}")

    return str(repo_dir)


def _read_trusted_list(trusted_file: Path) -> set[str]:
    trusted_file.parent.mkdir(exist_ok=True, parents=True)
    trusted_file.touch(exist_ok=True)

    with open(trusted_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _add_to_trusted_list(trusted_file: Path, owner_name: str) -> None:
    current_trusted = _read_trusted_list(trusted_file)
    if owner_name in current_trusted:
        logger.debug(f"{owner_name} is already in trusted list.")
        return
    current_trusted.add(owner_name)

    # Atomically write the updated list
    temp_file = trusted_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(current_trusted)) + "\n")
    temp_file.replace(trusted_file)


def _prompt_user_trust(owner_name: str) -> bool:
    prompt = (
        f"The repository '{owner_name}' is not in your trusted list.\n"
        "Downloading and running code from untrusted repositories may be risky.\n"
        "Do you trust this repository and want to add it to your trusted list? (y/N): "
    )

    while True:
        response = input(prompt).strip().lower()
        if not response:
            return False
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print(f"Unrecognized response: '{response}'. Please enter 'y' (yes) or 'n' (no).")


def _check_repo_is_trusted(
    repo_owner: str,
    repo_name: str,
    trust_repo: TrustRepoType,
):
    """
    Check if the repository is trusted.

    Args:
        repo_owner: The owner of the repository.
        repo_name: The name of the repository.
        trust_repo: The trust repository mode:
            - True: Directly trusts, no prompt
            - False: Forces explicit confirmation prompt
            - "check": Prompts only if the repo is not already trusted

    Raises:
        UntrustedRepoError: User denied trust for the repository
        ValueError: Invalid trust_repo value
    """
    if trust_repo not in (True, False, "check"):
        raise ValueError(
            f"Invalid value for 'trust_repo': {trust_repo}\n"
            f"Allowed values: True, False, 'check'"
        )

    hub_dir = Path(get_dir())
    trusted_file = hub_dir / "trusted_list"
    owner_name = f"{repo_owner}_{repo_name}"

    trusted_repos = _read_trusted_list(trusted_file)

    is_trusted = (
        owner_name in trusted_repos
        or repo_owner in _TRUSTED_REPO_OWNERS
    )

    if is_trusted or trust_repo is True:
        if trust_repo is True and not is_trusted:
            _add_to_trusted_list(trusted_file, owner_name)
            logger.info(f"Added {owner_name} to trusted list: {trusted_file}")
        return

    user_approved = _prompt_user_trust(owner_name)
    if not user_approved:
        raise UntrustedRepoError(f"Untrusted repository: {owner_name}")

    _add_to_trusted_list(trusted_file, owner_name)
    logger.info(f"Successfully added {owner_name} to trusted list.")


def _load_attr_from_module(module: ModuleType, attr_name: str) -> Optional[Any]:
    if not isinstance(module, ModuleType):
        raise TypeError(f"Expected module object, got {type(module).__name__}")
    return getattr(module, attr_name) if hasattr(module, attr_name) else None


def _check_dependencies(module: ModuleType) -> None:
    try:
        dependencies = _load_attr_from_module(module, VAR_DEPENDENCY)
    except NameError:
        return

    if not is_non_string_iterable(dependencies):
        raise TypeError(
            f"Invalid type for {VAR_DEPENDENCY} in {module.__name__}: "
            f"Expected iterable (list, tuple, etc.), got {type(dependencies).__name__}"
        )

    missing_deps = [pkg for pkg in dependencies if not _check_module_exists(str(pkg))]

    if missing_deps:
        raise RuntimeError(
            f"Module {module.__name__} is missing required dependencies: {', '.join(missing_deps)}\n"
        )


def _load_entry_from_hub_gateway(m: ModuleType, model: str) -> Callable[..., Any]:
    """
    Load a callable entrypoint function from a hub gateway module.

    Args:
        m: hub gateway module object (must be successfully imported, with no missing dependency errors)
        model: name of the entrypoint function to load (non-empty string)

    Returns:
        Callable[..., Any]: the loaded callable function

    Raises:
        TypeError: if ``m`` is not a module object or ``model`` is not a string
        ValueError: if ``model`` is an empty string
        RuntimeError: if the function does not exist in the module or exists but is not callable
    """
    if not isinstance(m, ModuleType):
        raise TypeError(
            f"The first argument must be a module object, but got {type(m).__name__}\n"
            "Please ensure the hub gateway module is successfully imported with no missing dependencies."
        )
    if not isinstance(model, str):
        raise ValueError(
            f"The second argument must be a string type function name, but got {type(model).__name__}\n"
            "Please specify a valid function name."
        )
    if not model.strip():
        raise ValueError(
            "The second argument 'model' must be a non-empty string function name.\n"
            "Please specify a valid function name."
        )

    _check_dependencies(m)

    func = _load_attr_from_module(m, model.strip())

    if func is None:
        raise RuntimeError(
            f"Function '{model}' not found in hub gateway module '{m.__name__}'\n"
            "Please check: 1. Function name spelling; 2. Function defined at top level."
        )
    if not callable(func):
        raise RuntimeError(
            f"Function '{model}' in hub gateway module '{m.__name__}' exists but is not callable (function or class).\n"
            f"Current type: {type(func).__name__}. Please ensure the target is a top-level callable function."
        )

    return func


def entrypoints(
    github: str,
    force_reload: bool = False,
    skip_validation: bool = False,
    trust_repo: TrustRepoType = "check",
    verbose: bool = True,
) -> List[str]:
    r"""
    List all callable entrypoints available in the repo specified by ``github``.

    entrypoints are defined as top-level callable functions in the gateway.py file,
    excluding those prefixed with an underscore.

    Args:
        github (str): github repository name with format <repo_owner/repo_name[:ref]>
        force_reload (bool): force reload of all callable entrypoints
        skip_validation (bool): skip validation of callable entrypoints
        trust_repo (TrustRepoType): trust repo or not
        verbose (bool): print verbose log messages

    Returns:
        List[str]: list of callable entrypoints

    Raises:
        RuntimeError: if the repository download fails or the gateway.py file is missing
    """
    repo_dir = _get_cache_or_reload(
        github=github,
        force_reload=force_reload,
        trust_repo=trust_repo,
        verbose=verbose,
        skip_validation=skip_validation,
    )
    repo_dir_path = Path(repo_dir).resolve()

    gateway_file = repo_dir_path / ORIONVIS_GATEWAY
    if not gateway_file.exists():
        raise RuntimeError(
            f"{ORIONVIS_GATEWAY} is not found in repo {github} from {gateway_file}\n"
            "Please check if the repo contains a valid gateway.py file."
        )

    try:
        with sys_path(repo_dir_path):
            hub_module = _import_module(ORIONVIS_GATEWAY[:-3], str(gateway_file))
    except ImportError as e:
        raise RuntimeError(
            f"Import {ORIONVIS_GATEWAY} failed (possibly due to syntax errors or missing dependencies)\n"
            f"Path: {gateway_file}\n"
            f"Error details: {e}"
        ) from e

    entrypoints = []
    for name in dir(hub_module):
        if name.startswith("_"):
            continue

        attr = getattr(hub_module, name)
        import inspect
        if inspect.isfunction(attr) and attr.__module__ == hub_module.__name__:
            entrypoints.append(name)

    if verbose:
        logger.info(
            f"Successfully loaded repo: {github}\n" 
            f"OrionVis Hub path: {gateway_file}\n"
            f"Found {len(entrypoints)} available entrypoints: {', '.join(entrypoints) if entrypoints else 'None'}"
        )
    return entrypoints


def docs(
    github: str,
    model: str,
    force_reload: bool = False,
    skip_validation: bool = False,
    trust_repo: TrustRepoType = "check",
    verbose: bool = True,
) -> Optional[str]:
    r"""
    Show the docstring of the specified entrypoint function in the repo specified by ``github``.

    Args:
        github (str): github repository name with format <repo_owner/repo_name[:ref]>
        model (str): entrypoint function name to show docstring for
        force_reload (bool): force reload of all callable entrypoints
        skip_validation (bool): skip validation of callable entrypoints
        trust_repo (TrustRepoType): trust repo or not
        verbose (bool): print verbose log messages

    Returns:
        Optional[str]: docstring of the specified entrypoint function, or None if not found

    Raises:
        RuntimeError: if the repository download fails or the gateway.py file is missing
    """
    _verify_string(model)

    repo_dir = _get_cache_or_reload(
        github=github,
        force_reload=force_reload,
        trust_repo=trust_repo,
        verbose=verbose,
        skip_validation=skip_validation,
    )
    repo_dir_path = Path(repo_dir).resolve()

    gateway_file = repo_dir_path / ORIONVIS_GATEWAY
    if not gateway_file.exists():
        raise RuntimeError(
            f"{ORIONVIS_GATEWAY} is not found in repo {github} from {gateway_file}\n"
            "Please check if the repo contains a valid gateway.py file."
        )

    module_name = ORIONVIS_GATEWAY[:-3] if ORIONVIS_GATEWAY.endswith(".py") else ORIONVIS_GATEWAY
    try:
        with sys_path(repo_dir_path):
            hub_module = _import_module(module_name, str(gateway_file))
    except ImportError as e:
        raise RuntimeError(
            f"Import {ORIONVIS_GATEWAY} failed (possibly due to syntax errors or missing dependencies)\n"
            f"Path: {gateway_file}\n"
            f"Error details: {e}"
        ) from e


    entry = _load_entry_from_hub_gateway(hub_module, model)

    docstring = entry.__doc__
    if docstring is None:
        docstring = f"Entrypoint function '{model}' has no docstring."
    else:
        docstring = docstring.strip()

    if verbose:
        logger.info(
            f"Successfully loaded help docstring: {model}\n"
            f"Repository: {github}\n"
            f"Entrypoint: {model}\n"
            f"Documentation source: {gateway_file}"
        )

    return docstring


def load(
        repo_or_dir: str,
        model: str,
        *args: Any,
        source: Literal["github", "local"] = "github",
        trust_repo: TrustRepoType = "check",
        force_reload: bool = False,
        verbose: bool = True,
        skip_validation: bool = False,
        **kwargs: Any,
) -> Any:
    r"""
    Load a model or entrypoint function from a GitHub repository or local directory.

    Supported sources:
    - source="github"：Load from a GitHub repository.(format: repo_owner/repo_name[:ref])
    - source="local"：Load from a local directory.(path must point to a directory containing gateway.py)

    Args:
        repo_or_dir (str): Source identifier:
            - source="github"：GitHub repository address (e.g., 'pytorch/vision:0.10')
            - source="local"：Local directory path (e.g., '/path/to/local/repo')
        model (str): Entrypoint function name defined in gateway.py
        *args (Any, optional): Positional arguments to pass to the entrypoint function
        source (Literal["github", "local"], optional): Source type, default "github"
        trust_repo (TrustRepoType, optional): Repository trust validation mode (only applies to source="github"):
            - ``False``: Prompt user for confirmation to trust the repository
            - ``True``: Automatically add to trusted list without prompt
            - "check": Prompt only if repository is not trusted (default in v2.0)
            - ``None``: Compatibility with old behavior (only warns, removed in v2.0)
            Default: ``None``
        force_reload (bool, optional): Whether to force reload of all callable entrypoints (only applies to source="github")，default ``False``
        verbose (bool, optional): Whether to print verbose log messages, default ``True``
        skip_validation (bool, optional): Whether to skip validation of callable entrypoints (only applies to source="github"), default ``False``
        **kwargs (Any, optional): Keyword arguments to pass to the entrypoint function

    Returns:
        Any: Result of calling the entrypoint function (e.g., model instance, utility class)

    Raises:
        ValueError: Invalid source, empty model name, or argument mismatch
        RuntimeError: Repository download failed, invalid local directory, missing/import error in gateway.py, or function not found
        ImportError: gateway.py syntax error or missing dependencies
        NameError: Core constants undefined (e.g., ORIONVIS_GATEWAY)
    """
    _verify_string(model)
    source = source.lower()

    if source not in ("github", "local"):
        raise ValueError(
            f"Invalid source: {source}\n"
            f"Allowed values: 'github', 'local'"
        )

    if source == "github":
        repo_dir = _get_cache_or_reload(
            github=repo_or_dir,
            force_reload=force_reload,
            trust_repo=trust_repo,
            verbose=verbose,
            skip_validation=skip_validation,
        )
        repo_dir_path = Path(repo_dir).resolve()
        if verbose:
            logger.info(f"Loaded from GitHub cache: {repo_or_dir} → Local path: {repo_dir_path}")

    else:
        repo_dir_path = Path(repo_or_dir).resolve()
        if not repo_dir_path.exists():
            raise RuntimeError(f"Local directory does not exist: {repo_dir_path}")
        if not repo_dir_path.is_dir():
            raise RuntimeError(f"Not a valid local directory: {repo_dir_path}")

        gateway_file = repo_dir_path / ORIONVIS_GATEWAY
        if not gateway_file.exists():
            raise RuntimeError(
                f"{ORIONVIS_GATEWAY} is not found in the local directory:\n"
                f"Directory path: {repo_dir_path}\n"
                "Please ensure the top-level directory contains a valid gateway.py"
            )

        if verbose:
            if force_reload:
                logger.warning("force_reload parameter is invalid for source='local'")
            if skip_validation:
                logger.warning("skip_validation parameter is invalid for source='local'")
            logger.info(f"Loading local directory: {repo_dir_path}")

    with sys_path(repo_dir_path):
        result = _load_local(
            repo_dir=str(repo_dir_path),
            model=model,
            *args,
            **kwargs
        )

    if verbose:
        logger.info(
            f"Successfully loaded entrypoint: {model}\n"
            f"Source type: {source}\n"
            f"Return type: {type(result).__name__}"
        )

    return result


def _load_local(gateway_dir: str, model: str, verbose: bool, *args, **kwargs):
    r"""
    Load a model from a local directory with a ``gateway.py``.

    Args:
        gateway_dir (str): path to a local directory that contains a
            ``gateway.py``.
        model (str): name of an entrypoint defined in the directory's
            ``gateway.py``.
        *args (optional): the corresponding args for callable ``model``.
        **kwargs (optional): the corresponding kwargs for callable ``model``.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> path = "/some/local/path/pytorch/vision"
        >>> model = _load_local(
        ...     gateway_dir,
        ...     "resnet50",
        ...     weights="ResNet50_Weights.IMAGENET1K_V1",
        ... )
    """
    _verify_string(model)

    gateway_dir_path = Path(gateway_dir).resolve()
    if not gateway_dir_path.exists():
        raise RuntimeError(f"{gateway_dir_path} does not exist")
    if not gateway_dir_path.is_dir():
        raise RuntimeError(f"{gateway_dir_path} is not a valid local directory")

    gateway_file = gateway_dir_path / ORIONVIS_GATEWAY
    if not gateway_file.exists():
        raise RuntimeError(
            f"{ORIONVIS_GATEWAY} is not found in the local directory:\n"
            f"Directory path: {gateway_dir_path}\n"
            "Please ensure the top-level directory contains a valid gateway.py"
        )

    module_name = ORIONVIS_GATEWAY[:-3] if ORIONVIS_GATEWAY.endswith(".py") else ORIONVIS_GATEWAY

    try:
        with sys_path(gateway_dir_path):
            if verbose:
                logger.info(f"Importing gateway module: {gateway_file} (module name: {module_name})")
            gateway_module = _import_module(module_name, str(gateway_file))
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import gateway module:\n"
            f"Module path: {gateway_file}\n"
            f"Error details: {e}\n"
            "Possible reasons: module syntax error, missing dependencies, incorrect filename suffix"
        ) from e

    entry = _load_entry_from_hub_gateway(gateway_module, model)

    if verbose:
        logger.info(f"Calling entrypoint function: {model} (args: {args}, kwargs: {kwargs})")
    instance = entry(*args, **kwargs)

    if verbose:
        logger.info(
            f"Successfully loaded entrypoint: {model}\n"
            f"Gateway directory: {gateway_dir_path}\n"
            f"Entrypoint function: {model}\n"
            f"Return result type: {type(instance).__name__}"
        )

    return instance


def download_url_to_file(
        url: str,
        dst: str | Path,
        hash_prefix: Optional[str] = None,
        progress: bool = True,
        timeout: float = 10.0,
        max_retries: int = 3,
        overwrite: bool = False,
        allow_resume: bool = True,
        user_agent: str = USER_AGENT,
) -> None:
    r"""
    Download a URL to a local file(Safe download: temp file + hash check + progress feedback).

    Features:
    - First download to a temp file, then move to the destination path to avoid corrupting the destination file.
    - Support SHA256 hash prefix check to ensure file integrity.
    - Show download progress bar (disable with progress=False).
    - Support network timeout and retry mechanism for stability.
    - Automatically create parent directories for the destination path.

    Args:
        url (str): URL address to download (supports HTTP/HTTPS).
        dst (str | Path): Destination path (including filename) to save the file.
        hash_prefix (Optional[str]): SHA256 hash prefix for integrity check, default None.
        progress (bool): Whether to show download progress bar, default True.
        timeout (float): Network request timeout in seconds, default 10.0.
        max_retries (int): Maximum number of retry attempts for network errors, default 3.
        overwrite (bool): Whether to overwrite existing file, default False.
        allow_resume (bool): Whether to support resuming interrupted downloads, default True.
        user_agent (str): Custom User-Agent header for HTTP requests, default USER_AGENT.

    Raises:
        URLError: Network connection failed (timeout/access denied).
        HTTPError: HTTP request failed (e.g., 404/500).
        RuntimeError: Hash check failed, download interrupted, or file operation failed.
        PermissionError: Destination path has no write permission.
    """
    if not url.strip():
        raise ValueError("Download URL cannot be an empty string")
    if hash_prefix is not None:
        if not isinstance(hash_prefix, str) or len(hash_prefix) < 4:
            raise TypeError(
                f"hash_prefix must be a string with length ≥ 4, but got {type(hash_prefix).__name__} "
                f"(length: {len(hash_prefix) if isinstance(hash_prefix, str) else 'N/A'})"
            )

    dst_path = Path(dst).resolve()
    dst_parent = dst_path.parent

    try:
        dst_parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Checked parent directory: {dst_parent}")
    except PermissionError as e:
        raise RuntimeError(f"Permission denied: cannot create parent directory {dst_parent} - {e}") from e

    if dst_path.exists():
        if overwrite:
            logger.warning(f"Target file {dst_path} already exists, will be overwritten.")
            dst_path.unlink(missing_ok=True)
        else:
            logger.info(f"Target file {dst_path} already exists, skip downloading.")
            return

    tmp_suffix = f".partial.{uuid.uuid4().hex}"
    tmp_dst = dst_path.with_suffix(f"{dst_path.suffix}{tmp_suffix}")
    downloaded_size = 0
    sha256: Optional[hashlib.sha256] = hashlib.sha256() if hash_prefix else None

    if allow_resume and tmp_dst.exists():
        try:
            downloaded_size = tmp_dst.stat().st_size
            if downloaded_size > 0:
                logger.info(f"Load from existing partial file: {tmp_dst} (size: {unit_scale(downloaded_size)})")
                if sha256:
                    with open(tmp_dst, "rb") as f:
                        while chunk := f.read(READ_DATA_CHUNK):
                            sha256.update(chunk)
                    logger.debug(f"Checksum: {sha256.hexdigest()}")
            else:
                tmp_dst.unlink(missing_ok=True)
                logger.debug(f"lean up empty partial file: {tmp_dst}")
        except Exception as e:
            tmp_dst.unlink(missing_ok=True)
            logger.warning(f"Failed to load from existing partial file: {tmp_dst} - {e}")
            downloaded_size = 0

    retry_count = 0
    while retry_count < max_retries:
        retry_delay = min(DEFAULT_RETRY_DELAY * (2 ** retry_count), MAX_RETRY_DELAY)
        try:
            headers = {"User-Agent": user_agent, "Accept": "*/*"}
            if allow_resume and downloaded_size > 0:
                headers["Range"] = f"bytes={downloaded_size}-"
                logger.debug(f"Send Range request: {headers['Range']}")

            req = Request(url, headers=headers)
            logger.info(f"Start downloading: {url} → temp file: {tmp_dst}")

            with urlopen(req, timeout=timeout) as u:
                status_code = u.status
                if status_code == 200:
                    logger.info(f"Downloading: {url} → {dst_path.name}")
                    total_size = int(u.headers.get("Content-Length", 0)) if u.headers.get("Content-Length", "").isdigit() else None
                elif status_code == 206 and allow_resume and downloaded_size > 0:
                    logger.info(f"Resume downloading: {url} → {dst_path.name} (from {unit_scale(downloaded_size)})")
                    remaining_size = int(u.headers.get("Content-Length", 0)) if u.headers.get("Content-Length", "").isdigit() else None
                    total_size = downloaded_size + remaining_size if remaining_size is not None else None
                elif status_code == 404:
                    raise HTTPError(url, status_code, "File not found", u.headers, None)
                elif status_code >= 500:
                    raise HTTPError(url, status_code, "Server internal error", u.headers, None)
                else:
                    raise RuntimeError(f"Unsupported HTTP status code: {status_code} (URL: {url})")

                if allow_resume and "accept-ranges" not in u.headers:
                    logger.warning(f"Server does not support resuming (missing Accept-Ranges header), will download from scratch.")

                tqdm_desc = f"[OrionVis] {'Resumed' if downloaded_size > 0 else 'Downloading'} {dst_path.name}"
                tqdm_kwargs = {
                    "total": total_size,
                    "initial": downloaded_size,
                    "disable": not progress,
                    "unit": "B",
                    "unit_scale": True,
                    "unit_divisor": 1024,
                    "desc": tqdm_desc,
                    "file": sys.stderr,
                    "leave": True,
                }

                with open(tmp_dst, "ab") as f, tqdm(**tqdm_kwargs) as pbar:
                    while True:
                        buffer = u.read(READ_DATA_CHUNK)
                        if not buffer:
                            break
                        f.write(buffer)
                        if sha256:
                            sha256.update(buffer)
                        pbar.update(len(buffer))

                if hash_prefix:
                    assert sha256 is not None, "Hash checker is not initialized"
                    digest = sha256.hexdigest()
                    if not digest.startswith(hash_prefix.lower()):
                        tmp_dst.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Hash check failed!\n"
                            f"File path: {dst_path}\n"
                            f"Expected prefix: {hash_prefix}\n"
                            f"Actual hash: {digest}\n"
                            f"Possible reasons: Network interrupt, URL error, file tampering"
                        )
                    logger.info(f"Hash check passed (SHA256 prefix: {hash_prefix})")

                try:
                    shutil.move(str(tmp_dst), str(dst_path))
                    logger.debug(f"Successfully moved: {tmp_dst} → {dst_path}")
                except PermissionError as e:
                    logger.warning(f"Permission denied: {e}. Retrying in 1 second...")
                    time.sleep(1)
                    shutil.move(str(tmp_dst), str(dst_path))
                except Exception as e:
                    raise RuntimeError(f"Failed to move temp file to final destination: {e}") from e

                return

        except HTTPError as e:
            retry_count += 1
            status_code = e.code
            error_msg = f"HTTP error (status code: {status_code}): {e.reason}"
            if status_code == 404:
                tmp_dst.unlink(missing_ok=True)
                raise RuntimeError(f"{error_msg} → URL:{url} (File not found, no retry needed)") from e
            elif status_code >= 500:
                error_msg += " (Server internal error, may be temporary)"
            logger.warning(f"Download failed (Retry {retry_count}/{max_retries}): {error_msg}")

            logger.debug(f"Next retry delay: {retry_delay:.1f} seconds")
            time.sleep(retry_delay)

            if retry_count >= max_retries:
                tmp_dst.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Network error: Retried {max_retries} times, still failed\n"
                    f"URL: {url}\n"
                    f"Error details: {error_msg}\n"
                    f"Suggestion: Check network connection, URL validity, or try again later"
                ) from e

        except URLError as e:
            retry_count += 1
            error_msg = f"Network error: {type(e).__name__} - {e}"
            logger.warning(f"Download failed (Retry {retry_count}/{max_retries}): {error_msg}")
            logger.debug(f"Next retry delay: {retry_delay:.1f} seconds")
            time.sleep(retry_delay)
            if retry_count >= max_retries:
                tmp_dst.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Network error: Retried {max_retries} times, still failed\n"
                    f"URL: {url}\n"
                    f"Error details: {error_msg}\n"
                    f"Suggestion: Check network connection, DNS resolution, or confirm URL accessibility"
                ) from e

        except (RuntimeError, FileExistsError) as e:
            tmp_dst.unlink(missing_ok=True)
            logger.error(f"Download failed: {e}")
            raise

        except PermissionError as e:
            tmp_dst.unlink(missing_ok=True)
            raise RuntimeError(f"Permission error: {e}\n  Suggestion: Check target path read/write permissions") from e

        except Exception as e:
            tmp_dst.unlink(missing_ok=True)
            logger.error(f"Unknown error: {type(e).__name__} - {e}")
            raise

        finally:
            if tmp_dst.exists() and (not allow_resume or retry_count >= max_retries):
                try:
                    tmp_dst.unlink(missing_ok=True)
                    logger.debug(f"Cleaned up temporary file: {tmp_dst}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {e} (Manual deletion recommended: {tmp_dst})")


def load_state_dict_from_url(
    url: str,
    model_dir: Optional[Union[str, Path]] = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    weights_only: bool = False,
) -> Dict[str, Any]:
    r"""
    Load serialized state dict from a URL.

    Supported auto-download, cache, zip extraction, and hash verification.

    Features:
    - auto generate cache to `model_dir` (default: <hub_dir>/checkpoints)
    - auto extract zip file (support legacy zip format, torch.save() <1.6)
    - hash check (download + cache) to ensure file integrity
    - cross-platform path compatibility (support str/Path input)
    - detailed logging for easy troubleshooting

    Args:
        url (str): URL of the serialized state dict to download.
        model_dir (Optional[Union[str, Path]]): Cache directory path. Default None (use <hub_dir>/checkpoints).
        map_location (MAP_LOCATION): Storage location mapping (refer to torch.load's map_location parameter). Default None.
        progress (bool): Whether to display download progress bar. Default True.
        check_hash (bool): Whether to verify file hash (filename must follow filename-<sha256-prefix>.ext format). Default False.
        file_name (Optional[str]): Custom download filename. Default None (use filename from URL).
        weights_only (bool): Whether to only load weights, reject complex serialized objects (recommended for untrusted sources). Default False.

    Returns:
        Dict[str, Any]: Loaded PyTorch state dict.

    Raises:
        ValueError: Invalid input (e.g., empty URL, hash format error)
        RuntimeError: Download failed, file corrupted, extract failed, hash check failed
        PermissionError: Directory has no write/read permission
        FileNotFoundError: Cache file not found and download failed
        torch.SerializationError: Serialized file corrupted, cannot load
    """

    if not url.strip():
        raise ValueError("URL is empty or whitespace-only")
    if check_hash and not HASH_REGEX:
        raise RuntimeError("HASH_REGEX must be specified, when check_hash=True")

    if model_dir is None:
        hub_dir = Path(get_dir())
        model_dir = hub_dir / "weights"
    else:
        model_dir = Path(model_dir).resolve()

    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError(f"Permission denied: Failed to create cache directory: {model_dir} - {e}") from e

    parsed_url = urlparse(url)
    default_filename = Path(parsed_url.path).name
    filename = file_name.strip() if (file_name and file_name.strip()) else default_filename
    if not filename:
        raise ValueError("Filename is empty or whitespace-only (file_name or URL path must contain a valid filename)")

    cached_file = model_dir / filename
    hash_prefix: Optional[str] = None

    if check_hash:
        hash_match = HASH_REGEX.search(filename)
        if not hash_match:
            raise ValueError(
                f"Filename must follow format: filename-<sha256-prefix>.ext\n"
                f"Current filename: {filename}\n"
                f"Example: resnet18-5c106cde.pth (hash prefix: 5c106cde)"
            )
        hash_prefix = hash_match.group(2)
        logger.debug(f"Parsed hash prefix: {hash_prefix} (from filename: {filename})")

    if not cached_file.exists():
        logger.info(f"Cache file not found, starting download...")
        download_url_to_file(
            url=url,
            dst=cached_file,
            hash_prefix=hash_prefix,
            progress=progress,
            timeout=15.0,
            max_retries=5,
        )
    else:
        logger.info(f"Cache file found: {cached_file}")
        if check_hash and hash_prefix:
            sha256 = hashlib.sha256()
            with open(cached_file, "rb") as f:
                while chunk := f.read(READ_DATA_CHUNK):
                    sha256.update(chunk)
            digest = sha256.hexdigest()
            if not digest.startswith(hash_prefix):
                raise RuntimeError(
                    f"Hash check failed for cached file!\n"
                    f"File path: {cached_file}\n"
                    f"Expected hash prefix: {hash_prefix}\n"
                    f"Actual hash: {digest}\n"
                    f"Please delete cache file ({cached_file}) and try again."
                )
            logger.info(f"Hash check passed for cached file: {digest[:len(hash_prefix)]}")

    load_path: Path = cached_file
    if zipfile.is_zipfile(cached_file):
        logger.info(f"Detected zip file, auto extract: {cached_file}")
        extract_dir = cached_file.with_suffix("")
        extract_dir.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(cached_file, "r") as zf:
                zf.extractall(extract_dir)

            pt_files = list(extract_dir.glob("*.pth")) + list(extract_dir.glob("*.pt"))
            if not pt_files:
                raise RuntimeError(
                    f"Zip file does not contain any PyTorch serialized files (.pth/.pt)\n"
                    f"Extract directory: {extract_dir}\n"
                    f"Zip contents: {[f.filename for f in zipfile.ZipFile(cached_file).infolist()]}"
                )
            load_path = pt_files[0]
            logger.info(f"Loaded first .pth/.pt file from zip: {load_path}")

        except zipfile.BadZipFile as e:
            shutil.rmtree(extract_dir, ignore_errors=True)
            raise RuntimeError(f"Zip file corrupted, extract failed: {cached_file} - {e}") from e
        except Exception as e:
            shutil.rmtree(extract_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to extract zip file: {cached_file} - {e}") from e

    try:
        logger.info(f"Loading state dict from: {load_path}")
        state_dict = torch.load(
            str(load_path),
            map_location=map_location,
            weights_only=weights_only,
        )

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Loaded file is not a state dict (expected dict, got {type(state_dict).__name__})\n"
                f"File path: {load_path}\n"
                f"Suggest: Check URL points to valid PyTorch state dict file"
            )

        logger.info(f"Loaded state dict from: {load_path} (contains {len(state_dict)} keys)")
        return state_dict

    except torch.SerializationError as e:
        raise RuntimeError(f"State dict file corrupted, failed to load: {load_path} - {e}\n") from e
    except RuntimeError as e:
        if "weights_only" in str(e).lower():
            raise RuntimeError(
                f"Failed to load state dict from: {load_path} (weights_only={weights_only})\n"
                f"Reason: File contains complex serialized objects, not trusted source.\n"
                f"Suggest keeping weights_only=True, or set to False at risk of loading."
            ) from e
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict from: {load_path} - {type(e).__name__} - {e}") from e