import os
import sys
import logging
import warnings
import contextlib
from pathlib import Path
from typing import Union, Iterator

# xFormers Logger and Warning Configuration
_XFORMERS_LOGGER = logging.getLogger("xformers")
logger = logging.getLogger(__name__)

class XFormersConfig:
    def __init__(self):
        self._warning_enabled: bool = False
        self._log_level: int = logging.ERROR
        self._apply_warning_config()
        self._apply_log_config()

    @property
    def warning_enabled(self) -> bool:
        return self._warning_enabled

    @warning_enabled.setter
    def warning_enabled(self, value: bool) -> None:
        if self._warning_enabled == value:
            return
        self._warning_enabled = value
        self._apply_warning_config()

    @property
    def log_level(self) -> int:
        return self._log_level

    @log_level.setter
    def log_level(self, value: Union[int, str]) -> None:
        if isinstance(value, str):
            value = logging.getLevelName(value.upper())
        if not isinstance(value, int) or not (logging.DEBUG <= value <= logging.CRITICAL):
            raise ValueError(
                f"Invalid log level: {value}. "
                f"Please use logging module constants (e.g., logging.INFO) or strings (e.g., 'INFO')."
            )
        if self._log_level == value:
            return
        self._log_level = value
        self._apply_log_config()

    def _apply_warning_config(self) -> None:
        for idx in reversed(range(len(warnings.filters))):
            filter_rule = warnings.filters[idx]
            if filter_rule[2] == "xFormers is available*":
                warnings.filters.pop(idx)

        action = "default" if self._warning_enabled else "ignore"
        warnings.filterwarnings(action, message="xFormers is available*")

    def _apply_log_config(self) -> None:
        _XFORMERS_LOGGER.setLevel(self._log_level)


@contextlib.contextmanager
def suppress_output(stdout: bool = False, stderr: bool = True):
    path = 'nul' if os.name == 'nt' else '/dev/null'
    devnull = open(path, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        if stdout:
            sys.stdout = devnull
        if stderr:
            sys.stderr = devnull
        yield
    finally:
        if stdout:
            sys.stdout = old_stdout
        if stderr:
            sys.stderr = old_stderr
        devnull.close()


@contextlib.contextmanager
def sys_path(path: Union[str, Path]) -> Iterator[None]:
    abs_path = str(Path(path).resolve())
    added = False

    try:
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            added = True
        yield

    finally:
        if added and abs_path in sys.path:
            try:
                sys.path.remove(abs_path)
            except ValueError:
                logger.info(f"[sys_path exit] Path {abs_path} not found in sys.path, skipping removal.")
