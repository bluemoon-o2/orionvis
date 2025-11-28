import os
import logging
import threading
from typing import Optional
from lazy_object_proxy import Proxy
from gdown import download as gdown_load
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from ..hub import get_dir, download_url_to_file
from ..api import WeightsRegistry, Weight


def _extract_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    filename = os.path.basename(path)
    return filename

def _create_registry():
    with _lock:
        return WeightsRegistry()

_lock = threading.Lock()
_registry_proxy: WeightsRegistry = Proxy(lambda: _create_registry())


def verify(model: str) -> None:
    models = _registry_proxy.list_models()
    if model not in models:
        raise ValueError(f"{model} be not found. Please choose backbone from {models}")


def get_weights(model: str) -> Weight:
    verify(model)
    return _registry_proxy.get(model)


def get_model_list() -> list[str]:
    return _registry_proxy.list_models()


def ensure_gdown_checkpoint(model: str, version: Optional[str] = None, progress: bool = True) -> str:
    w = _registry_proxy.get(model, version)
    filename = f"{model}.pth"
    save_path = os.path.join(get_dir(), "weights", filename)
    if not os.path.exists(save_path):
        gdown_load(url=w.url, output=save_path, quiet=not progress)
    else:
        logger.info(f"Using cache found in: {save_path}")
    return save_path


def ensure_checkpoint(model: str, version: Optional[str] = None, progress: bool = True) -> str:
    verify(model)
    w = _registry_proxy.get(model, version)
    filename = _extract_filename_from_url(w.url)
    save_path = os.path.join(get_dir(), "weights", filename)
    if not os.path.exists(save_path):
        download_url_to_file(url=w.url, dst=save_path, progress=progress)
    else:
        logger.info(f"Using cache found in: {save_path}")
    return save_path
