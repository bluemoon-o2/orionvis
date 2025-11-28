import os
import logging
from typing import Optional
from lazy_object_proxy import Proxy
from gdown import download as gdown_load

logger = logging.getLogger(__name__)

from ..api import WeightsRegistry, Weight
import threading

CACHE_PATH = os.environ.get("ORIONVIS_CACHE", os.path.expanduser("~/.cache"))
GDOWN_PATH = os.path.join(CACHE_PATH, "gdown")

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
    verify(model)
    os.makedirs(GDOWN_PATH, exist_ok=True)
    w = _registry_proxy.get(model, version)
    filename = f"{model}.pth"
    save_path = os.path.join(GDOWN_PATH, filename)
    if not os.path.exists(save_path):
        gdown_load(url=w.url, output=save_path, quiet=not progress)
    else:
        logger.info(f"Using cache found in: {save_path}")
    return save_path
