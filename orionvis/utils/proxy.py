import threading
from typing import Union
from lazy_object_proxy import Proxy

from .log import XFormersConfig

_lock = threading.Lock()
_instances = {}


def get_real_instance(obj: Union[Proxy, XFormersConfig]) -> XFormersConfig:
    if hasattr(obj, '_wrapped'):
        return obj._wrapped  # noqa
    return obj


def _create_xformers_config():
    with _lock:
        if "xformers_config" not in _instances:
            _instances["xformers_config"] = XFormersConfig()
        return _instances["xformers_config"]


XFORMER_CONFIG: XFormersConfig = Proxy(_create_xformers_config)
