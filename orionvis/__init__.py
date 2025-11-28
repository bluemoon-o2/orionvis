"""
Copyright (c) 2025 zlx. All rights reserved.

Licensed under the Apache License 2.0

orionvis: A PyTorch‑based Computer Vision toolkit

Overview
- Unified backbone loading for common backbone models (e.g., ResNet, DINO, MobileMamba).
- YAML‑based weight management with lazy loading and cached index.
- Registered preprocessing transforms for classification‑ready pipelines.
- Feature extraction and utility helpers.

Quickstart
    >>> import orionvis
    >>> model = orionvis.load("wide_resnet50_2")
    >>> weight = orionvis.get_weights("wide_resnet50_2")
    >>> transform = orionvis.get_transform("wide_resnet50_2")

Key APIs
- `load(name, pretrained=True, progress=True, device_map="auto", dtype=torch.float32)`
- `get_weights(model)`
- `get_model_list()` / `get_transform(model)`
"""
import logging
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API*")

# Logger Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)

logging.getLogger("xformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="xFormers is available*")

# Proper order of imports to avoid circular imports
from .transforms import *
from .backbones import *
from .features import *
from .utils import *

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"
    warnings.warn(f"Attention: version meta is missing.")
