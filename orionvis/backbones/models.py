import os
import torch
import logging
from torch.hub import load as hub_load
from importlib import import_module
from typing import Union

from ..utils import suppress_output
from .weights import ensure_gdown_checkpoint

logger = logging.getLogger(__name__)


_MobileMamba_Weights = {
    "mobilemamba_t2" : "1LXh5wjVEUu5Kj1foclC9zmlKsy7JpMQ-",
    "mobilemamba_t2s" : "14UUeEeN62a1jo5qit-m0yChQnNU8JWQG",
    "mobilemamba_t4" : "17-5Q93_cEhBYMDzWTq4sDcc4_c_ezwpd",
    "mobilemamba_t4s" : "12e1RKJEYDWVjOZuQLycjOUkA5PqCUnwu",
    "mobilemamba_s6" : "13en_zNB2wuHmnHKZMUQDe4KizlaTCHbP",
    "mobilemamba_s6s" : "1ujVeWpoICjioRF0aB7PD1fadM4PGFRfl",
    "mobilemamba_b1" : "1ODO1XM2luMfczbDLJxHrEl-ie7yUscAC",
    "mobilemamba_b1s" : "15rBhYky_f7jNSknXujuwQESaIkiTBAR1",
    "mobilemamba_b2" : "1tJJZqeP09D_IAKKFuQtJQOYm4eNGjr1r",
    "mobilemamba_b2s" : "1KuleL6Fi84zkjIhkRFbR4lqGa2iu6bOp",
    "mobilemamba_b4" : "1MiGLHSldK2JpbQEe-7VlmFsaCdugDnYR",
    "mobilemamba_b4s" : "1KevdesGm8Pb4fEbgZ5JY4FMy9IQgYbiZ",
}


def _resnet(
        arch: str,
        pretrained: bool,
        progress: bool,
        device_map: Union[str, torch.device] = "auto",
        dtype: torch.dtype = torch.float32
):
    models = import_module("torchvision.models")
    backbone_class = getattr(models, arch)
    weights = "IMAGENET1K_V2" if pretrained else None
    return backbone_class(weights=weights, progress=progress).to(dtype=dtype, device=device_map)


def _dino_v2(
        arch: str,
        pretrained: bool,
        progress: bool,
        device_map: Union[str, torch.device] = "auto",
        dtype: torch.dtype = torch.float32,
):
    with suppress_output(stderr=not progress):
        return hub_load(f"facebookresearch/dinov2", arch, pretrained=pretrained).to(dtype=dtype, device=device_map)


def _dino_v3(
        arch: str,
        pretrained: bool,
        progress: bool,
        device_map: Union[str, torch.device] = "auto",
        dtype: torch.dtype = torch.float32,
):
    with suppress_output(stderr=not progress):
        return hub_load(f"facebookresearch/dinov3", arch, pretrained=pretrained).to(dtype=dtype, device=device_map)


def _mobile_mamba(
        arch: str,
        pretrained: bool,
        progress: bool,
        device_map: Union[str, torch.device] = "auto",
        dtype: torch.dtype = torch.float32,
        **kwargs,
):
    arches = {
        "mobilemamba_t2": "MobileMamba_T2", "mobilemamba_t2s": "MobileMamba_T2",
        "mobilemamba_t4": "MobileMamba_T4", "mobilemamba_t4s": "MobileMamba_T4",
        "mobilemamba_s6": "MobileMamba_S6", "mobilemamba_s6s": "MobileMamba_S6",
        "mobilemamba_b1": "MobileMamba_B1", "mobilemamba_b1s": "MobileMamba_B1",
        "mobilemamba_b2": "MobileMamba_B2", "mobilemamba_b2s": "MobileMamba_B2",
        "mobilemamba_b4": "MobileMamba_B4", "mobilemamba_b4s": "MobileMamba_B4",
    }
    save_path = None
    num_classes = kwargs.pop('num_classes', 1000)
    distillation = kwargs.pop('distillation', False)
    fuse = kwargs.pop('fuse', False)
    ema = kwargs.pop('ema', False)
    strict = kwargs.pop('strict', True)

    if pretrained:
        save_path = ensure_gdown_checkpoint(arch, progress=progress)

    cfg_model = {
        "name": arches[arch],
        "model_kwargs": {
            "num_classes": num_classes,
            "distillation": distillation,
            "fuse": fuse,
            "checkpoint_path": save_path,
            "ema": ema,
            "strict": strict,
            "device_map": device_map,
            "dtype": dtype,
        }
    }
    with suppress_output(stdout=not progress, stderr=not progress):
        from .mamba_mobile import get_model
        return get_model(cfg_model)
