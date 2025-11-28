# ==============================================================================
# Copyright (c) 2025 zlx
# ==============================================================================
import torch
from typing import Union

from .weights import get_weights, get_model_list
from .models import _resnet, _dino_v2, _dino_v3, _mobile_mamba, _tresnet_v2

__all__ = ["BACKBONES", "load", "verify", "get_weights", "get_model_list"]

BACKBONES = [
    # ResNet
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d",
    "resnext101_32x8d", "resnext101_64x4d", "wide_resnet50_2", "wide_resnet101_2",
    # TResNet_v2
    "tresnet_l_v2",
    # DINO_v2
    "dinov2_vitb14", "dinov2_vitl14",
    # DINO_v2_reg
    "dinov2_vitb14_reg", "dinov2_vitl14_reg",
    # DINO_v3
    "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16",
    # MobileMamba
    "mobilemamba_t2", "mobilemamba_t2s", "mobilemamba_t4", "mobilemamba_t4s", "mobilemamba_s6", "mobilemamba_s6s",
    "mobilemamba_b1", "mobilemamba_b1s", "mobilemamba_b2", "mobilemamba_b2s", "mobilemamba_b4", "mobilemamba_b4s",
]


def verify(name: str):
    if name not in BACKBONES:
        raise ValueError(f"{name} be not found.\nPlease choose backbone from {BACKBONES}")


def load(
        name: str,
        pretrained: bool = True,
        progress: bool = True,
        device_map: Union[str, torch.device] = "auto",
        dtype: torch.dtype = torch.float32,
    ):
    """
    Load a backbone model by name.

    Args:
        name (str): The name of the model to load, must be in BACKBONES.
        pretrained (bool): Whether to load the pretrained weights. Defaults to True.
        progress (bool): Whether to show the progress of the download. Defaults to True.
        device_map (str, torch.device): The device map to use for loading the model. Defaults to "auto".
        dtype (torch.dtype): The torch dtype to use for loading the model. Defaults to torch.float32.
    """
    verify(name)

    if device_map == "auto":
        device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device_map, torch.device):
        raise ValueError(f"device_map must be 'auto' or a torch.device, but got {device_map}")
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"dtype must be a torch.dtype, but got {dtype}")
    if "tresnet" in name:
        return _tresnet_v2(name, pretrained, progress, device_map, dtype)
    if "resnet" in name or "xresnet" in name:
        return _resnet(name, pretrained, progress, device_map, dtype)
    elif "dino" in name:
        if "v2" in name:
            return _dino_v2(name, pretrained, progress, device_map, dtype)
        else:
            return _dino_v3(name, pretrained, progress, device_map, dtype)
    else:
        return _mobile_mamba(name, pretrained, progress, device_map, dtype)