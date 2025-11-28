# ==============================================================================
# Copyright (c) 2025 zlx
# ==============================================================================
from ._transforms import TransformT, register_transform, get_transform, TRANSFORM_LIST, unregister_transform

__all__ = ["register_transform", "get_transform", "TRANSFORM_LIST", "unregister_transform"]
# ==============================================================================
# IDE Static Type Hint & Dynamic Attribute Resolution
# ==============================================================================
# Type variable bounded to Callables that return torchvision.transforms.Compose
# Ensures type consistency for registered transform components

def __getattr__(name: str) -> TransformT:
    """
    Dynamic attribute resolver for registered transform components.
    Fixes IDE's static analysis limitation (cannot detect runtime-added attributes).

    Args:
        name: Name of the registered transform component to retrieve
    """
    try:
        return get_transform(name)
    except KeyError as e:
        raise AttributeError(
            f"Module '{__name__}' has no attribute '{name}'. "
            f"Available registered transforms: {TRANSFORM_LIST}"
        ) from e