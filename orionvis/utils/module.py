import torch
from typing import Union, Callable


def recursive_apply(tensors: Union[torch.Tensor, list, tuple, dict], func: Callable):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(recursive_apply(t, func) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: recursive_apply(v, func) for k, v in tensors.items()}
    elif isinstance(tensors, torch.Tensor):
        return func(tensors)
    else:
        raise TypeError(f"Unsupported type in `recursive_apply`: {type(tensors)}")