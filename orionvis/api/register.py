import logging
from typing import Callable, TypeVar, Union, Optional, List

from ._register import Registry

logger = logging.getLogger(__name__)

try:
    import albumentations
    from torchvision import transforms
    ComposeType = Union[transforms.Compose, albumentations.Compose]
except ImportError:
    from torchvision import transforms
    ComposeType = transforms.Compose
    logger.info(f"albumentations is missing, transforms can only be registered from torchvision.")


TransformT = TypeVar("TransformT", bound=Callable[..., ComposeType])
TRANSFORM_REGISTRY = Registry[TransformT](name="Transform", expose_module="orionvis.transforms", allow_override=False)

TRANSFORM_LIST: List[str] = TRANSFORM_REGISTRY.list
register_transform: Callable[[Optional[str], Optional[bool]], Callable[[TransformT], TransformT]] = TRANSFORM_REGISTRY.register
get_transform: Callable[[str], TransformT] = TRANSFORM_REGISTRY.get
unregister_transform: Callable[[str], None] = TRANSFORM_REGISTRY.unregister
