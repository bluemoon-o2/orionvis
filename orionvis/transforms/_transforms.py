from torchvision import transforms

from ..api.register import TransformT, register_transform, get_transform, TRANSFORM_LIST, unregister_transform

__all__ = ["TransformT", "register_transform", "get_transform", "TRANSFORM_LIST", "unregister_transform"]
# ==============================================================================
# Registered Transform Components
# ==============================================================================
@register_transform("ImageClassification")
def image_classification(crop_size: int = 224, resize_size: int = 256, normalize: bool = True) -> transforms.Compose:
    """
    Standard preprocessing pipeline for image classification tasks (compatible with ImageNet-pretrained orionvis).
    Input: PIL Image / NumPy ndarray / PyTorch Tensor
    Output: Normalized PyTorch Tensor (shape: [3, crop_size, crop_size], range: [-1, 1] approx)

    Args:
        crop_size: Final size of the center-cropped image (default: 224, standard for ImageNet)
        resize_size: Size to resize the image before center cropping (default: 256, follows torchvision best practices)

    Returns:
        transforms.Compose: Composed transform pipeline with the following steps:
            1. Resize: Scale image to (resize_size, resize_size)
            2. CenterCrop: Extract center region of size (crop_size, crop_size)
            3. ToTensor: Convert to PyTorch Tensor (HWC → CHW, uint8 → float32, scale [0,255] → [0,1])
            4. Normalize: Standardize using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """
    ops = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)
