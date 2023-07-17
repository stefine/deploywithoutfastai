import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from PIL import Image
import typing
import math


def crop(image: Image.Image, size: typing.Tuple[int, int]) -> Image:

    h, w, c_h, c_w = image.size[-2], image.size[-1], size[-2], size[-1]
    top = (h - c_h) // 2
    left = (w - c_w) // 2

    top = max(top, 0)
    left = max(left, 0)

    right = min(top + c_h, h)
    bottom = min(left + c_w, w)
    return image.crop((left, top, bottom, right))


def pad(image: Image.Image, size: typing.Tuple[int, int]) -> Image:
    h, w, p_h, p_w = image.size[-2], image.size[-1], size[0], size[1]
    pad_h = (h - p_h) // 2
    pad_w = (w - p_w) // 2

    pad_h = max(-pad_h, 0)
    pad_w = max(-pad_w, 0)

    pad_hr = max(p_h - pad_h - h, 0)
    pad_wr = max(p_w - pad_w - w, 0)

    return tvf.pad(
        image,
        [pad_w, pad_h, pad_wr, pad_hr],
        padding_mode="constant"
    )


def resized_crop_pad(
    image: typing.Union[Image.Image, torch.tensor],
    size: typing.Tuple[int, int],
    extra_crop_ratio: float = 0.14,
) -> Image:

    maximum_space = max(size[0], size[1])
    extra_space = maximum_space * extra_crop_ratio
    extra_space = math.ceil(extra_space / 8) * 8
    extended_size = (size[0] + extra_space, size[1] + extra_space)
    resized_image = image.resize(extended_size, resample=Image.BILINEAR)

    if extended_size != size:
        resized_image = pad(crop(resized_image, size), size)

    return resized_image


def gpu_crop(
    batch: torch.tensor,
    size: typing.Tuple[int, int]
):
    """
    Crops each image in `batch` to a particular `size`.

    Args:
        batch (array of `torch.Tensor`):
            A batch of images, should be of shape `NxCxWxH`
        size (`tuple` of integers):
            A size to pad to, should be in the form
            of (width, height)

    Returns:
        A batch of cropped images
    """
    # Split into multiple lines for clarity
    affine_matrix = torch.eye(3, device=batch.device).float()
    affine_matrix = affine_matrix.unsqueeze(0)
    affine_matrix = affine_matrix.expand(batch.size(0), 3, 3)
    affine_matrix = affine_matrix.contiguous()[:, :2]

    coords = F.affine_grid(
        affine_matrix, batch.shape[:2] + size, align_corners=True
    )

    top_range, bottom_range = coords.min(), coords.max()
    zoom = 1/(bottom_range - top_range).item()*2

    resizing_limit = min(
        batch.shape[-2]/coords.shape[-2],
        batch.shape[-1]/coords.shape[-1]
    )/2

    if resizing_limit > 1 and resizing_limit > zoom:
        batch = F.interpolate(
            batch,
            scale_factor=1/resizing_limit,
            mode='area',
            recompute_scale_factor=True
        )
    return F.grid_sample(batch,
                         coords,
                         mode='bilinear',
                         padding_mode='reflection',
                         align_corners=True)
