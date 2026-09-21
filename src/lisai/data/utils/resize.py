from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np
import torch


def crop_center(img: Union[np.ndarray, torch.Tensor], crop_size: Union[int, Tuple[int, int]]):
    """
    Center-crop any array/tensor of dim >= 2. If not perfectly centered, off-centered to top-left.
    """
    if isinstance(crop_size, tuple):
        crop_h, crop_w = crop_size
    elif isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        raise TypeError("crop_size should be tuple or integer.")
    
    img_h, img_w = img.shape[-2:]

    if crop_h > img_h or crop_w > img_w:
        raise ValueError(
            f"crop_size {(crop_h, crop_w)} exceeds image size {(img_h, img_w)}"
        )

    start_h = math.floor(img_h / 2 - (crop_h / 2))
    start_w = math.floor(img_w / 2 - (crop_w / 2))
    stop_h = start_h + crop_h
    stop_w = start_w + crop_w
    return img[..., start_h:stop_h, start_w:stop_w]


def center_pad(
    img: Union[np.ndarray, torch.Tensor],
    pad_size: Tuple[int, int] | None = None,
    target_size: Tuple[int, int] | None = None,
):
    """
    Center padding with zeros in last 2 directions. If cannot be centered, off-centered to top-left.
    """
    if pad_size is None:
        if target_size is None:
            raise ValueError("if pad_size is None, target_size must be defined")
        if not isinstance(target_size, tuple) or len(target_size) != 2:
            raise ValueError("target_size should be a tuple: (height,width)")

        pad_size = (
            abs(target_size[-2] - img.shape[-2]),
            abs(target_size[-1] - img.shape[-1]),
        )
        if pad_size[0] == 0 and pad_size[1] == 0:
            return img
    else:
        if not isinstance(pad_size, tuple) or len(pad_size) != 2:
            raise ValueError("pad_size should be a tuple: (pad_h, pad_w)")

    padh = (math.floor(pad_size[0] / 2), math.ceil(pad_size[0] / 2))
    padw = (math.floor(pad_size[1] / 2), math.ceil(pad_size[1] / 2))

    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            pad_width = (padh, padw)
        elif len(img.shape) == 3:
            pad_width = ((0, 0), padh, padw)
        elif len(img.shape) == 4:
            pad_width = ((0, 0), (0, 0), padh, padw)
        elif len(img.shape) == 5:
            pad_width = ((0, 0), (0, 0), (0, 0), padh, padw)
        else:
            raise ValueError(f"Unsupported img ndim: {img.ndim}")
        return np.pad(img, pad_width=pad_width, mode="constant", constant_values=0)

    if isinstance(img, torch.Tensor):
        pad_width = (padw[0], padw[1], padh[0], padh[1])  # left,right,top,bottom
        return torch.nn.functional.pad(img, pad_width, mode="constant", value=0)

    raise TypeError(f"Unsupported type: {type(img)}")


def adjust_img_size(img: Union[np.ndarray, torch.Tensor], mltpl_of: int, mode: str):
    """
    Adjust size to a multiple of `mltpl_of` by cropping or padding.
    """
    if mode not in ["crop", "pad"]:
        raise ValueError(f"Expected mode to be 'crop' or 'pad', got {mode}")

    img_h, img_w = img.shape[-2:]
    new_h = (img_h // mltpl_of) * mltpl_of
    new_w = (img_w // mltpl_of) * mltpl_of

    if mode == "crop":
        return crop_center(img, crop_size=(new_h, new_w))

    pad_h = mltpl_of - img_h % mltpl_of if img_h % mltpl_of != 0 else 0
    pad_w = mltpl_of - img_w % mltpl_of if img_w % mltpl_of != 0 else 0
    return center_pad(img, pad_size=(pad_h, pad_w))


def adjust_pred_size(img: Union[np.ndarray, torch.Tensor], original_size: Tuple[int, int], upsamp: int):
    """
    Adjust img size to match original_size * upsamp with padding or cropping.
    """
    target_size = (original_size[0] * upsamp, original_size[1] * upsamp)
    if target_size[0] < img.shape[-2] or target_size[1] < img.shape[-1]:
        return crop_center(img, crop_size=target_size)
    if target_size[0] > img.shape[-2] or target_size[1] > img.shape[-1]:
        return center_pad(img, target_size=target_size)
    return img
