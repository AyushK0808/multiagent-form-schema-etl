"""
finetune/augmentation.py
========================
Dataset-specific Albumentations image augmentation for the training split.
Gracefully degrades to a no-op if albumentations is not installed.

Install with:
    pip install albumentations
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    _ALBUMENTATIONS_OK = True
except ImportError:
    _ALBUMENTATIONS_OK = False
    logger.info(
        "albumentations not installed - training without image augmentation. "
        "Install with: pip install albumentations"
    )


AUGMENTATION_VERSION = 2


def _identity(image: Image.Image) -> Image.Image:
    return image


def _build_pipeline(transforms):
    if not _ALBUMENTATIONS_OK:
        return None
    return A.Compose(transforms)


def _apply_pipeline(image: Image.Image, pipeline) -> Image.Image:
    if pipeline is None:
        return image
    arr = np.array(image.convert("RGB"))
    return Image.fromarray(pipeline(image=arr)["image"])


_RECEIPT_PIPELINE = _build_pipeline([
    A.Rotate(limit=2, border_mode=0, p=0.35),
    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.45),
    A.ImageCompression(quality_range=(65, 95), p=0.35),
    A.GaussNoise(std_range=(0.01, 0.04), p=0.15),
])

_FORM_PIPELINE = _build_pipeline([
    A.Rotate(limit=3, border_mode=0, p=0.45),
    A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.10, p=0.40),
    A.GaussianBlur(blur_limit=(3, 5), p=0.10),
    A.ImageCompression(quality_range=(70, 95), p=0.25),
])

_LAYOUT_PIPELINE = _build_pipeline([
    A.Rotate(limit=3, border_mode=0, p=0.40),
    A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.10, p=0.35),
    A.ImageCompression(quality_range=(70, 95), p=0.20),
])

_REASONING_PIPELINE = _build_pipeline([
    A.Rotate(limit=2, border_mode=0, p=0.30),
    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.08, p=0.35),
    A.ImageCompression(quality_range=(75, 95), p=0.20),
])


def augment_cord(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _RECEIPT_PIPELINE)


def augment_sroie(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _RECEIPT_PIPELINE)


def augment_synthdog_en(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _RECEIPT_PIPELINE)


def augment_funsd(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _FORM_PIPELINE)


def augment_docbank(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _LAYOUT_PIPELINE)


def augment_doclaynet(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _LAYOUT_PIPELINE)


def augment_rvl_cdip(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _LAYOUT_PIPELINE)


def augment_docvqa(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _REASONING_PIPELINE)


def augment_kleister_nda(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _REASONING_PIPELINE)


def augment_infographicvqa(image: Image.Image) -> Image.Image:
    return _apply_pipeline(image, _REASONING_PIPELINE)


AUGMENTORS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "CORD": augment_cord,
    "DOCBANK": augment_docbank,
    "DOCLAYNET": augment_doclaynet,
    "DOCVQA": augment_docvqa,
    "FUNSD": augment_funsd,
    "INFOGRAPHICVQA": augment_infographicvqa,
    "KLEISTER_NDA": augment_kleister_nda,
    "RVL-CDIP": augment_rvl_cdip,
    "SROIE": augment_sroie,
    "SYNTHDOG_EN": augment_synthdog_en,
}


augmentation_available: bool = _ALBUMENTATIONS_OK


def augment_image(image: Image.Image, dataset_name: str) -> Image.Image:
    """Apply dataset-specific augmentation; no-op if albumentations is absent."""
    augmentor = AUGMENTORS.get(dataset_name, _identity)
    return augmentor(image)
