"""
finetune/augmentation.py
========================
Optional Albumentations image augmentation for the training split.
Gracefully degrades to a no-op if albumentations is not installed.

Install with:
    pip install albumentations
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    _ALBUMENTATIONS_OK = True
except ImportError:
    _ALBUMENTATIONS_OK = False
    logger.info(
        "albumentations not installed — training without image augmentation. "
        "Install with: pip install albumentations"
    )


def _build_aug_pipeline():
    if not _ALBUMENTATIONS_OK:
        return None
    return A.Compose([
        A.Rotate(limit=3, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.10, p=0.5),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ])


_AUG_PIPELINE = _build_aug_pipeline()

# Public flag so callers can check availability before setting --augment
augmentation_available: bool = _ALBUMENTATIONS_OK


def augment_image(image: Image.Image) -> Image.Image:
    """Apply training augmentation to a PIL image; no-op if albumentations absent."""
    if _AUG_PIPELINE is None:
        return image
    arr = np.array(image.convert("RGB"))
    return Image.fromarray(_AUG_PIPELINE(image=arr)["image"])