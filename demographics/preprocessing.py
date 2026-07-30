"""Preprocessing helpers for body-only MiVOLO demographic inference.

MiVOLO upstream preprocessing (`mivolo.data.misc.prepare_classification_images`)
letterboxes each crop to the model input size with black padding, bilinear
resize, RGB ImageNet Z-score normalization, CHW tensor layout, and a normalized
black image when a face crop is absent. This package receives RGB frames from the
pipeline, so it intentionally does not apply the upstream BGR-to-RGB conversion a
second time.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .exceptions import DemographicInputError

INPUT_SIZE = 224
IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _crop_context(track_id: str, frame_id: str, bbox: dict[str, float]) -> str:
    return f"track_id={track_id} frame_id={frame_id} bbox={bbox}"


def validate_frame_image(image: Any, frame_id: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise DemographicInputError(f"Frame {frame_id} image must be a NumPy array")
    if image.dtype != np.uint8:
        raise DemographicInputError(f"Frame {frame_id} image must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise DemographicInputError(f"Frame {frame_id} image must have shape H x W x 3")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise DemographicInputError(f"Frame {frame_id} image must have positive dimensions")
    return image


def crop_body(image: np.ndarray, bbox: dict[str, float], track_id: str, frame_id: str) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = max(0, min(width, math.floor(float(bbox["x1"]))))
    y1 = max(0, min(height, math.floor(float(bbox["y1"]))))
    x2 = max(0, min(width, math.ceil(float(bbox["x2"]))))
    y2 = max(0, min(height, math.ceil(float(bbox["y2"]))))
    if x2 <= x1 or y2 <= y1:
        raise DemographicInputError(
            f"Body crop has zero area after clipping: {_crop_context(track_id, frame_id, bbox)}"
        )
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def letterbox_rgb(image: np.ndarray, target_size: int = INPUT_SIZE) -> np.ndarray:
    """Match upstream MiVOLO `class_letterbox` for RGB images."""

    import cv2

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise DemographicInputError("Crop image must have positive dimensions")
    if height == target_size and width == target_size:
        return image
    scale = min(target_size / height, target_size / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = image
    if (width, height) != (resized_width, resized_height):
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    dw = (target_size - resized_width) / 2
    dh = (target_size - resized_height) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _normalise_rgb(image: np.ndarray) -> np.ndarray:
    prepared = image.astype(np.float32) / 255.0
    return (prepared - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD


def _missing_face_tensor() -> np.ndarray:
    black_rgb = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    return np.ascontiguousarray(_normalise_rgb(black_rgb).transpose(2, 0, 1), dtype=np.float32)


def body_crop_to_mivolo_input(crop_rgb: np.ndarray) -> np.ndarray:
    if not isinstance(crop_rgb, np.ndarray) or crop_rgb.dtype != np.uint8:
        raise DemographicInputError("Body crop must be a uint8 NumPy array")
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise DemographicInputError("Body crop must have shape H x W x 3")
    body_hwc = _normalise_rgb(letterbox_rgb(np.ascontiguousarray(crop_rgb), INPUT_SIZE))
    body_chw = np.ascontiguousarray(body_hwc.transpose(2, 0, 1), dtype=np.float32)
    model_input = np.concatenate([_missing_face_tensor(), body_chw], axis=0).astype(np.float32, copy=False)
    if model_input.shape != (6, INPUT_SIZE, INPUT_SIZE):
        raise DemographicInputError(f"MiVOLO input shape must be 6 x 224 x 224, got {model_input.shape}")
    return np.ascontiguousarray(model_input)
