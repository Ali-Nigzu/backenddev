"""Preprocessing helpers for body-only MiVOLO demographic inference."""

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
    import cv2

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise DemographicInputError("Crop image must have positive dimensions")
    scale = min(target_size / height, target_size / width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top = (target_size - resized_height) // 2
    left = (target_size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def body_crop_to_mivolo_input(crop_rgb: np.ndarray) -> np.ndarray:
    if not isinstance(crop_rgb, np.ndarray) or crop_rgb.dtype != np.uint8:
        raise DemographicInputError("Body crop must be a uint8 NumPy array")
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise DemographicInputError("Body crop must have shape H x W x 3")
    prepared = letterbox_rgb(np.ascontiguousarray(crop_rgb), INPUT_SIZE).astype(np.float32) / 255.0
    prepared = (prepared - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
    body_chw = np.ascontiguousarray(prepared.transpose(2, 0, 1), dtype=np.float32)
    zero_face = np.zeros_like(body_chw, dtype=np.float32)
    model_input = np.concatenate([zero_face, body_chw], axis=0).astype(np.float32, copy=False)
    if model_input.shape != (6, INPUT_SIZE, INPUT_SIZE):
        raise DemographicInputError(f"MiVOLO input shape must be 6 x 224 x 224, got {model_input.shape}")
    return np.ascontiguousarray(model_input)
