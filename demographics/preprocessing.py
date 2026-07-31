"""MiVOLO-compatible RGB body-crop preprocessing."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from .exceptions import DemographicInputError

_INPUT_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def frame_image(image: Any, frame_id: str) -> np.ndarray:
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
    left = max(0, min(width, math.floor(float(bbox["x1"]))))
    top = max(0, min(height, math.floor(float(bbox["y1"]))))
    right = max(0, min(width, math.ceil(float(bbox["x2"]))))
    bottom = max(0, min(height, math.ceil(float(bbox["y2"]))))
    if right <= left or bottom <= top:
        raise DemographicInputError(
            "Body crop has zero area after clipping: "
            f"track_id={track_id} frame_id={frame_id} bbox={bbox} image_width={width} image_height={height}"
        )
    return np.ascontiguousarray(image[top:bottom, left:right])


def _letterbox_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    scale = min(_INPUT_SIZE / height, _INPUT_SIZE / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = image
    if (width, height) != (resized_width, resized_height):
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    dw = (_INPUT_SIZE - resized_width) / 2
    dh = (_INPUT_SIZE - resized_height) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _normalise_rgb(image: np.ndarray) -> np.ndarray:
    scaled = image.astype(np.float32) / 255.0
    return (scaled - _IMAGENET_MEAN) / _IMAGENET_STD


@lru_cache(maxsize=1)
def _missing_face_tensor() -> np.ndarray:
    black_rgb = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
    return np.ascontiguousarray(_normalise_rgb(black_rgb).transpose(2, 0, 1), dtype=np.float32)


def mivolo_input_from_body_crop(crop_rgb: np.ndarray) -> np.ndarray:
    if not isinstance(crop_rgb, np.ndarray) or crop_rgb.dtype != np.uint8:
        raise DemographicInputError("Body crop must be a uint8 NumPy array")
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise DemographicInputError("Body crop must have shape H x W x 3")
    if crop_rgb.shape[0] <= 0 or crop_rgb.shape[1] <= 0:
        raise DemographicInputError("Body crop must have positive dimensions")
    body_hwc = _normalise_rgb(_letterbox_rgb(np.ascontiguousarray(crop_rgb)))
    body_chw = np.ascontiguousarray(body_hwc.transpose(2, 0, 1), dtype=np.float32)
    model_input = np.concatenate((_missing_face_tensor(), body_chw), axis=0).astype(np.float32, copy=False)
    if model_input.shape != (6, _INPUT_SIZE, _INPUT_SIZE):
        raise DemographicInputError(f"MiVOLO input shape must be 6 x 224 x 224, got {model_input.shape}")
    return np.ascontiguousarray(model_input)


def stack_mivolo_inputs(inputs: list[np.ndarray]) -> np.ndarray:
    if not inputs:
        return np.empty((0, 6, _INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
    return np.ascontiguousarray(np.stack(inputs, axis=0).astype(np.float32, copy=False))
