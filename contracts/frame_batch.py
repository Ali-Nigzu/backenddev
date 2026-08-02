"""Canonical FrameBatch contract validation."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import numpy as np

__all__ = ["FrameBatchError", "build_frame_lookup", "validate_frame_batch"]

_REQUIRED_FRAME_FIELDS = frozenset(("frame_id", "timestamp", "image"))
_REQUIRED_BATCH_FIELDS = frozenset(("frames",))


class FrameBatchError(ValueError):
    """Raised when a FrameBatch does not satisfy the canonical contract."""


def _validate_image(image: Any, frame_id: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise FrameBatchError(f"Frame {frame_id} image must be a NumPy array")
    if image.dtype != np.uint8:
        raise FrameBatchError(f"Frame {frame_id} image must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise FrameBatchError(f"Frame {frame_id} image must have shape H x W x 3")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise FrameBatchError(f"Frame {frame_id} image must have positive dimensions")
    if not image.flags.c_contiguous:
        raise FrameBatchError(f"Frame {frame_id} image must be C-contiguous")
    return image


def _validate_timestamp(timestamp: Any, frame_id: str) -> float:
    if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not isfinite(float(timestamp)):
        raise FrameBatchError(f"Frame {frame_id} timestamp must be finite")
    return float(timestamp)


def validate_frame_batch(frame_batch: Any) -> list[Mapping[str, Any]]:
    """Validate ``frame_batch`` and return the original frame mappings in input order.

    The validator preserves frame references and never copies image arrays.
    """

    if not isinstance(frame_batch, Mapping):
        raise FrameBatchError("FrameBatch must be a mapping")

    batch_fields = set(frame_batch.keys())
    missing_batch_fields = _REQUIRED_BATCH_FIELDS - batch_fields
    if missing_batch_fields:
        missing = sorted(missing_batch_fields)[0]
        raise FrameBatchError(f"Missing required FrameBatch field: {missing}")
    extra_batch_fields = batch_fields - _REQUIRED_BATCH_FIELDS
    if extra_batch_fields:
        extra = sorted(str(field) for field in extra_batch_fields)[0]
        raise FrameBatchError(f"Unexpected FrameBatch field: {extra}")

    frames = frame_batch["frames"]
    if not isinstance(frames, list):
        raise FrameBatchError("FrameBatch.frames must be a list")

    seen_frame_ids: set[str] = set()
    validated_frames: list[Mapping[str, Any]] = []
    for index, frame_value in enumerate(frames):
        if not isinstance(frame_value, Mapping):
            raise FrameBatchError(f"Frame at index {index} must be a mapping")

        frame_fields = set(frame_value.keys())
        missing_frame_fields = _REQUIRED_FRAME_FIELDS - frame_fields
        if missing_frame_fields:
            missing = sorted(missing_frame_fields)[0]
            raise FrameBatchError(f"Missing required FrameBatch.frames[{index}] field: {missing}")
        extra_frame_fields = frame_fields - _REQUIRED_FRAME_FIELDS
        if extra_frame_fields:
            extra = sorted(str(field) for field in extra_frame_fields)[0]
            raise FrameBatchError(f"Unexpected FrameBatch.frames[{index}] field: {extra}")

        frame_id = frame_value["frame_id"]
        if not isinstance(frame_id, str) or not frame_id:
            raise FrameBatchError(f"Frame at index {index} frame_id must be a non-empty string")
        if frame_id in seen_frame_ids:
            raise FrameBatchError(f"Duplicate frame_id: {frame_id}")
        seen_frame_ids.add(frame_id)

        _validate_timestamp(frame_value["timestamp"], frame_id)
        _validate_image(frame_value["image"], frame_id)
        validated_frames.append(frame_value)

    return validated_frames


def build_frame_lookup(frame_batch: Any, required_ids: set[str] | None = None) -> dict[str, Mapping[str, Any]]:
    """Validate a FrameBatch and return a frame_id lookup.

    ``required_ids`` may be supplied by integration stages to force clear missing-frame
    failures without scanning or reopening the source media.
    """

    frames = validate_frame_batch(frame_batch)
    frames_by_id = {str(frame["frame_id"]): frame for frame in frames}
    if required_ids is not None:
        for frame_id in sorted(required_ids):
            if frame_id not in frames_by_id:
                raise FrameBatchError(f"Missing frame_id in FrameBatch: {frame_id}")
    return frames_by_id
