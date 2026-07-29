"""Production Demographic stage using body-only MiVOLO-compatible inference."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import DemographicInputError
from .model import MiVOLOBackend
from .preprocessing import body_crop_to_mivolo_input, crop_body, validate_frame_image

BBox = dict[str, float]


def _require_mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise DemographicInputError(f"{name} must be an object")
    return value


def _require_fields(value: Mapping, fields: tuple[str, ...], name: str) -> None:
    for field in fields:
        if field not in value:
            raise DemographicInputError(f"Missing required {name} field: {field}")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise DemographicInputError(f"{name} must be finite")
    return float(value)


def _validate_bbox(value: Any, name: str) -> BBox:
    bbox = _require_mapping(value, name)
    _require_fields(bbox, ("x1", "y1", "x2", "y2"), name)
    copied = {axis: _finite_number(bbox[axis], f"{name}.{axis}") for axis in ("x1", "y1", "x2", "y2")}
    if copied["x2"] <= copied["x1"] or copied["y2"] <= copied["y1"]:
        raise DemographicInputError(f"{name} must have positive area")
    return copied


def _validate_event_batch(event_batch: Any) -> list[dict[str, Any]]:
    batch = _require_mapping(event_batch, "EventBatch")
    _require_fields(batch, ("events",), "EventBatch")
    if not isinstance(batch["events"], list):
        raise DemographicInputError("EventBatch.events must be a list")
    events: list[dict[str, Any]] = []
    for index, event_value in enumerate(batch["events"]):
        name = f"EventBatch.events[{index}]"
        event = _require_mapping(event_value, name)
        _require_fields(event, ("track_id", "timestamp", "event_type", "best_crop"), name)
        track_id = event["track_id"]
        if not isinstance(track_id, str) or not track_id:
            raise DemographicInputError(f"{name}.track_id must be a non-empty string")
        timestamp = _finite_number(event["timestamp"], f"{name}.timestamp")
        event_type = event["event_type"]
        if event_type not in (0, 1) or isinstance(event_type, bool):
            raise DemographicInputError(f"{name}.event_type must be exactly 0 or 1")
        best_crop = _require_mapping(event["best_crop"], f"{name}.best_crop")
        _require_fields(best_crop, ("frame_id", "bbox"), f"{name}.best_crop")
        frame_id = best_crop["frame_id"]
        if not isinstance(frame_id, str) or not frame_id:
            raise DemographicInputError(f"{name}.best_crop.frame_id must be a non-empty string")
        bbox = _validate_bbox(best_crop["bbox"], f"{name}.best_crop.bbox")
        events.append(
            {
                "track_id": track_id,
                "timestamp": timestamp,
                "event_type": int(event_type),
                "best_crop": {"frame_id": frame_id, "bbox": bbox},
            }
        )
    return events


def _validate_frame_batch(frame_batch: Any) -> dict[str, Mapping[str, Any]]:
    batch = _require_mapping(frame_batch, "FrameBatch")
    _require_fields(batch, ("frames",), "FrameBatch")
    if not isinstance(batch["frames"], list):
        raise DemographicInputError("FrameBatch.frames must be a list")
    frames_by_id: dict[str, Mapping[str, Any]] = {}
    for index, frame_value in enumerate(batch["frames"]):
        name = f"FrameBatch.frames[{index}]"
        frame = _require_mapping(frame_value, name)
        _require_fields(frame, ("frame_id", "timestamp", "image"), name)
        frame_id = frame["frame_id"]
        if not isinstance(frame_id, str) or not frame_id:
            raise DemographicInputError(f"{name}.frame_id must be a non-empty string")
        if frame_id in frames_by_id:
            raise DemographicInputError(f"Duplicate frame_id in FrameBatch: {frame_id}")
        _finite_number(frame["timestamp"], f"{name}.timestamp")
        validate_frame_image(frame["image"], frame_id)
        frames_by_id[frame_id] = frame
    return frames_by_id


def _group_selected_crops(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_track: dict[str, dict[str, Any]] = {}
    for event in events:
        track_id = event["track_id"]
        crop = event["best_crop"]
        if track_id not in by_track:
            by_track[track_id] = {
                "track_id": track_id,
                "timestamp": event["timestamp"],
                "best_crop": crop,
            }
            continue
        selected = by_track[track_id]
        selected["timestamp"] = min(float(selected["timestamp"]), float(event["timestamp"]))
        if selected["best_crop"] != crop:
            raise DemographicInputError(
                f"Conflicting best_crop records for track_id={track_id}; EventBatch has no crop-quality field"
            )
    return sorted(by_track.values(), key=lambda item: (float(item["timestamp"]), str(item["track_id"])))


class Demographic:
    """Callable Demographic stage.

    The model backend is lazy: empty EventBatch validation returns immediately and
    never resolves, hashes, or loads the checkpoint.
    """

    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "auto", backend: Any | None = None) -> None:
        self._backend = backend if backend is not None else MiVOLOBackend(checkpoint_path=checkpoint_path, device=device)

    def __call__(self, event_batch: dict, frame_batch: dict) -> dict:
        events = _validate_event_batch(event_batch)
        if not events:
            return {"results": []}

        frames_by_id = _validate_frame_batch(frame_batch)
        selected_crops = _group_selected_crops(events)
        tensors: list[np.ndarray] = []
        track_ids: list[str] = []
        for selected in selected_crops:
            track_id = selected["track_id"]
            best_crop = selected["best_crop"]
            frame_id = best_crop["frame_id"]
            bbox = best_crop["bbox"]
            if frame_id not in frames_by_id:
                raise DemographicInputError(f"Missing source frame: track_id={track_id} frame_id={frame_id} bbox={bbox}")
            frame = frames_by_id[frame_id]
            image = validate_frame_image(frame["image"], frame_id)
            crop = crop_body(image, bbox, track_id, frame_id)
            tensors.append(body_crop_to_mivolo_input(crop))
            track_ids.append(track_id)

        batch = np.ascontiguousarray(np.stack(tensors, axis=0).astype(np.float32, copy=False))
        predictions = self._backend.predict(batch)
        if len(predictions) != len(track_ids):
            raise DemographicInputError("Demographic backend returned an unexpected number of results")
        return {
            "results": [
                {"track_id": track_id, "age": int(prediction["age"]), "sex": int(prediction["sex"])}
                for track_id, prediction in zip(track_ids, predictions, strict=True)
            ]
        }
