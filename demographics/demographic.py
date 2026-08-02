"""Production Demographic stage orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from contracts import FrameBatchError, build_frame_lookup

from .exceptions import DemographicInputError
from .model import _MiVOLOModelRunner

BBox = dict[str, float]


@dataclass(frozen=True)
class _CropDescriptor:
    track_id: str
    timestamp: float
    frame_id: str
    bbox: BBox


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DemographicInputError(f"{name} must be an object")
    return value


def _require_fields(value: Mapping[str, Any], fields: tuple[str, ...], name: str) -> None:
    for field in fields:
        if field not in value:
            raise DemographicInputError(f"Missing required {name} field: {field}")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise DemographicInputError(f"{name} must be a finite number")
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
        if isinstance(event_type, bool) or not isinstance(event_type, int) or event_type not in (0, 1):
            raise DemographicInputError(f"{name}.event_type must be exactly integer 0 or 1")
        best_crop = _require_mapping(event["best_crop"], f"{name}.best_crop")
        _require_fields(best_crop, ("frame_id", "bbox"), f"{name}.best_crop")
        frame_id = best_crop["frame_id"]
        if not isinstance(frame_id, str) or not frame_id:
            raise DemographicInputError(f"{name}.best_crop.frame_id must be a non-empty string")
        bbox = _validate_bbox(best_crop["bbox"], f"{name}.best_crop.bbox")
        events.append({"track_id": track_id, "timestamp": timestamp, "frame_id": frame_id, "bbox": bbox})
    return events


def _select_unique_tracks(events: list[dict[str, Any]]) -> list[_CropDescriptor]:
    by_track: dict[str, _CropDescriptor] = {}
    for event in events:
        descriptor = _CropDescriptor(
            track_id=event["track_id"],
            timestamp=event["timestamp"],
            frame_id=event["frame_id"],
            bbox=event["bbox"],
        )
        existing = by_track.get(descriptor.track_id)
        if existing is None:
            by_track[descriptor.track_id] = descriptor
            continue
        if existing.frame_id != descriptor.frame_id or existing.bbox != descriptor.bbox:
            raise DemographicInputError(
                "Conflicting best_crop records for "
                f"track_id={descriptor.track_id}; existing frame_id={existing.frame_id} bbox={existing.bbox}; "
                f"new frame_id={descriptor.frame_id} bbox={descriptor.bbox}"
            )
        if descriptor.timestamp < existing.timestamp:
            by_track[descriptor.track_id] = _CropDescriptor(
                track_id=existing.track_id,
                timestamp=descriptor.timestamp,
                frame_id=existing.frame_id,
                bbox=existing.bbox,
            )
    return sorted(by_track.values(), key=lambda item: (item.timestamp, item.track_id))


def _build_required_frame_lookup(frame_batch: Any, descriptors: list[_CropDescriptor]) -> dict[str, Mapping[str, Any]]:
    required = {descriptor.frame_id for descriptor in descriptors}
    try:
        frames_by_id = build_frame_lookup(frame_batch, required_ids=required)
    except FrameBatchError as exc:
        message = str(exc)
        if message.startswith("Missing frame_id in FrameBatch: "):
            missing_frame_id = message.rsplit(": ", 1)[1]
            for descriptor in descriptors:
                if descriptor.frame_id == missing_frame_id:
                    raise DemographicInputError(
                        f"Missing source frame: track_id={descriptor.track_id} "
                        f"frame_id={descriptor.frame_id} bbox={descriptor.bbox}"
                    ) from exc
        raise DemographicInputError(message) from exc
    return frames_by_id


class Demographic:
    """Callable production demographic stage."""

    def __init__(self) -> None:
        self._model = _MiVOLOModelRunner()

    def __call__(self, event_batch: Any, frame_batch: Any) -> dict[str, list[dict[str, int | str]]]:
        events = _validate_event_batch(event_batch)
        if not events:
            return {"results": []}

        descriptors = _select_unique_tracks(events)
        frames_by_id = _build_required_frame_lookup(frame_batch, descriptors)
        from .preprocessing import frame_image

        for descriptor in descriptors:
            frame_image(frames_by_id[descriptor.frame_id]["image"], descriptor.frame_id)
        predictions = self._model.predict(descriptors, frames_by_id)
        if len(predictions) != len(descriptors):
            raise DemographicInputError("Demographic model returned an unexpected number of results")
        return {
            "results": [
                {"track_id": descriptor.track_id, "age": int(prediction["age"]), "sex": int(prediction["sex"])}
                for descriptor, prediction in zip(descriptors, predictions, strict=True)
            ]
        }
