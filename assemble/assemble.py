"""Production Event and demographic assembly stage."""

from collections.abc import Mapping
import hashlib
import json
from math import isfinite
from typing import Any


class AssembleInputError(ValueError):
    """Raised when an Assemble input contract is invalid."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssembleInputError(f"{name} must be an object")
    return value


def _require_fields(value: Mapping[str, Any], fields: tuple[str, ...], name: str) -> None:
    for field in fields:
        if field not in value:
            raise AssembleInputError(f"Missing required {name} field: {field}")


def _event_values(event_batch: Any) -> list[Any]:
    batch = _require_mapping(event_batch, "EventBatch")
    _require_fields(batch, ("events",), "EventBatch")
    events = batch["events"]
    if not isinstance(events, list):
        raise AssembleInputError("EventBatch.events must be a list")
    return events


def _index_demographics(demographics_batch: Any) -> dict[str, Mapping[str, Any]]:
    batch = _require_mapping(demographics_batch, "DemographicsBatch")
    _require_fields(batch, ("results",), "DemographicsBatch")
    results = batch["results"]
    if not isinstance(results, list):
        raise AssembleInputError("DemographicsBatch.results must be a list")

    by_track: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(results):
        name = f"DemographicsBatch.results[{index}]"
        result = _require_mapping(value, name)
        _require_fields(result, ("track_id", "age", "sex"), name)
        track_id = result["track_id"]
        if type(track_id) is not str or not track_id:
            raise AssembleInputError(f"{name}.track_id must be a non-empty string")
        age = result["age"]
        if type(age) is not int or age < 0:
            raise AssembleInputError(f"{name}.age must be a non-negative integer")
        sex = result["sex"]
        if type(sex) is not int or sex not in (0, 1):
            raise AssembleInputError(f"{name}.sex must be exactly integer 0 or 1")
        if track_id in by_track:
            raise AssembleInputError(f"Duplicate demographic result for track_id {track_id}")
        by_track[track_id] = result
    return by_track


def _validate_event(value: Any, index: int) -> Mapping[str, Any]:
    name = f"EventBatch.events[{index}]"
    event = _require_mapping(value, name)
    _require_fields(event, ("track_id", "timestamp", "event_type", "best_crop"), name)

    track_id = event["track_id"]
    if type(track_id) is not str or not track_id:
        raise AssembleInputError(f"{name}.track_id must be a non-empty string")
    timestamp = event["timestamp"]
    try:
        timestamp_is_finite = type(timestamp) in (int, float) and isfinite(float(timestamp))
    except OverflowError:
        timestamp_is_finite = False
    if not timestamp_is_finite:
        raise AssembleInputError(f"{name}.timestamp must be a finite number")
    event_type = event["event_type"]
    if type(event_type) is not int or event_type not in (0, 1):
        raise AssembleInputError(f"{name}.event_type must be exactly integer 0 or 1")

    best_crop_name = f"{name}.best_crop"
    best_crop = _require_mapping(event["best_crop"], best_crop_name)
    _require_fields(best_crop, ("frame_id", "bbox"), best_crop_name)
    return event


def _age_to_bucket(age: int) -> int:
    if age <= 4:
        return 0
    if age <= 13:
        return 1
    if age <= 25:
        return 2
    if age <= 45:
        return 3
    if age <= 65:
        return 4
    return 5


def _create_event_id(event: Mapping[str, Any], event_index: int) -> str:
    values = [
        event["track_id"],
        int(event["event_type"]),
        float(event["timestamp"]).hex(),
    ]
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"event-{event_index}-{digest}"


class Assemble:
    """Join Event and demographic batches into final output rows."""

    __slots__ = ()

    def __call__(
        self, event_batch: Any, demographics_batch: Any
    ) -> dict[str, list[dict[str, Any]]]:
        events = _event_values(event_batch)
        demographics_by_track = _index_demographics(demographics_batch)
        unused_track_ids = set(demographics_by_track)
        rows: list[dict[str, Any]] = []

        for event_index, value in enumerate(events):
            event = _validate_event(value, event_index)
            track_id = event["track_id"]
            demographic = demographics_by_track.get(track_id)
            if demographic is None:
                raise AssembleInputError(
                    f"EventBatch.events[{event_index}] track_id {track_id} "
                    "has no matching demographic result"
                )
            unused_track_ids.discard(track_id)
            rows.append(
                {
                    "event_id": _create_event_id(event, event_index),
                    "event": int(event["event_type"]),
                    "timestamp": float(event["timestamp"]),
                    "sex": int(demographic["sex"]),
                    "age_bucket": _age_to_bucket(demographic["age"]),
                }
            )

        if unused_track_ids:
            track_id = next(
                track_id for track_id in demographics_by_track if track_id in unused_track_ids
            )
            raise AssembleInputError(f"Unused demographic result for track_id {track_id}")
        return {"rows": rows}
