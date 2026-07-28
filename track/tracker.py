"""Stateless deterministic Track reducer."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, MutableMapping

from track.config import _CONFIG, _TrackConfig
from track.matcher import _match_tier, _observation_sort_key, _track_sort_key

TrackingState = MutableMapping[str, list[dict[str, Any]]]
ObservationBatch = dict[str, Any]

_ACTIVE = "active"
_TENTATIVE = "tentative"
_INACTIVE = "inactive"

_REQUIRED_TRACKING_STATE_FIELDS = ("tracks",)
_REQUIRED_TRACK_FIELDS = ("track_id", "path", "best_crop", "best_crop_confidence")
_REQUIRED_BEST_CROP_FIELDS = ("frame_id", "bbox", "embedding")
_REQUIRED_POINT_FIELDS = ("timestamp", "center")
_REQUIRED_OBSERVATION_BATCH_FIELDS = ("frame_id", "timestamp", "observations")
_REQUIRED_OBSERVATION_FIELDS = ("detection_id", "bbox", "center", "embedding", "confidence")
_REQUIRED_BBOX_FIELDS = ("x1", "y1", "x2", "y2")
_REQUIRED_CENTER_FIELDS = ("x", "y")


@dataclass(frozen=True)
class _TrackStatus:
    state: str
    active: bool


def _require_mapping(value, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")


def _require_fields(value, fields, name: str) -> None:
    _require_mapping(value, name)
    for field in fields:
        if field not in value:
            raise ValueError(f"Missing required {name} field: {field}")


def _require_finite_number(value, name: str) -> float:
    if not isinstance(value, (float, int)) or not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_bbox(bbox, name: str) -> None:
    _require_fields(bbox, _REQUIRED_BBOX_FIELDS, name)
    x1 = _require_finite_number(bbox["x1"], f"{name}.x1")
    y1 = _require_finite_number(bbox["y1"], f"{name}.y1")
    x2 = _require_finite_number(bbox["x2"], f"{name}.x2")
    y2 = _require_finite_number(bbox["y2"], f"{name}.y2")
    if x1 > x2 or y1 > y2:
        raise ValueError(f"{name} must have x1 <= x2 and y1 <= y2")


def _validate_center(center, name: str) -> None:
    _require_fields(center, _REQUIRED_CENTER_FIELDS, name)
    _require_finite_number(center["x"], f"{name}.x")
    _require_finite_number(center["y"], f"{name}.y")


def _validate_tracking_state(state) -> None:
    _require_fields(state, _REQUIRED_TRACKING_STATE_FIELDS, "TrackingState")
    if not isinstance(state["tracks"], list):
        raise ValueError("TrackingState.tracks must be a list")

    seen_ids = set()
    for track_index, track in enumerate(state["tracks"]):
        name = f"TrackingState.tracks[{track_index}]"
        _require_fields(track, _REQUIRED_TRACK_FIELDS, name)
        track_id = track["track_id"]
        if not isinstance(track_id, str) or not track_id:
            raise ValueError(f"{name}.track_id must be a non-empty string")
        if track_id in seen_ids:
            raise ValueError(f"Duplicate track_id: {track_id}")
        seen_ids.add(track_id)

        if not isinstance(track["path"], list) or not track["path"]:
            raise ValueError(f"{name}.path must be a non-empty list")
        previous_timestamp = None
        for point_index, point in enumerate(track["path"]):
            point_name = f"{name}.path[{point_index}]"
            _require_fields(point, _REQUIRED_POINT_FIELDS, point_name)
            timestamp = _require_finite_number(point["timestamp"], f"{point_name}.timestamp")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(f"{name}.path timestamps must be monotonic")
            previous_timestamp = timestamp
            _validate_center(point["center"], f"{point_name}.center")

        _require_fields(track["best_crop"], _REQUIRED_BEST_CROP_FIELDS, f"{name}.best_crop")
        if not isinstance(track["best_crop"]["frame_id"], str):
            raise ValueError(f"{name}.best_crop.frame_id must be a string")
        _validate_bbox(track["best_crop"]["bbox"], f"{name}.best_crop.bbox")
        _require_finite_number(track["best_crop_confidence"], f"{name}.best_crop_confidence")


def _validate_observation_batch(batch) -> None:
    _require_fields(batch, _REQUIRED_OBSERVATION_BATCH_FIELDS, "ObservationBatch")
    if not isinstance(batch["frame_id"], str) or not batch["frame_id"]:
        raise ValueError("ObservationBatch.frame_id must be a non-empty string")
    _require_finite_number(batch["timestamp"], "ObservationBatch.timestamp")
    if not isinstance(batch["observations"], list):
        raise ValueError("ObservationBatch.observations must be a list")

    seen_ids = set()
    for observation_index, observation in enumerate(batch["observations"]):
        name = f"ObservationBatch.observations[{observation_index}]"
        _require_fields(observation, _REQUIRED_OBSERVATION_FIELDS, name)
        detection_id = observation["detection_id"]
        if not isinstance(detection_id, str) or not detection_id:
            raise ValueError(f"{name}.detection_id must be a non-empty string")
        if detection_id in seen_ids:
            raise ValueError(f"Duplicate detection_id: {detection_id}")
        seen_ids.add(detection_id)
        _validate_bbox(observation["bbox"], f"{name}.bbox")
        _validate_center(observation["center"], f"{name}.center")
        _require_finite_number(observation["confidence"], f"{name}.confidence")


def _classify_track(track: dict, current_frame_number: float, config: _TrackConfig) -> _TrackStatus:
    frame_delta = float(current_frame_number) - float(track["path"][-1]["timestamp"])
    if frame_delta < 0:
        raise ValueError("Track path contains a frame newer than the observation batch")

    if len(track["path"]) >= int(config.confirmation_hits):
        active = int(frame_delta) <= int(config.active_timeout_frames)
        return _TrackStatus(_ACTIVE if active else _INACTIVE, active)

    tentative = int(frame_delta) <= int(config.tentative_timeout_frames)
    return _TrackStatus(_TENTATIVE if tentative else _INACTIVE, False)


def _append_observation(track, observation, frame_id: str, frame_number: float) -> None:
    track["path"].append(
        {
            "timestamp": float(frame_number),
            "center": {
                "x": float(observation["center"]["x"]),
                "y": float(observation["center"]["y"]),
            },
        }
    )
    confidence = float(observation["confidence"])
    if confidence > float(track["best_crop_confidence"]):
        track["best_crop"] = {
            "frame_id": frame_id,
            "bbox": dict(observation["bbox"]),
            "embedding": observation["embedding"],
        }
        track["best_crop_confidence"] = confidence


def _create_track(observation, frame_id: str, frame_number: float, track_id: str) -> dict:
    return {
        "track_id": str(track_id),
        "path": [
            {
                "timestamp": float(frame_number),
                "center": {
                    "x": float(observation["center"]["x"]),
                    "y": float(observation["center"]["y"]),
                },
            }
        ],
        "best_crop": {
            "frame_id": frame_id,
            "bbox": dict(observation["bbox"]),
            "embedding": observation["embedding"],
        },
        "best_crop_confidence": float(observation["confidence"]),
    }


def _next_numeric_track_id(tracks) -> int:
    max_numeric_id = 0
    for track in tracks:
        track_id = str(track["track_id"])
        if track_id.isdecimal():
            max_numeric_id = max(max_numeric_id, int(track_id))
    return max_numeric_id + 1


def Track(tracking_state: TrackingState, observation_batch: ObservationBatch) -> TrackingState:
    """Update ``tracking_state`` in place from one ``ObservationBatch`` and return it."""

    _validate_tracking_state(tracking_state)
    _validate_observation_batch(observation_batch)

    frame_number = float(observation_batch["timestamp"])
    frame_id = observation_batch["frame_id"]

    tracking_state["tracks"].sort(key=_track_sort_key)
    ordered_observations = [
        observation
        for _index, observation in sorted(
            enumerate(observation_batch["observations"]), key=_observation_sort_key
        )
    ]

    statuses = [_classify_track(track, frame_number, _CONFIG) for track in tracking_state["tracks"]]
    active_track_indices = [index for index, status in enumerate(statuses) if status.state == _ACTIVE]
    tentative_track_indices = [index for index, status in enumerate(statuses) if status.state == _TENTATIVE]
    all_observation_indices = list(range(len(ordered_observations)))

    active_matches, remaining_observation_indices = _match_tier(
        tracking_state["tracks"],
        ordered_observations,
        statuses,
        active_track_indices,
        all_observation_indices,
        _CONFIG,
    )
    tentative_matches, remaining_observation_indices = _match_tier(
        tracking_state["tracks"],
        ordered_observations,
        statuses,
        tentative_track_indices,
        remaining_observation_indices,
        _CONFIG,
    )

    for state_track_index, observation_index in sorted(active_matches + tentative_matches):
        _append_observation(
            tracking_state["tracks"][state_track_index],
            ordered_observations[observation_index],
            frame_id,
            frame_number,
        )

    next_id = _next_numeric_track_id(tracking_state["tracks"])
    for observation_index in sorted(remaining_observation_indices):
        tracking_state["tracks"].append(
            _create_track(ordered_observations[observation_index], frame_id, frame_number, str(next_id))
        )
        next_id += 1

    tracking_state["tracks"].sort(key=_track_sort_key)
    return tracking_state
