"""Stateless deterministic Track V2 reducer."""

from math import isfinite

from track.config import TrackV2Config
from track.lifecycle import append_observation, create_track, is_expired
from track.matching import assign_matches, observation_sort_key, track_sort_key
from track.models import (
    REQUIRED_BBOX_FIELDS,
    REQUIRED_BEST_CROP_FIELDS,
    REQUIRED_CENTER_FIELDS,
    REQUIRED_OBSERVATION_BATCH_FIELDS,
    REQUIRED_OBSERVATION_FIELDS,
    REQUIRED_POINT_FIELDS,
    REQUIRED_TRACK_FIELDS,
    REQUIRED_TRACKING_STATE_FIELDS,
)


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
    _require_fields(bbox, REQUIRED_BBOX_FIELDS, name)
    x1 = _require_finite_number(bbox["x1"], f"{name}.x1")
    y1 = _require_finite_number(bbox["y1"], f"{name}.y1")
    x2 = _require_finite_number(bbox["x2"], f"{name}.x2")
    y2 = _require_finite_number(bbox["y2"], f"{name}.y2")
    if x1 > x2 or y1 > y2:
        raise ValueError(f"{name} must have x1 <= x2 and y1 <= y2")


def _validate_center(center, name: str) -> None:
    _require_fields(center, REQUIRED_CENTER_FIELDS, name)
    _require_finite_number(center["x"], f"{name}.x")
    _require_finite_number(center["y"], f"{name}.y")


def _validate_tracking_state(state) -> None:
    _require_fields(state, REQUIRED_TRACKING_STATE_FIELDS, "TrackingState")
    if not isinstance(state["tracks"], list):
        raise ValueError("TrackingState.tracks must be a list")

    seen_ids = set()
    for track_index, track in enumerate(state["tracks"]):
        name = f"TrackingState.tracks[{track_index}]"
        _require_fields(track, REQUIRED_TRACK_FIELDS, name)
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
            _require_fields(point, REQUIRED_POINT_FIELDS, point_name)
            timestamp = _require_finite_number(point["timestamp"], f"{point_name}.timestamp")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(f"{name}.path timestamps must be monotonic")
            previous_timestamp = timestamp
            _validate_center(point["center"], f"{point_name}.center")

        _require_fields(track["best_crop"], REQUIRED_BEST_CROP_FIELDS, f"{name}.best_crop")
        if not isinstance(track["best_crop"]["frame_id"], str):
            raise ValueError(f"{name}.best_crop.frame_id must be a string")
        _validate_bbox(track["best_crop"]["bbox"], f"{name}.best_crop.bbox")
        _require_finite_number(track["best_crop_confidence"], f"{name}.best_crop_confidence")


def _validate_observation_batch(batch) -> None:
    _require_fields(batch, REQUIRED_OBSERVATION_BATCH_FIELDS, "ObservationBatch")
    if not isinstance(batch["frame_id"], str) or not batch["frame_id"]:
        raise ValueError("ObservationBatch.frame_id must be a non-empty string")
    _require_finite_number(batch["timestamp"], "ObservationBatch.timestamp")
    if not isinstance(batch["observations"], list):
        raise ValueError("ObservationBatch.observations must be a list")

    seen_ids = set()
    for observation_index, observation in enumerate(batch["observations"]):
        name = f"ObservationBatch.observations[{observation_index}]"
        _require_fields(observation, REQUIRED_OBSERVATION_FIELDS, name)
        detection_id = observation["detection_id"]
        if not isinstance(detection_id, str) or not detection_id:
            raise ValueError(f"{name}.detection_id must be a non-empty string")
        if detection_id in seen_ids:
            raise ValueError(f"Duplicate detection_id: {detection_id}")
        seen_ids.add(detection_id)
        _validate_bbox(observation["bbox"], f"{name}.bbox")
        _validate_center(observation["center"], f"{name}.center")
        _require_finite_number(observation["confidence"], f"{name}.confidence")


def _next_numeric_track_id(tracks) -> int:
    numeric_ids = [int(track["track_id"]) for track in tracks if str(track["track_id"]).isdecimal()]
    return max(numeric_ids, default=0) + 1


def _reorder_state_tracks(state) -> None:
    state["tracks"].sort(key=track_sort_key)


def Track(tracking_state, observation_batch, config: TrackV2Config | None = None):
    """Update ``tracking_state`` in place from one ``ObservationBatch`` and return it."""

    config = config or TrackV2Config()
    _validate_tracking_state(tracking_state)
    _validate_observation_batch(observation_batch)

    timestamp = float(observation_batch["timestamp"])
    frame_id = observation_batch["frame_id"]

    _reorder_state_tracks(tracking_state)
    ordered_observations = [
        observation
        for _index, observation in sorted(
            enumerate(observation_batch["observations"]), key=observation_sort_key
        )
    ]

    next_id = _next_numeric_track_id(tracking_state["tracks"])

    matches, unmatched_track_indices, unmatched_observation_indices = assign_matches(
        tracking_state["tracks"], ordered_observations, timestamp, config
    )

    for track_index, observation_index in sorted(matches):
        append_observation(
            tracking_state["tracks"][track_index],
            ordered_observations[observation_index],
            frame_id,
            timestamp,
            config,
        )

    retained_tracks = []
    matched_track_indices = {track_index for track_index, _observation_index in matches}
    for track_index, track in enumerate(tracking_state["tracks"]):
        if track_index in matched_track_indices or not is_expired(track, timestamp, config):
            retained_tracks.append(track)
    tracking_state["tracks"] = retained_tracks

    for observation_index in sorted(unmatched_observation_indices):
        observation = ordered_observations[observation_index]
        tracking_state["tracks"].append(create_track(observation, frame_id, timestamp, str(next_id)))
        next_id += 1

    _reorder_state_tracks(tracking_state)
    return tracking_state


class TrackV2:
    """Callable wrapper that stores configuration only, never analytics state."""

    __slots__ = ("config",)

    def __init__(self, config: TrackV2Config | None = None):
        self.config = config or TrackV2Config()

    def __call__(self, tracking_state, observation_batch):
        return Track(tracking_state, observation_batch, self.config)
