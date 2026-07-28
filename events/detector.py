"""Final TrackingState -> EventBatch reducer."""

from collections.abc import Mapping
from math import isfinite

from .geometry import GEOMETRY_EPSILON, Point, _segments_intersect, _side
from .models import EventBatch, EventRecord


def _require_mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_fields(value: Mapping, fields: tuple[str, ...], name: str) -> None:
    for field in fields:
        if field not in value:
            raise ValueError(f"Missing required {name} field: {field}")


def _finite_number(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _point_from_mapping(value, name: str) -> Point:
    point = _require_mapping(value, name)
    _require_fields(point, ("x", "y"), name)
    return (
        _finite_number(point["x"], f"{name}.x"),
        _finite_number(point["y"], f"{name}.y"),
    )


def _validate_bbox(value, name: str) -> dict[str, float]:
    bbox = _require_mapping(value, name)
    _require_fields(bbox, ("x1", "y1", "x2", "y2"), name)
    copied = {
        "x1": _finite_number(bbox["x1"], f"{name}.x1"),
        "y1": _finite_number(bbox["y1"], f"{name}.y1"),
        "x2": _finite_number(bbox["x2"], f"{name}.x2"),
        "y2": _finite_number(bbox["y2"], f"{name}.y2"),
    }
    if copied["x2"] < copied["x1"] or copied["y2"] < copied["y1"]:
        raise ValueError(f"{name} must have x2 >= x1 and y2 >= y1")
    return copied


def _copy_best_crop(value, name: str) -> dict:
    best_crop = _require_mapping(value, name)
    _require_fields(best_crop, ("frame_id", "bbox"), name)
    frame_id = best_crop["frame_id"]
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError(f"{name}.frame_id must be a non-empty string")
    return {
        "frame_id": frame_id,
        "bbox": _validate_bbox(best_crop["bbox"], f"{name}.bbox"),
    }


def _validate_track_point(value, name: str) -> tuple[float, Point]:
    point = _require_mapping(value, name)
    _require_fields(point, ("timestamp", "centre"), name)
    return (
        _finite_number(point["timestamp"], f"{name}.timestamp"),
        _point_from_mapping(point["centre"], f"{name}.centre"),
    )


def _validate_inputs(tracking_state, line_config) -> tuple[list[dict], Point, Point]:
    state = _require_mapping(tracking_state, "TrackingState")
    _require_fields(state, ("tracks",), "TrackingState")
    tracks = state["tracks"]
    if not isinstance(tracks, list):
        raise ValueError("TrackingState.tracks must be a list")

    for track_index, track in enumerate(tracks):
        track_name = f"TrackingState.tracks[{track_index}]"
        track_mapping = _require_mapping(track, track_name)
        _require_fields(track_mapping, ("track_id", "path", "best_crop"), track_name)
        track_id = track_mapping["track_id"]
        if not isinstance(track_id, str) or not track_id:
            raise ValueError(f"{track_name}.track_id must be a non-empty string")
        if not isinstance(track_mapping["path"], list):
            raise ValueError(f"{track_name}.path must be a list")
        previous_timestamp = None
        for point_index, path_point in enumerate(track_mapping["path"]):
            timestamp, _centre = _validate_track_point(
                path_point, f"{track_name}.path[{point_index}]"
            )
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(f"{track_name}.path timestamps must be monotonic")
            previous_timestamp = timestamp
        _copy_best_crop(track_mapping["best_crop"], f"{track_name}.best_crop")

    config = _require_mapping(line_config, "LineConfig")
    _require_fields(config, ("point_a", "point_b"), "LineConfig")
    point_a = _point_from_mapping(config["point_a"], "LineConfig.point_a")
    point_b = _point_from_mapping(config["point_b"], "LineConfig.point_b")
    if (point_a[0] - point_b[0]) ** 2 + (
        point_a[1] - point_b[1]
    ) ** 2 <= GEOMETRY_EPSILON**2:
        raise ValueError(
            "LineConfig point_a and point_b must define a non-zero segment"
        )
    return tracks, point_a, point_b


def _path_point(value) -> tuple[float, Point]:
    return (
        float(value["timestamp"]),
        (float(value["centre"]["x"]), float(value["centre"]["y"])),
    )


def _path_span_intersects(
    path: list, start_index: int, end_index: int, line_a: Point, line_b: Point
) -> bool:
    for index in range(start_index, end_index):
        _timestamp_a, point_a = _path_point(path[index])
        _timestamp_b, point_b = _path_point(path[index + 1])
        if _segments_intersect(point_a, point_b, line_a, line_b):
            return True
    return False


def _make_event(track: Mapping, timestamp: float, event_type: int) -> EventRecord:
    return {
        "track_id": track["track_id"],
        "timestamp": timestamp,
        "event_type": event_type,
        "best_crop": _copy_best_crop(track["best_crop"], "Track.best_crop"),
    }


def _events_for_track(
    track: Mapping, line_a: Point, line_b: Point
) -> list[EventRecord]:
    path = track["path"]
    events: list[EventRecord] = []
    last_side = None
    last_side_index = None

    for index, path_point in enumerate(path):
        timestamp, point = _path_point(path_point)
        current_side = _side(point, line_a, line_b)
        if current_side == 0:
            continue
        if last_side is None:
            last_side = current_side
            last_side_index = index
            continue
        if current_side == last_side:
            last_side_index = index
            continue
        if last_side_index is not None and _path_span_intersects(
            path, last_side_index, index, line_a, line_b
        ):
            events.append(
                _make_event(
                    track, timestamp, 1 if last_side == -1 and current_side == 1 else 0
                )
            )
        last_side = current_side
        last_side_index = index

    return events


def Event(tracking_state, line_config) -> EventBatch:
    """Return all finite-segment crossing events from the final TrackingState."""

    tracks, line_a, line_b = _validate_inputs(tracking_state, line_config)
    events: list[EventRecord] = []
    for track in tracks:
        for event in _events_for_track(track, line_a, line_b):
            events.append(event)
    return {"events": events}
