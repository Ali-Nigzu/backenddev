"""Stateless deterministic Track reducer."""

from collections.abc import Mapping
from math import isfinite
from track.config import _ACTIVE_TIMEOUT_FRAMES, _CONFIRMATION_HITS, _TENTATIVE_TIMEOUT_FRAMES
from track.matcher import _match_tier, _track_sort_key


def _require_fields(value, fields, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    for field in fields:
        if field not in value:
            raise ValueError(f"Missing required {name} field: {field}")


def _require_finite_number(value, name: str) -> float:
    if not isinstance(value, (float, int)) or not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_bbox(bbox, name: str) -> None:
    _require_fields(bbox, ("x1", "y1", "x2", "y2"), name)
    x1 = _require_finite_number(bbox["x1"], f"{name}.x1")
    y1 = _require_finite_number(bbox["y1"], f"{name}.y1")
    x2 = _require_finite_number(bbox["x2"], f"{name}.x2")
    y2 = _require_finite_number(bbox["y2"], f"{name}.y2")
    if x1 > x2 or y1 > y2:
        raise ValueError(f"{name} must have x1 <= x2 and y1 <= y2")


def _validate_centre(centre, name: str) -> None:
    _require_fields(centre, ("x", "y"), name)
    _require_finite_number(centre["x"], f"{name}.x")
    _require_finite_number(centre["y"], f"{name}.y")


def _validate_tracking_state(state) -> None:
    _require_fields(state, ("tracks",), "TrackingState")
    if not isinstance(state["tracks"], list):
        raise ValueError("TrackingState.tracks must be a list")

    seen_ids = set()
    for track_index, track in enumerate(state["tracks"]):
        name = f"TrackingState.tracks[{track_index}]"
        _require_fields(track, ("track_id", "path", "best_crop", "best_crop_confidence"), name)
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
            _require_fields(point, ("timestamp", "centre"), point_name)
            timestamp = _require_finite_number(point["timestamp"], f"{point_name}.timestamp")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(f"{name}.path timestamps must be monotonic")
            previous_timestamp = timestamp
            _validate_centre(point["centre"], f"{point_name}.centre")

        _require_fields(track["best_crop"], ("frame_id", "bbox"), f"{name}.best_crop")
        if not isinstance(track["best_crop"]["frame_id"], str):
            raise ValueError(f"{name}.best_crop.frame_id must be a string")
        _validate_bbox(track["best_crop"]["bbox"], f"{name}.best_crop.bbox")
        _require_finite_number(track["best_crop_confidence"], f"{name}.best_crop_confidence")


def _validate_frame_detections(batch, name: str = "Detections") -> None:
    _require_fields(batch, ("frame_id", "timestamp", "detections"), name)
    if not isinstance(batch["frame_id"], str) or not batch["frame_id"]:
        raise ValueError(f"{name}.frame_id must be a non-empty string")
    _require_finite_number(batch["timestamp"], f"{name}.timestamp")
    if not isinstance(batch["detections"], list):
        raise ValueError(f"{name}.detections must be a list")

    seen_ids = set()
    for detection_index, detection in enumerate(batch["detections"]):
        detection_name = f"{name}.detections[{detection_index}]"
        _require_fields(detection, ("detection_id", "bbox", "centre", "confidence"), detection_name)
        detection_id = detection["detection_id"]
        if not isinstance(detection_id, str) or not detection_id:
            raise ValueError(f"{detection_name}.detection_id must be a non-empty string")
        if detection_id in seen_ids:
            raise ValueError(f"Duplicate detection_id: {detection_id}")
        seen_ids.add(detection_id)
        _validate_bbox(detection["bbox"], f"{detection_name}.bbox")
        _validate_centre(detection["centre"], f"{detection_name}.centre")
        _require_finite_number(detection["confidence"], f"{detection_name}.confidence")


def _classify_track(track: dict, current_frame_number: float) -> str:
    frame_delta = float(current_frame_number) - float(track["path"][-1]["timestamp"])
    if frame_delta < 0:
        raise ValueError("Track path contains a frame newer than the Detections object")

    if len(track["path"]) >= _CONFIRMATION_HITS:
        return "active" if int(frame_delta) <= _ACTIVE_TIMEOUT_FRAMES else "inactive"

    return "tentative" if int(frame_delta) <= _TENTATIVE_TIMEOUT_FRAMES else "inactive"


def _append_detection(track, detection, frame_id: str, frame_number: float) -> None:
    track["path"].append({"timestamp": float(frame_number), "centre": dict(detection["centre"])})
    confidence = float(detection["confidence"])
    if confidence > float(track["best_crop_confidence"]):
        track["best_crop"] = {"frame_id": frame_id, "bbox": dict(detection["bbox"])}
        track["best_crop_confidence"] = confidence


def _create_track(detection, frame_id: str, frame_number: float, track_id: str) -> dict:
    return {
        "track_id": str(track_id),
        "path": [{"timestamp": float(frame_number), "centre": dict(detection["centre"])}],
        "best_crop": {"frame_id": frame_id, "bbox": dict(detection["bbox"])},
        "best_crop_confidence": float(detection["confidence"]),
    }


def _next_numeric_track_id(tracks) -> int:
    max_numeric_id = 0
    for track in tracks:
        track_id = str(track["track_id"])
        if track_id.isdecimal():
            max_numeric_id = max(max_numeric_id, int(track_id))
    return max_numeric_id + 1


def _validate_detection_batch(detection_batch) -> list:
    _require_fields(detection_batch, ("detections",), "DetectionBatch")
    detections = detection_batch["detections"]
    if not isinstance(detections, list):
        raise ValueError("DetectionBatch.detections must be a list")
    return detections


def _process_frame(tracking_state, frame_detections):
    frame_number = float(frame_detections["timestamp"])
    frame_id = frame_detections["frame_id"]

    tracking_state["tracks"].sort(key=_track_sort_key)
    ordered_detections = [
        detection
        for _index, detection in sorted(
            enumerate(frame_detections["detections"]),
            key=lambda item: (str(item[1]["detection_id"]), item[0]),
        )
    ]

    statuses = [_classify_track(track, frame_number) for track in tracking_state["tracks"]]
    active_track_indices = [index for index, status in enumerate(statuses) if status == "active"]
    tentative_track_indices = [index for index, status in enumerate(statuses) if status == "tentative"]
    all_detection_indices = list(range(len(ordered_detections)))

    active_matches, remaining_detection_indices = _match_tier(
        tracking_state["tracks"], ordered_detections, statuses, active_track_indices, all_detection_indices
    )
    tentative_matches, remaining_detection_indices = _match_tier(
        tracking_state["tracks"], ordered_detections, statuses, tentative_track_indices, remaining_detection_indices
    )

    for state_track_index, detection_index in sorted(active_matches + tentative_matches):
        _append_detection(tracking_state["tracks"][state_track_index], ordered_detections[detection_index], frame_id, frame_number)

    next_id = _next_numeric_track_id(tracking_state["tracks"])
    for detection_index in sorted(remaining_detection_indices):
        tracking_state["tracks"].append(_create_track(ordered_detections[detection_index], frame_id, frame_number, str(next_id)))
        next_id += 1

    tracking_state["tracks"].sort(key=_track_sort_key)
    return tracking_state


def Track(tracking_state, detection_batch):
    """Update ``tracking_state`` in place from one complete ``DetectionBatch`` and return it."""

    _validate_tracking_state(tracking_state)
    frame_detections_batch = _validate_detection_batch(detection_batch)
    for index, frame_detections in enumerate(frame_detections_batch):
        _validate_frame_detections(frame_detections, f"DetectionBatch.detections[{index}]")
        tracking_state = _process_frame(tracking_state, frame_detections)
    return tracking_state
