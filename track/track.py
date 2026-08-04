"""Deterministic historical-anchor tracking stage."""

_LOCATION_HISTORY_WINDOW_FRAMES = 7
_ANCHOR_WEIGHT_EXPONENT = 1.0
_MAX_ANCHOR_DISTANCE_PX = 100.0
_ANCHOR_TIE_DISTANCE_PX = 20.0
_CONFIRMATION_HITS = 3
_ACTIVE_TIMEOUT_FRAMES = 30
_TENTATIVE_TIMEOUT_FRAMES = 15

import math
from typing import Iterable, Sequence


def _track_sort_key(track) -> tuple:
    track_id = str(track["track_id"])
    return (0, int(track_id), track_id) if track_id.isdecimal() else (1, track_id)


def _historical_anchor(path) -> tuple[float, float]:
    window = min(len(path), _LOCATION_HISTORY_WINDOW_FRAMES)
    points = path[-window:]
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for index, point in enumerate(points, start=1):
        weight = float(index) ** _ANCHOR_WEIGHT_EXPONENT
        total_weight += weight
        weighted_x += point["centre"]["x"] * weight
        weighted_y += point["centre"]["y"] * weight
    return weighted_x / total_weight, weighted_y / total_weight


def _beats(current: tuple[int, int, float] | None, challenger: tuple[int, int, float], tie_keys: Sequence[tuple]) -> bool:
    if current is None:
        return True
    distance_delta = challenger[2] - current[2]
    if abs(distance_delta) > _ANCHOR_TIE_DISTANCE_PX:
        return distance_delta < 0
    return (tie_keys[challenger[0]], challenger[1], challenger[0]) < (tie_keys[current[0]], current[1], current[0])


def _best_for_detection(candidates: Iterable[tuple[int, int, float]], tie_keys: Sequence[tuple]) -> tuple[int, int, float] | None:
    best = None
    for candidate in candidates:
        if _beats(best, candidate, tie_keys):
            best = candidate
    return best


def _match_tier(
    tracks: Sequence[dict],
    detections: Sequence[dict],
    statuses: Sequence[str],
    track_indices: Sequence[int],
    detection_indices: Sequence[int],
) -> tuple[list[tuple[int, int]], list[int]]:
    candidates_by_detection: dict[int, list[tuple[int, int, float]]] = {index: [] for index in detection_indices}
    for track_index in track_indices:
        anchor_x, anchor_y = _historical_anchor(tracks[track_index]["path"])
        for detection_index in detection_indices:
            detection_centre = detections[detection_index]["centre"]
            candidate_distance = math.hypot(
                anchor_x - detection_centre["x"],
                anchor_y - detection_centre["y"],
            )
            if candidate_distance <= _MAX_ANCHOR_DISTANCE_PX:
                candidates_by_detection[detection_index].append((track_index, detection_index, candidate_distance))

    tie_keys = [
        (0 if status == "active" else 1, -len(track["path"]), _track_sort_key(track))
        for track, status in zip(tracks, statuses)
    ]
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    while True:
        proposals: list[tuple[int, int, float]] = []
        for detection_index in detection_indices:
            if detection_index in used_detections:
                continue
            best = _best_for_detection(
                (
                    candidate
                    for candidate in candidates_by_detection.get(detection_index, ())
                    if candidate[0] not in used_tracks
                ),
                tie_keys,
            )
            if best is not None:
                proposals.append(best)

        if not proposals:
            break

        chosen = _best_for_detection(proposals, tie_keys)
        matches.append(chosen)
        used_tracks.add(chosen[0])
        used_detections.add(chosen[1])

    return (
        [(track_index, detection_index) for track_index, detection_index, _distance in matches],
        [index for index in detection_indices if index not in used_detections],
    )

def _classify_track(track: dict, current_frame_number: float) -> str:
    frame_delta = current_frame_number - track["path"][-1]["timestamp"]
    if len(track["path"]) >= _CONFIRMATION_HITS:
        return "active" if int(frame_delta) <= _ACTIVE_TIMEOUT_FRAMES else "inactive"

    return "tentative" if int(frame_delta) <= _TENTATIVE_TIMEOUT_FRAMES else "inactive"


def _append_detection(track, detection, frame_id: str, frame_number: float) -> None:
    track["path"].append({"timestamp": frame_number, "centre": dict(detection["centre"])})
    confidence = detection["confidence"]
    if confidence > track["best_crop_confidence"]:
        track["best_crop"] = {"frame_id": frame_id, "bbox": dict(detection["bbox"])}
        track["best_crop_confidence"] = confidence


def _create_track(detection, frame_id: str, frame_number: float, track_id: str) -> dict:
    return {
        "track_id": track_id,
        "path": [{"timestamp": frame_number, "centre": dict(detection["centre"])}],
        "best_crop": {"frame_id": frame_id, "bbox": dict(detection["bbox"])},
        "best_crop_confidence": detection["confidence"],
    }


def _next_numeric_track_id(tracks) -> int:
    max_numeric_id = 0
    for track in tracks:
        track_id = str(track["track_id"])
        if track_id.isdecimal():
            max_numeric_id = max(max_numeric_id, int(track_id))
    return max_numeric_id + 1


def _process_frame(tracking_state, frame_detections, next_id: int):
    frame_number = frame_detections["timestamp"]
    frame_id = frame_detections["frame_id"]

    ordered_detections = [
        detection
        for _index, detection in sorted(
            enumerate(frame_detections["detections"]),
            key=lambda item: (str(item[1]["detection_id"]), item[0]),
        )
    ]

    statuses = []
    active_track_indices = []
    tentative_track_indices = []
    for index, track in enumerate(tracking_state["tracks"]):
        status = _classify_track(track, frame_number)
        statuses.append(status)
        if status == "active":
            active_track_indices.append(index)
        elif status == "tentative":
            tentative_track_indices.append(index)
    all_detection_indices = range(len(ordered_detections))

    active_matches, remaining_detection_indices = _match_tier(
        tracking_state["tracks"], ordered_detections, statuses, active_track_indices, all_detection_indices
    )
    tentative_matches, remaining_detection_indices = _match_tier(
        tracking_state["tracks"], ordered_detections, statuses, tentative_track_indices, remaining_detection_indices
    )

    for state_track_index, detection_index in sorted(active_matches + tentative_matches):
        _append_detection(tracking_state["tracks"][state_track_index], ordered_detections[detection_index], frame_id, frame_number)

    for detection_index in sorted(remaining_detection_indices):
        tracking_state["tracks"].append(_create_track(ordered_detections[detection_index], frame_id, frame_number, str(next_id)))
        next_id += 1

    tracking_state["tracks"].sort(key=_track_sort_key)
    return tracking_state, next_id


class Track:
    __slots__ = ()

    def __call__(self, tracking_state, detection_batch):
        tracking_state["tracks"].sort(key=_track_sort_key)
        next_id = _next_numeric_track_id(tracking_state["tracks"])
        for frame_detections in detection_batch["detections"]:
            tracking_state, next_id = _process_frame(
                tracking_state, frame_detections, next_id
            )
        return tracking_state
