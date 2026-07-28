"""Historical-anchor matching for Track."""

import math
from typing import Iterable, Sequence

from track.config import (
    _ANCHOR_TIE_DISTANCE_PX,
    _ANCHOR_WEIGHT_EXPONENT,
    _LOCATION_HISTORY_WINDOW_FRAMES,
    _MAX_ANCHOR_DISTANCE_PX,
)


def _track_sort_key(track) -> tuple:
    track_id = str(track["track_id"])
    return (0, int(track_id), track_id) if track_id.isdecimal() else (1, track_id)


def _historical_anchor(path) -> dict:
    window = min(len(path), _LOCATION_HISTORY_WINDOW_FRAMES)
    points = path[-window:]
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for index, point in enumerate(points, start=1):
        weight = float(index) ** _ANCHOR_WEIGHT_EXPONENT
        total_weight += weight
        weighted_x += float(point["centre"]["x"]) * weight
        weighted_y += float(point["centre"]["y"]) * weight
    return {"x": weighted_x / total_weight, "y": weighted_y / total_weight}


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
        anchor = _historical_anchor(tracks[track_index]["path"])
        for detection_index in detection_indices:
            detection_centre = detections[detection_index]["centre"]
            candidate_distance = math.hypot(
                float(anchor["x"]) - float(detection_centre["x"]),
                float(anchor["y"]) - float(detection_centre["y"]),
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
