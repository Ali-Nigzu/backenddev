"""Historical-anchor matching for Track."""

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from track.config import _TrackConfig


@dataclass(frozen=True)
class _Match:
    track_index: int
    observation_index: int
    distance: float


def _track_sort_key(track) -> tuple:
    track_id = str(track["track_id"])
    return (0, int(track_id), track_id) if track_id.isdecimal() else (1, track_id)


def _observation_sort_key(index_observation) -> tuple:
    index, observation = index_observation
    return (str(observation["detection_id"]), index)


def _distance(a, b) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _historical_anchor(path, config: _TrackConfig) -> dict:
    window = min(len(path), int(config.location_history_window_frames))
    points = path[-window:]
    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for index, point in enumerate(points, start=1):
        weight = float(index) ** float(config.anchor_weight_exponent)
        total_weight += weight
        weighted_x += float(point["center"]["x"]) * weight
        weighted_y += float(point["center"]["y"]) * weight
    return {"x": weighted_x / total_weight, "y": weighted_y / total_weight}


def _continuity_key(track: dict, status) -> tuple:
    return (0 if status.active else 1, -len(track["path"]), _track_sort_key(track))


def _candidate_tie_key(candidate: _Match, tie_keys: Sequence[tuple]) -> tuple:
    return (tie_keys[candidate.track_index], candidate.observation_index, candidate.track_index)


def _beats(current: _Match | None, challenger: _Match, tie_keys: Sequence[tuple], config: _TrackConfig) -> bool:
    if current is None:
        return True
    distance_delta = challenger.distance - current.distance
    if abs(distance_delta) > float(config.anchor_tie_distance_px):
        return distance_delta < 0
    return _candidate_tie_key(challenger, tie_keys) < _candidate_tie_key(current, tie_keys)


def _best_for_observation(candidates: Iterable[_Match], tie_keys: Sequence[tuple], config: _TrackConfig) -> _Match | None:
    best = None
    for candidate in candidates:
        if _beats(best, candidate, tie_keys, config):
            best = candidate
    return best


def _match_tier(
    tracks: Sequence[dict],
    observations: Sequence[dict],
    statuses: Sequence,
    track_indices: Sequence[int],
    observation_indices: Sequence[int],
    config: _TrackConfig,
) -> tuple[list[tuple[int, int]], list[int]]:
    candidates_by_observation: dict[int, list[_Match]] = {index: [] for index in observation_indices}
    for track_index in track_indices:
        anchor = _historical_anchor(tracks[track_index]["path"], config)
        for observation_index in observation_indices:
            candidate_distance = _distance(anchor, observations[observation_index]["center"])
            if candidate_distance <= float(config.max_anchor_distance_px):
                candidates_by_observation[observation_index].append(
                    _Match(track_index, observation_index, candidate_distance)
                )

    tie_keys = [_continuity_key(track, status) for track, status in zip(tracks, statuses)]
    used_tracks: set[int] = set()
    used_observations: set[int] = set()
    matches: list[_Match] = []

    while True:
        proposals: list[_Match] = []
        for observation_index in observation_indices:
            if observation_index in used_observations:
                continue
            best = _best_for_observation(
                (
                    candidate
                    for candidate in candidates_by_observation.get(observation_index, ())
                    if candidate.track_index not in used_tracks
                ),
                tie_keys,
                config,
            )
            if best is not None:
                proposals.append(best)

        if not proposals:
            break

        chosen = _best_for_observation(proposals, tie_keys, config)
        matches.append(chosen)
        used_tracks.add(chosen.track_index)
        used_observations.add(chosen.observation_index)

    return (
        [(match.track_index, match.observation_index) for match in matches],
        [index for index in observation_indices if index not in used_observations],
    )
