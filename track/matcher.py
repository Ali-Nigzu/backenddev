"""Historical-anchor matching for Track V2."""

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from track.config import TrackV2Config
from track.lifecycle import TrackStatus


@dataclass(frozen=True)
class Match:
    track_index: int
    observation_index: int
    distance: float


def numeric_track_id(track_id: str) -> int | None:
    if isinstance(track_id, str) and track_id.isdecimal():
        return int(track_id)
    return None


def track_sort_key(track) -> tuple:
    numeric = numeric_track_id(track["track_id"])
    return (0, numeric, str(track["track_id"])) if numeric is not None else (1, str(track["track_id"]))


def observation_sort_key(index_observation) -> tuple:
    index, observation = index_observation
    return (str(observation["detection_id"]), index)


def distance(a, b) -> float:
    ax = float(a["x"])
    ay = float(a["y"])
    bx = float(b["x"])
    by = float(b["y"])
    return math.hypot(ax - bx, ay - by)


def historical_anchor(path, config: TrackV2Config) -> tuple[dict, int]:
    if not path:
        raise ValueError("historical anchor requires a non-empty path")
    window = min(len(path), int(config.location_history_window_frames))
    points = path[-window:]
    return {
        "x": sum(float(point["center"]["x"]) for point in points) / window,
        "y": sum(float(point["center"]["y"]) for point in points) / window,
    }, window


def _candidate_sort_key(candidate: Match, tracks: Sequence[dict], statuses: Sequence[TrackStatus], config: TrackV2Config) -> tuple:
    track = tracks[candidate.track_index]
    status = statuses[candidate.track_index]
    track_id = str(track["track_id"])
    numeric = numeric_track_id(track_id)
    rounded_tie_bucket = math.floor(candidate.distance / float(config.anchor_tie_distance_px)) if config.anchor_tie_distance_px > 0 else candidate.distance
    older_key = (0, numeric, track_id) if numeric is not None else (1, track_id)
    return (
        rounded_tie_bucket,
        round(candidate.distance, 12),
        0 if status.active else 1,
        -len(track["path"]),
        older_key,
        candidate.observation_index,
    )


def _beats(current: Match | None, challenger: Match, tracks: Sequence[dict], statuses: Sequence[TrackStatus], config: TrackV2Config) -> bool:
    if current is None:
        return True
    distance_delta = challenger.distance - current.distance
    if abs(distance_delta) > float(config.anchor_tie_distance_px):
        return distance_delta < 0

    challenger_track = tracks[challenger.track_index]
    current_track = tracks[current.track_index]
    challenger_status = statuses[challenger.track_index]
    current_status = statuses[current.track_index]
    tie_key = (
        0 if challenger_status.active else 1,
        -len(challenger_track["path"]),
        track_sort_key(challenger_track),
    )
    current_key = (
        0 if current_status.active else 1,
        -len(current_track["path"]),
        track_sort_key(current_track),
    )
    return tie_key < current_key


def _best_for_observation(candidates: Iterable[Match], tracks: Sequence[dict], statuses: Sequence[TrackStatus], config: TrackV2Config) -> Match | None:
    best = None
    for candidate in candidates:
        if _beats(best, candidate, tracks, statuses, config):
            best = candidate
    return best


def match_tier(
    tracks: Sequence[dict],
    observations: Sequence[dict],
    statuses: Sequence[TrackStatus],
    track_indices: Sequence[int],
    observation_indices: Sequence[int],
    config: TrackV2Config,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return deterministic one-to-one matches and remaining observation indices."""

    candidates_by_observation: dict[int, list[Match]] = {index: [] for index in observation_indices}
    for track_index in track_indices:
        anchor, _points_used = historical_anchor(tracks[track_index]["path"], config)
        for observation_index in observation_indices:
            candidate_distance = distance(anchor, observations[observation_index]["center"])
            if candidate_distance <= float(config.max_anchor_distance_px):
                candidates_by_observation[observation_index].append(
                    Match(track_index, observation_index, candidate_distance)
                )

    used_tracks: set[int] = set()
    used_observations: set[int] = set()
    matches: list[Match] = []

    while True:
        proposals: list[Match] = []
        for observation_index in observation_indices:
            if observation_index in used_observations:
                continue
            available = [
                candidate
                for candidate in candidates_by_observation.get(observation_index, [])
                if candidate.track_index not in used_tracks
            ]
            best = _best_for_observation(available, tracks, statuses, config)
            if best is not None:
                proposals.append(best)

        if not proposals:
            break

        chosen = min(
            proposals,
            key=lambda candidate: _candidate_sort_key(candidate, tracks, statuses, config),
        )
        matches.append(chosen)
        used_tracks.add(chosen.track_index)
        used_observations.add(chosen.observation_index)

    remaining_observations = [
        observation_index for observation_index in observation_indices if observation_index not in used_observations
    ]
    return [(match.track_index, match.observation_index) for match in matches], remaining_observations
