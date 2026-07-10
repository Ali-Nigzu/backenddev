"""Deterministic Track V2 matching."""

import math
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from track.config import TrackV2Config
from track.models import vector_values
from track.motion import motion_gate


@dataclass(frozen=True)
class CandidateMatch:
    track_index: int
    observation_index: int
    track_id: str
    detection_id: str
    motion_distance: float
    normalized_motion: float
    appearance_similarity: float
    cost: float


def numeric_track_id(track_id: str) -> int | None:
    if isinstance(track_id, str) and track_id.isdecimal():
        return int(track_id)
    return None


def track_sort_key(track) -> tuple:
    numeric = numeric_track_id(track["track_id"])
    return (0, numeric, track["track_id"]) if numeric is not None else (1, track["track_id"])


def observation_sort_key(index_observation) -> tuple:
    index, observation = index_observation
    return (str(observation["detection_id"]), index)


def embedding_similarity(a, b) -> float:
    av = [float(v) for v in vector_values(a)]
    bv = [float(v) for v in vector_values(b)]
    if not av or not bv or len(av) != len(bv):
        return 0.0

    dot = sum(x * y for x, y in zip(av, bv))
    norm_a = math.sqrt(sum(x * x for x in av))
    norm_b = math.sqrt(sum(y * y for y in bv))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def build_candidates(
    ordered_tracks: Sequence[dict],
    ordered_observations: Sequence[dict],
    timestamp: float,
    config: TrackV2Config,
) -> List[CandidateMatch]:
    candidates: List[CandidateMatch] = []
    total_weight = config.motion_weight + config.appearance_weight
    motion_weight = config.motion_weight / total_weight if total_weight > 0 else 1.0
    appearance_weight = config.appearance_weight / total_weight if total_weight > 0 else 0.0

    for track_index, track in enumerate(ordered_tracks):
        for observation_index, observation in enumerate(ordered_observations):
            allowed, motion_distance, normalized_motion, _dt = motion_gate(
                track, observation, timestamp, config
            )
            if not allowed:
                continue

            similarity = embedding_similarity(
                track["best_crop"].get("embedding"), observation.get("embedding")
            )
            if similarity < config.min_appearance_similarity:
                continue

            cost = motion_weight * normalized_motion + appearance_weight * (1.0 - similarity)
            if cost > config.max_combined_cost:
                continue

            candidates.append(
                CandidateMatch(
                    track_index=track_index,
                    observation_index=observation_index,
                    track_id=str(track["track_id"]),
                    detection_id=str(observation["detection_id"]),
                    motion_distance=motion_distance,
                    normalized_motion=normalized_motion,
                    appearance_similarity=similarity,
                    cost=cost,
                )
            )
    return candidates


def candidate_sort_key(candidate: CandidateMatch) -> tuple:
    numeric = numeric_track_id(candidate.track_id)
    track_key = (0, numeric, candidate.track_id) if numeric is not None else (1, candidate.track_id)
    return (
        round(candidate.cost, 12),
        round(candidate.normalized_motion, 12),
        round(-candidate.appearance_similarity, 12),
        track_key,
        candidate.detection_id,
        candidate.observation_index,
    )


def assign_matches(
    ordered_tracks: Sequence[dict],
    ordered_observations: Sequence[dict],
    timestamp: float,
    config: TrackV2Config,
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    candidates = sorted(
        build_candidates(ordered_tracks, ordered_observations, timestamp, config),
        key=candidate_sort_key,
    )
    used_tracks: Set[int] = set()
    used_observations: Set[int] = set()
    matches: List[Tuple[int, int]] = []

    for candidate in candidates:
        if candidate.track_index in used_tracks or candidate.observation_index in used_observations:
            continue
        used_tracks.add(candidate.track_index)
        used_observations.add(candidate.observation_index)
        matches.append((candidate.track_index, candidate.observation_index))

    unmatched_tracks = set(range(len(ordered_tracks))) - used_tracks
    unmatched_observations = set(range(len(ordered_observations))) - used_observations
    return matches, unmatched_tracks, unmatched_observations
