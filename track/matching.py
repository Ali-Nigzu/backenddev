"""Deterministic Track V2 matching."""

import math
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from track.config import TrackV2Config
from track.models import vector_values
from track.motion import distance, motion_gate


@dataclass(frozen=True)
class CandidateMatch:
    track_index: int
    observation_index: int
    track_id: str
    detection_id: str
    motion_distance: float
    normalized_motion: float
    physically_plausible: bool
    appearance_similarity: float | None
    appearance_tiebreak_cost: float


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


def _embedding_similarity_values(a, b) -> float:
    av = [float(v) for v in a]
    bv = [float(v) for v in b]
    if not av or not bv or len(av) != len(bv):
        return 0.0

    dot = sum(x * y for x, y in zip(av, bv))
    norm_a = math.sqrt(sum(x * x for x in av))
    norm_b = math.sqrt(sum(y * y for y in bv))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def embedding_similarity(a, b) -> float:
    return _embedding_similarity_values(vector_values(a), vector_values(b))


def appearance_evidence(a, b) -> float | None:
    av = list(vector_values(a))
    bv = list(vector_values(b))
    if not av or not bv or len(av) != len(bv):
        return None
    return _embedding_similarity_values(av, bv)


def build_candidates(
    ordered_tracks: Sequence[dict],
    ordered_observations: Sequence[dict],
    timestamp: float,
    config: TrackV2Config,
) -> List[CandidateMatch]:
    candidates: List[CandidateMatch] = []

    for track_index, track in enumerate(ordered_tracks):
        for observation_index, observation in enumerate(ordered_observations):
            allowed, motion_distance, normalized_motion, _dt = motion_gate(
                track, observation, timestamp, config
            )
            if not allowed:
                motion_distance = distance(track["path"][-1]["center"], observation["center"])
                base_distance = max(config.base_motion_gate_px, config.epsilon)
                normalized_motion = motion_distance / base_distance

            similarity = appearance_evidence(
                track["best_crop"].get("embedding"), observation.get("embedding")
            )

            appearance_tiebreak_cost = 1.0 - similarity if similarity is not None else 1.0

            candidates.append(
                CandidateMatch(
                    track_index=track_index,
                    observation_index=observation_index,
                    track_id=str(track["track_id"]),
                    detection_id=str(observation["detection_id"]),
                    motion_distance=motion_distance,
                    normalized_motion=normalized_motion,
                    physically_plausible=allowed,
                    appearance_similarity=similarity,
                    appearance_tiebreak_cost=appearance_tiebreak_cost,
                )
            )
    return candidates


def candidate_sort_key(candidate: CandidateMatch) -> tuple:
    numeric = numeric_track_id(candidate.track_id)
    track_key = (0, numeric, candidate.track_id) if numeric is not None else (1, candidate.track_id)
    return (
        not candidate.physically_plausible,
        round(candidate.normalized_motion, 12),
        round(candidate.motion_distance, 12),
        round(candidate.appearance_tiebreak_cost, 12),
        track_key,
        candidate.detection_id,
        candidate.observation_index,
    )


def _cached_candidate_sort_key(candidate: CandidateMatch, cache: dict[CandidateMatch, tuple]) -> tuple:
    key = cache.get(candidate)
    if key is None:
        key = candidate_sort_key(candidate)
        cache[candidate] = key
    return key


def _better_assignment(
    candidate_matches: List[CandidateMatch],
    best_matches: List[CandidateMatch] | None,
    sort_key_cache: dict[CandidateMatch, tuple] | None = None,
) -> bool:
    if best_matches is None:
        return True

    if sort_key_cache is None:
        sort_key_cache = {}

    def score(matches: List[CandidateMatch]) -> tuple:
        ordered = sorted(
            matches,
            key=lambda match: _cached_candidate_sort_key(match, sort_key_cache),
        )
        return (
            -len(ordered),
            sum(0 if match.physically_plausible else 1 for match in ordered),
            round(sum(match.normalized_motion for match in ordered), 12),
            round(sum(match.motion_distance for match in ordered), 12),
            round(sum(match.appearance_tiebreak_cost for match in ordered), 12),
            tuple(_cached_candidate_sort_key(match, sort_key_cache) for match in ordered),
        )

    return score(candidate_matches) < score(best_matches)


def _maximum_continuity_assignment(candidates: List[CandidateMatch]) -> List[CandidateMatch]:
    by_observation: dict[int, List[CandidateMatch]] = {}
    sort_key_cache: dict[CandidateMatch, tuple] = {}
    for candidate in candidates:
        by_observation.setdefault(candidate.observation_index, []).append(candidate)

    ordered_observation_indices = sorted(by_observation)
    for observation_index in ordered_observation_indices:
        by_observation[observation_index].sort(
            key=lambda candidate: _cached_candidate_sort_key(candidate, sort_key_cache)
        )

    best_matches: List[CandidateMatch] | None = None

    def search(
        observation_position: int,
        used_tracks: Set[int],
        selected_matches: List[CandidateMatch],
    ) -> None:
        nonlocal best_matches

        remaining = len(ordered_observation_indices) - observation_position
        if best_matches is not None and len(selected_matches) + remaining < len(best_matches):
            return

        if observation_position >= len(ordered_observation_indices):
            if _better_assignment(selected_matches, best_matches, sort_key_cache):
                best_matches = list(selected_matches)
            return

        observation_index = ordered_observation_indices[observation_position]
        for candidate in by_observation[observation_index]:
            if candidate.track_index in used_tracks:
                continue
            used_tracks.add(candidate.track_index)
            selected_matches.append(candidate)
            search(observation_position + 1, used_tracks, selected_matches)
            selected_matches.pop()
            used_tracks.remove(candidate.track_index)

        search(observation_position + 1, used_tracks, selected_matches)

    search(0, set(), [])
    return best_matches or []


def assign_matches(
    ordered_tracks: Sequence[dict],
    ordered_observations: Sequence[dict],
    timestamp: float,
    config: TrackV2Config,
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    candidates = build_candidates(ordered_tracks, ordered_observations, timestamp, config)
    selected_candidates = _maximum_continuity_assignment(candidates)
    used_tracks = {candidate.track_index for candidate in selected_candidates}
    used_observations = {candidate.observation_index for candidate in selected_candidates}
    matches = [
        (candidate.track_index, candidate.observation_index)
        for candidate in sorted(selected_candidates, key=candidate_sort_key)
    ]

    unmatched_tracks = set(range(len(ordered_tracks))) - used_tracks
    unmatched_observations = set(range(len(ordered_observations))) - used_observations
    return matches, unmatched_tracks, unmatched_observations
