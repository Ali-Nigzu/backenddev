import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from track.config import TrackV2Config
from track.models import RuntimeTrackV2
from track.motion import distance, predict_center


@dataclass(frozen=True)
class CandidateMatch:
    track_index: int
    observation_index: int
    motion_distance: float
    embedding_similarity: float


def _iter_values(vector) -> List[float]:
    if vector is None:
        return []
    try:
        return [float(v) for v in vector]
    except TypeError:
        return []


def embedding_similarity(a, b) -> float:
    av = _iter_values(a)
    bv = _iter_values(b)

    if not av or not bv or len(av) != len(bv):
        return 0.0

    dot = sum(x * y for x, y in zip(av, bv))
    norm_a = math.sqrt(sum(x * x for x in av))
    norm_b = math.sqrt(sum(y * y for y in bv))

    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0

    return dot / (norm_a * norm_b)


def motion_gate(track: RuntimeTrackV2, observation: Dict, config: TrackV2Config) -> Tuple[bool, float]:
    dt = float(observation["timestamp"]) - float(track.last_seen_timestamp)
    if dt < 0:
        return False, float("inf")

    predicted = predict_center(track.current_center, track.velocity, dt)
    motion_distance = distance(predicted, observation["center"])
    allowed_distance = config.gate_multiplier * (
        config.base_motion_gate + config.max_speed_px_per_sec * dt
    )

    if motion_distance > allowed_distance:
        return False, motion_distance

    if dt > 1e-9 and (motion_distance / dt) > config.max_speed_px_per_sec:
        return False, motion_distance

    return True, motion_distance


def build_candidates(
    tracks: Sequence[RuntimeTrackV2],
    observations: Sequence[Dict],
    config: TrackV2Config,
) -> Tuple[List[CandidateMatch], Set[int]]:
    candidates: List[CandidateMatch] = []
    gated_observation_indices: Set[int] = set()

    for track_index, track in enumerate(tracks):
        for observation_index, observation in enumerate(observations):
            allowed, motion_distance = motion_gate(track, observation, config)
            if not allowed:
                continue

            gated_observation_indices.add(observation_index)
            candidates.append(
                CandidateMatch(
                    track_index=track_index,
                    observation_index=observation_index,
                    motion_distance=motion_distance,
                    embedding_similarity=embedding_similarity(
                        track.last_embedding,
                        observation.get("embedding"),
                    ),
                )
            )

    return candidates, gated_observation_indices


def _choose_from_ambiguous(
    best: CandidateMatch,
    candidates: Iterable[CandidateMatch],
    config: TrackV2Config,
) -> CandidateMatch:
    ambiguous = [
        candidate for candidate in candidates
        if (
            candidate.track_index == best.track_index
            or candidate.observation_index == best.observation_index
        )
        and abs(candidate.motion_distance - best.motion_distance) <= config.motion_ambiguity_delta
    ]

    if len(ambiguous) <= 1:
        return best

    strongest = max(ambiguous, key=lambda item: item.embedding_similarity)
    if strongest.embedding_similarity - best.embedding_similarity >= config.embedding_tie_threshold:
        return strongest

    return best


def assign_matches(
    tracks: Sequence[RuntimeTrackV2],
    observations: Sequence[Dict],
    config: TrackV2Config,
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int], Set[int]]:
    candidates, gated_observation_indices = build_candidates(tracks, observations, config)

    matches: List[Tuple[int, int]] = []
    used_tracks: Set[int] = set()
    used_observations: Set[int] = set()
    remaining = list(candidates)

    while remaining:
        remaining.sort(key=lambda item: (item.motion_distance, item.track_index, item.observation_index))
        best = remaining[0]
        chosen = _choose_from_ambiguous(best, remaining, config)

        if chosen.track_index not in used_tracks and chosen.observation_index not in used_observations:
            matches.append((chosen.track_index, chosen.observation_index))
            used_tracks.add(chosen.track_index)
            used_observations.add(chosen.observation_index)

        remaining = [
            candidate for candidate in remaining
            if candidate.track_index not in used_tracks
            and candidate.observation_index not in used_observations
        ]

    unmatched_tracks = set(range(len(tracks))) - used_tracks
    unmatched_observations = set(range(len(observations))) - used_observations

    return matches, unmatched_tracks, unmatched_observations, gated_observation_indices
