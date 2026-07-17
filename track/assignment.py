"""Deterministic continuity-first Track V2 assignment."""

from typing import Sequence, Set, Tuple

from track.candidate_builder import CandidateMatch, _CLASS_PENALTY, numeric_track_id
from track.normalize import _NormalizedTrackConfig


def _role_priority(candidate: CandidateMatch) -> int:
    if candidate.ownership_role == "protected_continuation":
        return 0
    if candidate.ownership_role == "reassociation_continuation":
        return 1
    if candidate.ownership_role == "tentative_continuation":
        return 2
    return 3


def _appearance_cost(candidate: CandidateMatch) -> float:
    if candidate.appearance_score is None:
        return 1.0
    return 1.0 - float(candidate.appearance_score)


def candidate_sort_key(candidate: CandidateMatch) -> tuple:
    numeric = numeric_track_id(candidate.track_id)
    track_key = (0, numeric, candidate.track_id) if numeric is not None else (1, candidate.track_id)
    return (
        _role_priority(candidate),
        _CLASS_PENALTY.get(candidate.classification, 99),
        round(candidate.motion_score, 12),
        round(candidate.distance_prediction, 12),
        round(candidate.distance_latest, 12),
        round(candidate.speed_required, 12),
        round(_appearance_cost(candidate), 12),
        track_key,
        candidate.detection_id,
        candidate.observation_index,
    )


def _track_claim_key(candidate: CandidateMatch) -> tuple:
    return (
        _role_priority(candidate),
        _CLASS_PENALTY.get(candidate.classification, 99),
        round(candidate.motion_score, 12),
        round(candidate.distance_prediction, 12),
        round(candidate.distance_latest, 12),
        round(_appearance_cost(candidate), 12),
        candidate.detection_id,
        candidate.observation_index,
    )


def _is_defensible_first_claim(
    candidate: CandidateMatch,
    by_observation: dict[int, list[CandidateMatch]],
    config: _NormalizedTrackConfig,
) -> bool:
    peers = by_observation.get(candidate.observation_index, ())
    if not peers:
        return True
    best_peer_cost = min(peer.motion_score for peer in peers)
    allowed_cost = best_peer_cost * (1.0 + config.takeover_margin) + config.continuity_strength
    return candidate.motion_score <= allowed_cost


def _claim_role(
    role: str,
    by_track: dict[int, list[CandidateMatch]],
    by_observation: dict[int, list[CandidateMatch]],
    selected: list[CandidateMatch],
    used_tracks: Set[int],
    used_observations: Set[int],
    config: _NormalizedTrackConfig,
) -> None:
    for track_index in sorted(by_track):
        if track_index in used_tracks:
            continue
        track_candidates = [
            candidate
            for candidate in by_track[track_index]
            if candidate.ownership_role == role
            and candidate.observation_index not in used_observations
            and _is_defensible_first_claim(candidate, by_observation, config)
        ]
        if not track_candidates:
            continue
        candidate = min(track_candidates, key=_track_claim_key)
        selected.append(candidate)
        used_tracks.add(candidate.track_index)
        used_observations.add(candidate.observation_index)


def _better_assignment(candidate_matches: list[CandidateMatch], best_matches: list[CandidateMatch] | None) -> bool:
    if best_matches is None:
        return True

    def score(matches: list[CandidateMatch]) -> tuple:
        ordered = sorted(matches, key=candidate_sort_key)
        return (
            -len(ordered),
            sum(_role_priority(match) for match in ordered),
            sum(_CLASS_PENALTY.get(match.classification, 99) for match in ordered),
            round(sum(match.motion_score for match in ordered), 12),
            round(sum(_appearance_cost(match) for match in ordered), 12),
            tuple(candidate_sort_key(match) for match in ordered),
        )

    return score(candidate_matches) < score(best_matches)


def _optimal_remaining_assignment(candidates: list[CandidateMatch]) -> list[CandidateMatch]:
    by_observation: dict[int, list[CandidateMatch]] = {}
    for candidate in candidates:
        by_observation.setdefault(candidate.observation_index, []).append(candidate)

    ordered_observation_indices = sorted(by_observation)
    for observation_index in ordered_observation_indices:
        by_observation[observation_index].sort(key=candidate_sort_key)

    best_matches: list[CandidateMatch] | None = None

    def search(observation_position: int, used_tracks: Set[int], selected_matches: list[CandidateMatch]) -> None:
        nonlocal best_matches
        remaining = len(ordered_observation_indices) - observation_position
        if best_matches is not None and len(selected_matches) + remaining < len(best_matches):
            return
        if observation_position >= len(ordered_observation_indices):
            if _better_assignment(selected_matches, best_matches):
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


def assign_candidates(
    candidates: Sequence[CandidateMatch],
    track_count: int,
    observation_count: int,
    config: _NormalizedTrackConfig,
) -> Tuple[list[tuple[int, int]], Set[int], Set[int], Set[int]]:
    by_track: dict[int, list[CandidateMatch]] = {}
    by_observation: dict[int, list[CandidateMatch]] = {}
    for candidate in candidates:
        by_track.setdefault(candidate.track_index, []).append(candidate)
        by_observation.setdefault(candidate.observation_index, []).append(candidate)

    selected: list[CandidateMatch] = []
    used_tracks: Set[int] = set()
    used_observations: Set[int] = set()

    _claim_role("protected_continuation", by_track, by_observation, selected, used_tracks, used_observations, config)
    _claim_role("reassociation_continuation", by_track, by_observation, selected, used_tracks, used_observations, config)

    remaining = [
        candidate
        for candidate in candidates
        if candidate.track_index not in used_tracks and candidate.observation_index not in used_observations
    ]
    selected.extend(_optimal_remaining_assignment(remaining))

    used_tracks = {candidate.track_index for candidate in selected}
    used_observations = {candidate.observation_index for candidate in selected}
    matches = [
        (candidate.track_index, candidate.observation_index)
        for candidate in sorted(selected, key=candidate_sort_key)
    ]

    unmatched_tracks = set(range(track_count)) - used_tracks
    unmatched_observations = set(range(observation_count)) - used_observations
    return matches, unmatched_tracks, unmatched_observations, set()
