"""Deterministic continuity-first Track V2 assignment.

Assignment receives only motion-eligible candidates. It owns continuity bias and
takeover policy, but it never creates eligibility for impossible motion.
"""

from typing import Sequence, Tuple

from track.candidate_builder import (
    CandidateMatch,
    PROTECTED_CONTINUATION,
    REASSOCIATION_CONTINUATION,
    TENTATIVE_CONTINUATION,
    _CLASS_PENALTY,
    numeric_track_id,
)
from track.policy import TrackerPolicy

_ROLE_PRIORITY = {
    PROTECTED_CONTINUATION: 0,
    REASSOCIATION_CONTINUATION: 1,
    TENTATIVE_CONTINUATION: 2,
}


def _role_priority(candidate: CandidateMatch) -> int:
    return _ROLE_PRIORITY.get(candidate.ownership_role, 99)


def _appearance_cost(candidate: CandidateMatch) -> float:
    if candidate.appearance_score is None:
        return 1.0
    return 1.0 - float(candidate.appearance_score)


def _track_key(track_id: str) -> tuple:
    numeric = numeric_track_id(track_id)
    return (0, numeric, track_id) if numeric is not None else (1, track_id)


def _continuity_adjusted_motion(candidate: CandidateMatch, config: TrackerPolicy) -> float:
    # Takeover margin and bias affect choice among already plausible candidates
    # only. A protected incumbent can be beaten, but a challenger must be
    # clearly better in motion score instead of microscopically better.
    if candidate.ownership_role == PROTECTED_CONTINUATION:
        return max(0.0, candidate.motion_score / (1.0 + config.takeover_margin) - config.continuity_bias)
    if candidate.ownership_role == REASSOCIATION_CONTINUATION:
        margin = 1.0 + (config.takeover_margin * 0.5)
        return max(0.0, candidate.motion_score / margin - (config.continuity_bias * 0.5))
    return candidate.motion_score


def candidate_sort_key(candidate: CandidateMatch) -> tuple:
    return (
        _role_priority(candidate),
        _CLASS_PENALTY.get(candidate.classification, 99),
        round(candidate.motion_score, 12),
        round(candidate.distance_prediction, 12),
        round(candidate.distance_latest, 12),
        round(candidate.speed_required, 12),
        round(_appearance_cost(candidate), 12),
        _track_key(candidate.track_id),
        candidate.detection_id,
        candidate.observation_index,
    )


def _assignment_cost(candidate: CandidateMatch, config: TrackerPolicy) -> tuple:
    return (
        _CLASS_PENALTY.get(candidate.classification, 99),
        round(_continuity_adjusted_motion(candidate, config), 12),
        round(candidate.motion_score, 12),
        _role_priority(candidate),
        round(candidate.distance_prediction, 12),
        round(candidate.distance_latest, 12),
        round(_appearance_cost(candidate), 12),
        _track_key(candidate.track_id),
        candidate.detection_id,
        candidate.observation_index,
    )


def _optimal_assignment(candidates: Sequence[CandidateMatch], config: TrackerPolicy) -> list[CandidateMatch]:
    """Select deterministic one-to-one matches without exponential search."""

    selected: list[CandidateMatch] = []
    used_tracks: set[int] = set()
    used_observations: set[int] = set()
    for candidate in sorted(candidates, key=lambda candidate: _assignment_cost(candidate, config)):
        if candidate.track_index in used_tracks or candidate.observation_index in used_observations:
            continue
        selected.append(candidate)
        used_tracks.add(candidate.track_index)
        used_observations.add(candidate.observation_index)
    return selected


def _validate_assignment(
    matches: list[tuple[int, int]],
    candidates: Sequence[CandidateMatch],
    track_count: int,
    observation_count: int,
) -> None:
    candidate_pairs = {(candidate.track_index, candidate.observation_index) for candidate in candidates}
    if len(candidate_pairs) != len(candidates):
        raise ValueError("duplicate candidate pair supplied to assignment")
    seen_tracks: set[int] = set()
    seen_observations: set[int] = set()
    for track_index, observation_index in matches:
        if track_index < 0 or track_index >= track_count:
            raise ValueError("assignment references nonexistent track")
        if observation_index < 0 or observation_index >= observation_count:
            raise ValueError("assignment references nonexistent observation")
        if (track_index, observation_index) not in candidate_pairs:
            raise ValueError("assignment references a pair that was not an eligible candidate")
        if track_index in seen_tracks:
            raise ValueError("multiple assignments for one track")
        if observation_index in seen_observations:
            raise ValueError("multiple assignments for one observation")
        seen_tracks.add(track_index)
        seen_observations.add(observation_index)


def assign_candidates(
    candidates: Sequence[CandidateMatch],
    track_count: int,
    observation_count: int,
    config: TrackerPolicy,
    max_births_allowed: int | None = None,
) -> Tuple[list[tuple[int, int]], set[int], set[int]]:
    if max_births_allowed is None:
        max_births_allowed = observation_count
    if max_births_allowed < 0 or max_births_allowed > observation_count:
        raise ValueError("max_births_allowed must be within observation count")

    required_matches = observation_count - max_births_allowed
    selected = _optimal_assignment(candidates, config)
    if len({candidate.observation_index for candidate in selected}) < required_matches:
        raise ValueError("assignment failed to satisfy birth suppression match requirement")
    used_tracks = {candidate.track_index for candidate in selected}
    used_observations = {candidate.observation_index for candidate in selected}
    matches = [
        (candidate.track_index, candidate.observation_index)
        for candidate in sorted(selected, key=lambda candidate: _assignment_cost(candidate, config))
    ]

    unmatched_tracks = set(range(track_count)) - used_tracks
    unmatched_observations = set(range(observation_count)) - used_observations
    if not unmatched_tracks.issubset(set(range(track_count))):
        raise ValueError("invalid unmatched track index")
    if not unmatched_observations.issubset(set(range(observation_count))):
        raise ValueError("invalid unmatched observation index")
    if len(unmatched_observations) > max_births_allowed:
        raise ValueError("assignment produced more births than birth suppression allows")
    _validate_assignment(matches, candidates, track_count, observation_count)
    return matches, unmatched_tracks, unmatched_observations
