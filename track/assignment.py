"""Deterministic continuity-first Track V2 assignment.

Assignment owns the identity decision: coverage first, deterministic cost second.
Normal candidates carry motion evidence; suppression coverage candidates carry the
explicit continuity requirement used before births are allowed.
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


def _assignment_cost(candidate: CandidateMatch, config: TrackerPolicy) -> tuple:
    return (
        round(candidate.distance_latest, 12),
        _role_priority(candidate),
        _CLASS_PENALTY.get(candidate.classification, 99),
        round(_continuity_adjusted_motion(candidate, config), 12),
        round(candidate.motion_score, 12),
        round(candidate.distance_prediction, 12),
        round(candidate.speed_required, 12),
        round(_appearance_cost(candidate), 12),
        _track_key(candidate.track_id),
        candidate.detection_id,
        candidate.observation_index,
    )


def _maximum_coverage_assignment(candidates: Sequence[CandidateMatch], config: TrackerPolicy) -> list[CandidateMatch]:
    """Select deterministic one-to-one matches with coverage as the first objective.

    Birth suppression makes unmatched observations expensive. This solver therefore
    finds a maximum-cardinality bipartite matching first, using deterministic
    candidate costs only to choose among equally coverable explanations. The result
    keeps births as an overflow outcome instead of a side effect of greedy local
    choices.
    """

    by_observation: dict[int, list[CandidateMatch]] = {}
    for candidate in candidates:
        by_observation.setdefault(candidate.observation_index, []).append(candidate)
    for observation_candidates in by_observation.values():
        observation_candidates.sort(key=lambda candidate: _assignment_cost(candidate, config))

    observation_order = sorted(
        by_observation,
        key=lambda observation_index: (
            len(by_observation[observation_index]),
            min(_assignment_cost(candidate, config) for candidate in by_observation[observation_index]),
            observation_index,
        ),
    )
    track_to_candidate: dict[int, CandidateMatch] = {}

    def try_assign(observation_index: int, visited_tracks: set[int]) -> bool:
        for candidate in by_observation[observation_index]:
            if candidate.track_index in visited_tracks:
                continue
            visited_tracks.add(candidate.track_index)
            incumbent = track_to_candidate.get(candidate.track_index)
            if incumbent is None or try_assign(incumbent.observation_index, visited_tracks):
                track_to_candidate[candidate.track_index] = candidate
                return True
        return False

    for observation_index in observation_order:
        try_assign(observation_index, set())

    selected = list(track_to_candidate.values())
    return sorted(selected, key=lambda candidate: _assignment_cost(candidate, config))


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
    selected = _maximum_coverage_assignment(candidates, config)
    if len(selected) > required_matches:
        removable = sorted(
            [candidate for candidate in selected if candidate.fallback],
            key=lambda candidate: _assignment_cost(candidate, config),
            reverse=True,
        )
        selected_set = set(selected)
        for candidate in removable:
            if len(selected_set) <= required_matches:
                break
            selected_set.remove(candidate)
        selected = sorted(selected_set, key=lambda candidate: _assignment_cost(candidate, config))
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
