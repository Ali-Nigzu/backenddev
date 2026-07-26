"""Deterministic continuity-first Track V2 assignment.

Assignment receives only motion-eligible candidates. It owns continuity bias and
takeover policy, but it never creates eligibility for impossible motion.
"""

from typing import Sequence, Set, Tuple

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


def _better_assignment(
    candidate_matches: list[CandidateMatch],
    best_matches: list[CandidateMatch] | None,
    config: TrackerPolicy,
) -> bool:
    if best_matches is None:
        return True

    def score(matches: list[CandidateMatch]) -> tuple:
        ordered = sorted(matches, key=lambda match: _assignment_cost(match, config))
        return (
            -len(ordered),
            sum(_CLASS_PENALTY.get(match.classification, 99) for match in ordered),
            round(sum(_continuity_adjusted_motion(match, config) for match in ordered), 12),
            sum(_role_priority(match) for match in ordered),
            round(sum(match.motion_score for match in ordered), 12),
            round(sum(_appearance_cost(match) for match in ordered), 12),
            tuple(_assignment_cost(match, config) for match in ordered),
        )

    return score(candidate_matches) < score(best_matches)


def _optimal_assignment(candidates: Sequence[CandidateMatch], config: TrackerPolicy) -> list[CandidateMatch]:
    by_observation: dict[int, list[CandidateMatch]] = {}
    for candidate in candidates:
        by_observation.setdefault(candidate.observation_index, []).append(candidate)

    ordered_observation_indices = sorted(by_observation)
    for observation_index in ordered_observation_indices:
        by_observation[observation_index].sort(key=lambda candidate: _assignment_cost(candidate, config))

    best_matches: list[CandidateMatch] | None = None

    def search(observation_position: int, used_tracks: Set[int], selected_matches: list[CandidateMatch]) -> None:
        nonlocal best_matches
        remaining = len(ordered_observation_indices) - observation_position
        if best_matches is not None and len(selected_matches) + remaining < len(best_matches):
            return
        if observation_position >= len(ordered_observation_indices):
            if _better_assignment(selected_matches, best_matches, config):
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
) -> Tuple[list[tuple[int, int]], Set[int], Set[int]]:
    selected = _optimal_assignment(candidates, config)
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
    _validate_assignment(matches, candidates, track_count, observation_count)
    return matches, unmatched_tracks, unmatched_observations
