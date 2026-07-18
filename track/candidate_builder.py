"""Observation-specific Track V2 candidate packaging."""

import math
from dataclasses import dataclass
from typing import Sequence

from track.lifecycle import CONFIRMED_LIVE, CONFIRMED_MISSING, TENTATIVE, TrackStatus
from track.models import vector_values
from track.motion import MotionAssessment, assess_motion
from track.policy import TrackerPolicy

STRONG = "strong"
NORMAL = "normal"
WEAK = "weak"
IMPOSSIBLE = "impossible"

_CLASS_PENALTY = {STRONG: 0, NORMAL: 1, WEAK: 3}


@dataclass(frozen=True)
class CandidateMatch:
    track_index: int
    observation_index: int
    track_id: str
    detection_id: str
    motion_score: float
    appearance_score: float | None
    ownership_role: str
    classification: str
    distance_prediction: float
    distance_latest: float
    speed_required: float


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


def appearance_evidence(a, b) -> float | None:
    av = list(vector_values(a))
    bv = list(vector_values(b))
    if not av or not bv or len(av) != len(bv):
        return None
    return _embedding_similarity_values(av, bv)


def _classify_motion(motion: MotionAssessment, status: TrackStatus, policy: TrackerPolicy) -> str:
    if not motion.eligible:
        return IMPOSSIBLE
    if motion.motion_score <= 1.0:
        return NORMAL
    if status.confirmed and policy.allow_weak_confirmed_matching:
        return WEAK
    return IMPOSSIBLE


def _ownership_role(status: TrackStatus) -> str:
    if status.state == CONFIRMED_LIVE:
        return "protected_continuation"
    if status.state == CONFIRMED_MISSING:
        return "reassociation_continuation"
    if status.state == TENTATIVE:
        return "tentative_continuation"
    return "stale"


def build_candidate(
    track_index: int,
    observation_index: int,
    track: dict,
    observation: dict,
    status: TrackStatus,
    motion: MotionAssessment,
    policy: TrackerPolicy,
) -> CandidateMatch | None:
    classification = _classify_motion(motion, status, policy)
    if classification == IMPOSSIBLE:
        return None

    similarity = None
    if policy.appearance_tiebreak_enabled:
        similarity = appearance_evidence(track["best_crop"].get("embedding"), observation.get("embedding"))

    return CandidateMatch(
        track_index=track_index,
        observation_index=observation_index,
        track_id=str(track["track_id"]),
        detection_id=str(observation["detection_id"]),
        motion_score=float(motion.motion_score),
        appearance_score=similarity,
        ownership_role=_ownership_role(status),
        classification=classification,
        distance_prediction=float(motion.distance_prediction),
        distance_latest=float(motion.distance_latest),
        speed_required=float(motion.speed_required),
    )


def build_candidates(
    ordered_tracks: Sequence[dict],
    ordered_observations: Sequence[dict],
    timestamp: float,
    policy: TrackerPolicy,
    track_statuses: Sequence[TrackStatus],
) -> list[CandidateMatch]:
    """Build realistic track explanations independently for each observation."""

    if len(track_statuses) != len(ordered_tracks):
        raise ValueError("track_statuses length must match ordered_tracks length")

    candidates: list[CandidateMatch] = []
    for observation_index, observation in enumerate(ordered_observations):
        for track_index, track in enumerate(ordered_tracks):
            status = track_statuses[track_index]
            motion = assess_motion(track, status, observation, timestamp, policy)
            candidate = build_candidate(
                track_index,
                observation_index,
                track,
                observation,
                status,
                motion,
                policy,
            )
            if candidate is not None:
                candidates.append(candidate)

    return candidates
