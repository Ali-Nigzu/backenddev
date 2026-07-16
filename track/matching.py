"""Deterministic Track V2 matching with observation-specific eligibility."""

import math
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from track.config import TrackV2Config
from track.models import vector_values
from track.motion import derive_velocity, distance

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
    motion_distance: float
    normalized_motion: float
    physically_plausible: bool
    appearance_similarity: float | None
    appearance_tiebreak_cost: float
    classification: str = NORMAL
    timestamp_gap: float = 0.0
    predicted_distance: float = 0.0
    latest_distance: float = 0.0
    required_speed: float = 0.0
    motion_score: float = 0.0
    track_state: str = "confirmed"


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


def _config_value(config: TrackV2Config, primary: str, fallback: str):
    value = getattr(config, primary)
    return getattr(config, fallback) if value is None else value


def _confirmation_min_path_points(config: TrackV2Config) -> int:
    return max(
        1,
        int(config.confirmation_min_path_points),
        int(config.active_confirmation_min_path_points),
        int(config.tentative_confirmation_min_path_points),
    )


def _is_confirmed(track, config: TrackV2Config) -> bool:
    return len(track["path"]) >= _confirmation_min_path_points(config)


def _track_window(track, config: TrackV2Config) -> float:
    if _is_confirmed(track, config):
        compatibility_window = float(config.max_reassociation_gap_sec)
        if config.confirmed_track_window_sec is not None:
            return min(float(config.confirmed_track_window_sec), compatibility_window)
        return min(float(config.confirmed_reassociation_window_sec), compatibility_window)
    return float(config.tentative_track_window_sec or config.tentative_recency_window_frames)


def _xy(center) -> tuple[float, float]:
    return float(center["x"]), float(center["y"])


def _predicted_center(track, timestamp: float, config: TrackV2Config) -> dict:
    latest = track["path"][-1]
    latest_x, latest_y = _xy(latest["center"])
    dt = timestamp - float(latest["timestamp"])
    if dt <= config.epsilon:
        return {"x": latest_x, "y": latest_y}
    velocity = derive_velocity(track["path"], config)
    return {
        "x": latest_x + float(velocity["x"]) * dt,
        "y": latest_y + float(velocity["y"]) * dt,
    }


def _classify_motion(score: float, confirmed: bool, config: TrackV2Config) -> str:
    if score <= float(config.strong_motion_threshold):
        return STRONG
    if score <= float(config.normal_motion_threshold):
        return NORMAL
    if score <= float(config.weak_motion_threshold):
        if confirmed and config.allow_weak_confirmed_matching:
            return WEAK
        if not confirmed and config.allow_weak_tentative_matching:
            return WEAK
    return IMPOSSIBLE


def evaluate_track_observation(
    track: dict,
    observation: dict,
    timestamp: float,
    config: TrackV2Config,
) -> dict:
    """Return deterministic eligibility metrics for one track-observation pair."""

    latest_timestamp = float(track["path"][-1]["timestamp"])
    gap = float(timestamp) - latest_timestamp
    confirmed = _is_confirmed(track, config)
    state = "confirmed" if confirmed else "tentative"

    base_metrics = {
        "timestamp_gap": gap,
        "predicted_distance": float("inf"),
        "latest_distance": float("inf"),
        "required_speed": float("inf"),
        "motion_score": float("inf"),
        "track_state": state,
    }

    if gap < -config.epsilon:
        return {
            "eligible": False,
            "classification": IMPOSSIBLE,
            "reason": "negative_time_gap",
            "metrics": base_metrics,
        }

    if not confirmed and not config.allow_tentative_matching:
        return {
            "eligible": False,
            "classification": IMPOSSIBLE,
            "reason": "tentative_matching_disabled",
            "metrics": base_metrics,
        }

    allowed_window = _track_window(track, config)
    if gap > allowed_window:
        return {
            "eligible": False,
            "classification": IMPOSSIBLE,
            "reason": "outside_track_window",
            "metrics": base_metrics,
        }

    predicted = _predicted_center(track, timestamp, config)
    predicted_distance = distance(predicted, observation["center"])
    latest_distance = distance(track["path"][-1]["center"], observation["center"])
    safe_gap = max(gap, config.epsilon)
    required_speed = latest_distance / safe_gap

    hard_speed_limit = float(_config_value(config, "hard_speed_limit_px_per_sec", "max_speed_px_per_sec"))
    if required_speed > hard_speed_limit:
        return {
            "eligible": False,
            "classification": IMPOSSIBLE,
            "reason": "speed_limit",
            "metrics": {
                **base_metrics,
                "predicted_distance": predicted_distance,
                "latest_distance": latest_distance,
                "required_speed": required_speed,
                "motion_score": float("inf"),
            },
        }

    max_believable_speed = float(
        _config_value(config, "max_believable_speed_px_per_sec", "max_speed_px_per_sec")
    )
    prediction_gate_px = float(_config_value(config, "prediction_gate_px", "base_motion_gate_px"))
    prediction_growth = float(
        _config_value(config, "prediction_gate_growth_px_per_sec", "max_speed_px_per_sec")
    )
    latest_gate_px = float(_config_value(config, "latest_position_gate_px", "base_motion_gate_px"))
    latest_growth = float(
        _config_value(config, "latest_position_gate_growth_px_per_sec", "max_speed_px_per_sec")
    )
    jitter = float(_config_value(config, "jitter_tolerance_px", "base_motion_gate_px"))

    maturity_scale = 1.0 if confirmed else 0.75
    prediction_gate = (prediction_gate_px + prediction_growth * max(gap, 0.0) + jitter) * maturity_scale
    latest_gate = (latest_gate_px + latest_growth * max(gap, 0.0) + jitter) * maturity_scale

    prediction_ratio = predicted_distance / max(prediction_gate, config.epsilon)
    latest_ratio = latest_distance / max(latest_gate, config.epsilon)
    speed_ratio = required_speed / max(max_believable_speed, config.epsilon)
    motion_score = min(prediction_ratio, latest_ratio, speed_ratio)

    position_allowed = prediction_ratio <= float(config.weak_motion_threshold) or latest_ratio <= float(
        config.weak_motion_threshold
    )
    speed_allowed = speed_ratio <= float(config.weak_motion_threshold)
    classification = _classify_motion(motion_score, confirmed, config)

    if not position_allowed:
        reason = "position_gate"
    elif not speed_allowed:
        reason = "believable_speed"
    elif classification == IMPOSSIBLE:
        reason = "weak_matching_disabled"
    else:
        reason = "eligible"

    eligible = reason == "eligible"
    if not eligible:
        classification = IMPOSSIBLE

    return {
        "eligible": eligible,
        "classification": classification,
        "reason": reason,
        "metrics": {
            **base_metrics,
            "predicted_distance": predicted_distance,
            "latest_distance": latest_distance,
            "required_speed": required_speed,
            "motion_score": motion_score,
            "prediction_ratio": prediction_ratio,
            "latest_ratio": latest_ratio,
            "speed_ratio": speed_ratio,
        },
    }


def build_candidates(
    ordered_tracks: Sequence[dict],
    ordered_observations: Sequence[dict],
    timestamp: float,
    config: TrackV2Config,
    diagnostics: list | None = None,
) -> List[CandidateMatch]:
    candidates: List[CandidateMatch] = []
    observation_diagnostics: list[dict] = []

    for observation_index, observation in enumerate(ordered_observations):
        candidate_details: list[dict] = []
        for track_index, track in enumerate(ordered_tracks):
            evaluation = evaluate_track_observation(track, observation, timestamp, config)
            metrics = evaluation["metrics"]
            candidate_details.append(
                {
                    "track_id": str(track["track_id"]),
                    "eligible": bool(evaluation["eligible"]),
                    "classification": evaluation["classification"],
                    "reason": evaluation["reason"],
                    "metrics": dict(metrics),
                }
            )
            if not evaluation["eligible"]:
                continue

            similarity = appearance_evidence(
                track["best_crop"].get("embedding"), observation.get("embedding")
            )
            appearance_tiebreak_cost = 1.0 - similarity if similarity is not None else 1.0
            motion_distance = min(metrics["predicted_distance"], metrics["latest_distance"])

            candidates.append(
                CandidateMatch(
                    track_index=track_index,
                    observation_index=observation_index,
                    track_id=str(track["track_id"]),
                    detection_id=str(observation["detection_id"]),
                    motion_distance=motion_distance,
                    normalized_motion=float(metrics["motion_score"]),
                    physically_plausible=True,
                    appearance_similarity=similarity,
                    appearance_tiebreak_cost=appearance_tiebreak_cost,
                    classification=evaluation["classification"],
                    timestamp_gap=float(metrics["timestamp_gap"]),
                    predicted_distance=float(metrics["predicted_distance"]),
                    latest_distance=float(metrics["latest_distance"]),
                    required_speed=float(metrics["required_speed"]),
                    motion_score=float(metrics["motion_score"]),
                    track_state=str(metrics["track_state"]),
                )
            )
        observation_diagnostics.append(
            {
                "observation_id": str(observation["detection_id"]),
                "observation_index": observation_index,
                "candidate_tracks": candidate_details,
                "final_assignment": None,
            }
        )

    if diagnostics is not None:
        diagnostics.extend(observation_diagnostics)
    return candidates


def candidate_sort_key(candidate: CandidateMatch) -> tuple:
    numeric = numeric_track_id(candidate.track_id)
    track_key = (0, numeric, candidate.track_id) if numeric is not None else (1, candidate.track_id)
    return (
        _CLASS_PENALTY.get(candidate.classification, 99),
        0 if candidate.track_state == "confirmed" else 1,
        round(candidate.normalized_motion, 12),
        round(candidate.predicted_distance, 12),
        round(candidate.latest_distance, 12),
        round(candidate.required_speed, 12),
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
        ordered = sorted(matches, key=lambda match: _cached_candidate_sort_key(match, sort_key_cache))
        return (
            -len(ordered),
            sum(_CLASS_PENALTY.get(match.classification, 99) for match in ordered),
            sum(0 if match.track_state == "confirmed" else 1 for match in ordered),
            round(sum(match.normalized_motion for match in ordered), 12),
            round(sum(match.predicted_distance for match in ordered), 12),
            round(sum(match.latest_distance for match in ordered), 12),
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


def _record_assignments(diagnostics: list | None, selected_candidates: Sequence[CandidateMatch]) -> None:
    if diagnostics is None:
        return
    by_observation = {candidate.observation_index: candidate for candidate in selected_candidates}
    for item in diagnostics:
        candidate = by_observation.get(item["observation_index"])
        if candidate is None:
            item["final_assignment"] = "birth"
        else:
            item["final_assignment"] = {
                "track_id": candidate.track_id,
                "classification": candidate.classification,
                "motion_score": candidate.motion_score,
            }


def assign_matches(
    ordered_tracks: Sequence[dict],
    ordered_observations: Sequence[dict],
    timestamp: float,
    config: TrackV2Config,
    diagnostics: list | None = None,
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int], Set[int]]:
    candidates = build_candidates(ordered_tracks, ordered_observations, timestamp, config, diagnostics)
    selected_candidates = _maximum_continuity_assignment(candidates)
    _record_assignments(diagnostics, selected_candidates)

    used_tracks = {candidate.track_index for candidate in selected_candidates}
    used_observations = {candidate.observation_index for candidate in selected_candidates}
    matches = [
        (candidate.track_index, candidate.observation_index)
        for candidate in sorted(selected_candidates, key=candidate_sort_key)
    ]

    unmatched_tracks = set(range(len(ordered_tracks))) - used_tracks
    unmatched_observations = set(range(len(ordered_observations))) - used_observations
    return matches, unmatched_tracks, unmatched_observations, set()
