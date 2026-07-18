"""Private Track V2 behaviour policy.

This is the only module that translates the public ``TrackV2Config`` surface,
including historical aliases, into the small behaviour object consumed by the
tracker internals.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any

from track.config import TrackV2Config


@dataclass(frozen=True)
class TrackerPolicy:
    """Single internal source of truth for Track V2 behaviour."""

    confirmation_hits: int
    detector_miss_tolerance_sec: float
    tentative_tolerance_sec: float
    motion_tolerance_px: float
    motion_tolerance_growth_px_per_sec: float
    max_physical_speed_px_per_sec: float
    localization_jitter_px: float
    continuity_strength: float
    takeover_margin: float
    weak_match_max_motion_score: float
    allow_weak_confirmed_matching: bool
    appearance_tiebreak_enabled: bool
    epsilon: float


def _first_configured(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (float, int)) or not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _non_negative(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _optional_non_negative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _non_negative(value, name)


def validate_public_config(config: TrackV2Config) -> None:
    """Validate the public config at the API boundary."""

    numeric_fields = (
        "confirmation_min_path_points",
        "tentative_track_window_sec",
        "confirmed_reassociation_window_sec",
        "confirmed_prediction_gate_px",
        "tentative_prediction_gate_px",
        "confirmed_latest_position_gate_px",
        "tentative_latest_position_gate_px",
        "confirmed_max_speed_px_per_sec",
        "tentative_max_speed_px_per_sec",
        "active_confirmation_min_path_points",
        "tentative_confirmation_min_path_points",
        "tentative_recency_window_frames",
        "max_reassociation_gap_sec",
        "max_speed_px_per_sec",
        "base_motion_gate_px",
        "strong_motion_threshold",
        "normal_motion_threshold",
        "weak_motion_threshold",
        "continuity_strength",
        "takeover_margin",
        "epsilon",
    )
    for field in numeric_fields:
        _non_negative(getattr(config, field), f"TrackV2Config.{field}")

    positive_fields = (
        "confirmation_min_path_points",
        "active_confirmation_min_path_points",
        "tentative_confirmation_min_path_points",
        "max_speed_px_per_sec",
        "base_motion_gate_px",
        "epsilon",
    )
    for field in positive_fields:
        if float(getattr(config, field)) <= 0:
            raise ValueError(f"TrackV2Config.{field} must be positive")

    optional_non_negative_fields = (
        "confirmation_hits",
        "detector_miss_tolerance_sec",
        "tentative_tolerance_sec",
        "motion_tolerance_px",
        "motion_tolerance_growth_px_per_sec",
        "confirmed_track_window_sec",
        "max_believable_speed_px_per_sec",
        "max_physical_speed_px_per_sec",
        "hard_speed_limit_px_per_sec",
        "prediction_gate_px",
        "prediction_gate_growth_px_per_sec",
        "latest_position_gate_px",
        "latest_position_gate_growth_px_per_sec",
        "localization_jitter_px",
        "jitter_tolerance_px",
        "forced_continuity_break_normalized_motion",
    )
    for field in optional_non_negative_fields:
        _optional_non_negative(getattr(config, field), f"TrackV2Config.{field}")

    if float(config.strong_motion_threshold) > float(config.normal_motion_threshold):
        raise ValueError("TrackV2Config.strong_motion_threshold must be <= normal_motion_threshold")
    if float(config.normal_motion_threshold) > float(config.weak_motion_threshold):
        raise ValueError("TrackV2Config.normal_motion_threshold must be <= weak_motion_threshold")


def _confirmation_hits(config: TrackV2Config) -> int:
    if config.confirmation_hits is not None:
        return max(1, int(config.confirmation_hits))
    return max(
        1,
        int(config.confirmation_min_path_points),
        int(config.active_confirmation_min_path_points),
        int(config.tentative_confirmation_min_path_points),
    )


def build_policy(config: TrackV2Config) -> TrackerPolicy:
    """Translate public/legacy config names into one private policy."""

    validate_public_config(config)
    detector_miss_tolerance_sec = _first_configured(
        config.detector_miss_tolerance_sec,
        config.confirmed_track_window_sec,
    )
    if detector_miss_tolerance_sec is None:
        detector_miss_tolerance_sec = min(
            float(config.confirmed_reassociation_window_sec),
            float(config.max_reassociation_gap_sec),
        )

    tentative_tolerance_sec = _first_configured(
        config.tentative_tolerance_sec,
        config.tentative_track_window_sec,
    )
    if tentative_tolerance_sec is None:
        tentative_tolerance_sec = config.tentative_recency_window_frames

    motion_tolerance_px = _first_configured(
        config.motion_tolerance_px,
        config.prediction_gate_px,
        config.latest_position_gate_px,
    )
    if motion_tolerance_px is None:
        motion_tolerance_px = max(
            float(config.confirmed_prediction_gate_px),
            float(config.confirmed_latest_position_gate_px),
        )

    motion_tolerance_growth_px_per_sec = _first_configured(
        config.motion_tolerance_growth_px_per_sec,
        config.prediction_gate_growth_px_per_sec,
        config.latest_position_gate_growth_px_per_sec,
    )
    if motion_tolerance_growth_px_per_sec is None:
        motion_tolerance_growth_px_per_sec = config.max_speed_px_per_sec

    max_physical_speed_px_per_sec = _first_configured(
        config.max_physical_speed_px_per_sec,
        config.hard_speed_limit_px_per_sec,
    )
    if max_physical_speed_px_per_sec is None:
        max_physical_speed_px_per_sec = config.confirmed_max_speed_px_per_sec

    localization_jitter_px = _first_configured(
        config.localization_jitter_px,
        config.jitter_tolerance_px,
    )
    if localization_jitter_px is None:
        localization_jitter_px = config.base_motion_gate_px

    policy = TrackerPolicy(
        confirmation_hits=_confirmation_hits(config),
        detector_miss_tolerance_sec=float(detector_miss_tolerance_sec),
        tentative_tolerance_sec=float(tentative_tolerance_sec),
        motion_tolerance_px=float(motion_tolerance_px),
        motion_tolerance_growth_px_per_sec=float(motion_tolerance_growth_px_per_sec),
        max_physical_speed_px_per_sec=float(max_physical_speed_px_per_sec),
        localization_jitter_px=float(localization_jitter_px),
        continuity_strength=float(config.continuity_strength),
        takeover_margin=float(config.takeover_margin),
        weak_match_max_motion_score=1.0 + float(config.takeover_margin),
        allow_weak_confirmed_matching=bool(config.allow_weak_confirmed_matching),
        appearance_tiebreak_enabled=bool(config.appearance_tiebreak_enabled),
        epsilon=float(config.epsilon),
    )
    validate_policy(policy)
    return policy


def validate_policy(policy: TrackerPolicy) -> None:
    """Fail loudly if normalized internal behaviour is malformed."""

    positive_fields = (
        "confirmation_hits",
        "max_physical_speed_px_per_sec",
        "epsilon",
    )
    non_negative_fields = (
        "detector_miss_tolerance_sec",
        "tentative_tolerance_sec",
        "motion_tolerance_px",
        "motion_tolerance_growth_px_per_sec",
        "localization_jitter_px",
        "continuity_strength",
        "takeover_margin",
        "weak_match_max_motion_score",
    )
    for field in positive_fields:
        value = _finite_number(getattr(policy, field), f"TrackerPolicy.{field}")
        if value <= 0:
            raise ValueError(f"TrackerPolicy.{field} must be positive")
    for field in non_negative_fields:
        _non_negative(getattr(policy, field), f"TrackerPolicy.{field}")
    if policy.weak_match_max_motion_score < 1.0:
        raise ValueError("TrackerPolicy.weak_match_max_motion_score must be >= 1.0")
