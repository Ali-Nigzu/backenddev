"""Private Track V2 behaviour policy.

This module is the single translation layer from the public, compatibility-heavy
``TrackV2Config`` surface to the small behaviour-oriented policy consumed by the
tracker internals. No motion, lifecycle, candidate, or assignment code should
read public config fields directly.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any

from track.config import TrackV2Config


@dataclass(frozen=True)
class TrackerPolicy:
    """Authoritative internal Track V2 behaviour settings."""

    confirmation_hits: int
    confirmed_max_missed_sec: float
    tentative_max_age_sec: float
    base_position_uncertainty_px: float
    miss_uncertainty_growth_px_per_sec: float
    localization_jitter_px: float
    max_speed_px_per_sec: float
    weak_confirmed_max_motion_score: float
    continuity_bias: float
    takeover_margin: float
    appearance_tiebreak_enabled: bool
    alpha_beta_position_gain: float
    alpha_beta_velocity_gain: float
    max_history_points: int
    epsilon: float

    @property
    def detector_miss_tolerance_sec(self) -> float:
        """Compatibility name for older private callers."""
        return self.confirmed_max_missed_sec

    @property
    def tentative_tolerance_sec(self) -> float:
        """Compatibility name for older private callers."""
        return self.tentative_max_age_sec

    @property
    def motion_tolerance_px(self) -> float:
        """Compatibility name for the base position uncertainty."""
        return self.base_position_uncertainty_px

    @property
    def motion_tolerance_growth_px_per_sec(self) -> float:
        """Compatibility name for miss uncertainty growth."""
        return self.miss_uncertainty_growth_px_per_sec

    @property
    def max_physical_speed_px_per_sec(self) -> float:
        """Compatibility name for hard physical speed."""
        return self.max_speed_px_per_sec

    @property
    def weak_match_max_motion_score(self) -> float:
        """Compatibility name for confirmed weak matching."""
        return self.weak_confirmed_max_motion_score

    @property
    def continuity_strength(self) -> float:
        """Compatibility name for assignment continuity bias."""
        return self.continuity_bias

    @property
    def allow_weak_confirmed_matching(self) -> bool:
        """Compatibility name retained for internal shims."""
        return self.weak_confirmed_max_motion_score > 1.0


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
    """Validate public config values without making them internal owners."""

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


def _spatial_uncertainty(config: TrackV2Config) -> float:
    configured = _first_configured(
        config.motion_tolerance_px,
        config.prediction_gate_px,
        config.latest_position_gate_px,
    )
    if configured is not None:
        return float(configured)
    return max(
        float(config.confirmed_prediction_gate_px),
        float(config.confirmed_latest_position_gate_px),
        float(config.base_motion_gate_px),
    )


def _miss_growth(config: TrackV2Config) -> float:
    configured = _first_configured(
        config.motion_tolerance_growth_px_per_sec,
        config.prediction_gate_growth_px_per_sec,
        config.latest_position_gate_growth_px_per_sec,
    )
    if configured is not None:
        return float(configured)
    # CCTV continuity should tolerate short misses, but uncertainty must not grow
    # as fast as the hard speed limit. Use a modest fraction of the base gate.
    return max(12.0, float(config.base_motion_gate_px) * 0.5)


def build_policy(config: TrackV2Config) -> TrackerPolicy:
    """Translate public/legacy config names into one private policy."""

    validate_public_config(config)
    confirmed_max_missed_sec = _first_configured(
        config.detector_miss_tolerance_sec,
        config.confirmed_track_window_sec,
    )
    if confirmed_max_missed_sec is None:
        confirmed_max_missed_sec = max(
            float(config.confirmed_reassociation_window_sec),
            float(config.max_reassociation_gap_sec),
        )

    tentative_max_age_sec = _first_configured(
        config.tentative_tolerance_sec,
        config.tentative_track_window_sec,
    )
    if tentative_max_age_sec is None:
        tentative_max_age_sec = config.tentative_recency_window_frames

    max_speed_px_per_sec = _first_configured(
        config.max_physical_speed_px_per_sec,
        config.hard_speed_limit_px_per_sec,
        config.max_believable_speed_px_per_sec,
    )
    if max_speed_px_per_sec is None:
        max_speed_px_per_sec = min(
            float(config.max_speed_px_per_sec),
            float(config.confirmed_max_speed_px_per_sec),
        )

    localization_jitter_px = _first_configured(
        config.localization_jitter_px,
        config.jitter_tolerance_px,
    )
    if localization_jitter_px is None:
        localization_jitter_px = max(8.0, float(config.base_motion_gate_px) * 0.35)

    weak_score = float(config.weak_motion_threshold)
    if not bool(config.allow_weak_confirmed_matching):
        weak_score = 1.0

    policy = TrackerPolicy(
        confirmation_hits=_confirmation_hits(config),
        confirmed_max_missed_sec=float(confirmed_max_missed_sec),
        tentative_max_age_sec=float(tentative_max_age_sec),
        base_position_uncertainty_px=_spatial_uncertainty(config),
        miss_uncertainty_growth_px_per_sec=_miss_growth(config),
        localization_jitter_px=float(localization_jitter_px),
        max_speed_px_per_sec=float(max_speed_px_per_sec),
        weak_confirmed_max_motion_score=max(1.0, weak_score),
        continuity_bias=float(config.continuity_strength),
        takeover_margin=float(config.takeover_margin),
        appearance_tiebreak_enabled=bool(config.appearance_tiebreak_enabled),
        alpha_beta_position_gain=0.65,
        alpha_beta_velocity_gain=0.18,
        max_history_points=12,
        epsilon=float(config.epsilon),
    )
    validate_policy(policy)
    return policy


def validate_policy(policy: TrackerPolicy) -> None:
    """Fail loudly if normalized internal behaviour is malformed."""

    positive_fields = (
        "confirmation_hits",
        "base_position_uncertainty_px",
        "max_speed_px_per_sec",
        "weak_confirmed_max_motion_score",
        "alpha_beta_position_gain",
        "alpha_beta_velocity_gain",
        "max_history_points",
        "epsilon",
    )
    non_negative_fields = (
        "confirmed_max_missed_sec",
        "tentative_max_age_sec",
        "miss_uncertainty_growth_px_per_sec",
        "localization_jitter_px",
        "continuity_bias",
        "takeover_margin",
    )
    for field in positive_fields:
        value = _finite_number(getattr(policy, field), f"TrackerPolicy.{field}")
        if value <= 0:
            raise ValueError(f"TrackerPolicy.{field} must be positive")
    for field in non_negative_fields:
        _non_negative(getattr(policy, field), f"TrackerPolicy.{field}")
    if policy.weak_confirmed_max_motion_score < 1.0:
        raise ValueError("TrackerPolicy.weak_confirmed_max_motion_score must be >= 1.0")
    if not 0.0 < policy.alpha_beta_position_gain <= 1.0:
        raise ValueError("TrackerPolicy.alpha_beta_position_gain must be in (0, 1]")
    if not 0.0 < policy.alpha_beta_velocity_gain <= 1.0:
        raise ValueError("TrackerPolicy.alpha_beta_velocity_gain must be in (0, 1]")
