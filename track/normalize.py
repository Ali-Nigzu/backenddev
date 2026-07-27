"""Compatibility normalization helpers for Track V2 configuration."""

from dataclasses import dataclass

from track.config import TrackV2Config
from track.policy import build_policy


@dataclass(frozen=True)
class NormalizedTrackV2Config:
    confirmation_hits: int
    detector_miss_tolerance_sec: float
    tentative_tolerance_sec: float
    motion_tolerance_px: float
    motion_tolerance_growth_px_per_sec: float
    max_physical_speed_px_per_sec: float
    localization_jitter_px: float
    location_history_window_frames: int
    continuity_strength: float
    takeover_margin: float


def _normalize_config(config: TrackV2Config) -> NormalizedTrackV2Config:
    """Return the stable compatibility view of public Track V2 config."""

    policy = build_policy(config)
    return NormalizedTrackV2Config(
        confirmation_hits=policy.confirmation_hits,
        detector_miss_tolerance_sec=policy.confirmed_max_missed_sec,
        tentative_tolerance_sec=policy.tentative_max_age_sec,
        motion_tolerance_px=policy.base_position_uncertainty_px,
        motion_tolerance_growth_px_per_sec=policy.miss_uncertainty_growth_px_per_sec,
        max_physical_speed_px_per_sec=policy.max_speed_px_per_sec,
        localization_jitter_px=policy.localization_jitter_px,
        location_history_window_frames=policy.location_history_window_frames,
        continuity_strength=policy.continuity_bias,
        takeover_margin=policy.takeover_margin,
    )
