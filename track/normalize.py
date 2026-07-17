"""Private Track V2 configuration normalization."""

from dataclasses import dataclass

from track.config import TrackV2Config


@dataclass(frozen=True)
class _NormalizedTrackConfig:
    """Behavioural configuration consumed by Track V2 internals only."""

    confirmation_hits: int
    detector_miss_tolerance_sec: float
    tentative_tolerance_sec: float
    motion_tolerance_px: float
    motion_tolerance_growth_px_per_sec: float
    max_physical_speed_px_per_sec: float
    localization_jitter_px: float
    continuity_strength: float
    takeover_margin: float
    allow_weak_confirmed_matching: bool
    appearance_tiebreak_enabled: bool
    epsilon: float


def _first_configured(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _confirmation_hits(config: TrackV2Config) -> int:
    if config.confirmation_hits is not None:
        return max(1, int(config.confirmation_hits))
    return max(
        1,
        int(config.confirmation_min_path_points),
        int(config.active_confirmation_min_path_points),
        int(config.tentative_confirmation_min_path_points),
    )


def _detector_miss_tolerance_sec(config: TrackV2Config) -> float:
    explicit = _first_configured(config.detector_miss_tolerance_sec, config.confirmed_track_window_sec)
    if explicit is not None:
        return float(explicit)
    return min(float(config.confirmed_reassociation_window_sec), float(config.max_reassociation_gap_sec))


def _tentative_tolerance_sec(config: TrackV2Config) -> float:
    explicit = _first_configured(config.tentative_tolerance_sec, config.tentative_track_window_sec)
    if explicit is not None:
        return float(explicit)
    return float(config.tentative_recency_window_frames)


def _motion_tolerance_px(config: TrackV2Config) -> float:
    explicit = _first_configured(config.motion_tolerance_px, config.prediction_gate_px, config.latest_position_gate_px)
    if explicit is not None:
        return float(explicit)
    return max(float(config.confirmed_prediction_gate_px), float(config.confirmed_latest_position_gate_px))


def _motion_tolerance_growth_px_per_sec(config: TrackV2Config) -> float:
    explicit = _first_configured(
        config.motion_tolerance_growth_px_per_sec,
        config.prediction_gate_growth_px_per_sec,
        config.latest_position_gate_growth_px_per_sec,
    )
    if explicit is not None:
        return float(explicit)
    return float(config.max_speed_px_per_sec)


def _max_physical_speed_px_per_sec(config: TrackV2Config) -> float:
    explicit = _first_configured(config.max_physical_speed_px_per_sec, config.hard_speed_limit_px_per_sec)
    if explicit is not None:
        return float(explicit)
    return float(config.confirmed_max_speed_px_per_sec)


def _localization_jitter_px(config: TrackV2Config) -> float:
    explicit = _first_configured(config.localization_jitter_px, config.jitter_tolerance_px)
    if explicit is not None:
        return float(explicit)
    return float(config.base_motion_gate_px)


def _normalize_config(config: TrackV2Config) -> _NormalizedTrackConfig:
    """Translate public/legacy config into the private behavioural surface."""

    return _NormalizedTrackConfig(
        confirmation_hits=_confirmation_hits(config),
        detector_miss_tolerance_sec=_detector_miss_tolerance_sec(config),
        tentative_tolerance_sec=_tentative_tolerance_sec(config),
        motion_tolerance_px=_motion_tolerance_px(config),
        motion_tolerance_growth_px_per_sec=_motion_tolerance_growth_px_per_sec(config),
        max_physical_speed_px_per_sec=_max_physical_speed_px_per_sec(config),
        localization_jitter_px=_localization_jitter_px(config),
        continuity_strength=float(config.continuity_strength),
        takeover_margin=float(config.takeover_margin),
        allow_weak_confirmed_matching=bool(config.allow_weak_confirmed_matching),
        appearance_tiebreak_enabled=bool(config.appearance_tiebreak_enabled),
        epsilon=float(config.epsilon),
    )
