"""Deterministic Track V2 matching parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackV2Config:
    """Centralized constants that affect Track V2 eligibility and assignment.

    The public reducer contract is intentionally unchanged. Internally these
    fields are collapsed by ``track.policy.build_policy`` into one small behaviour
    policy. The authoritative tuning concepts are: confirmation hits, tentative
    age, confirmed miss tolerance, position uncertainty, miss uncertainty growth,
    localization jitter, historical location anchoring, max speed, weak
    confirmed motion score, continuity bias, takeover margin, appearance
    tie-breaking, and birth suppression strength.
    Remaining historical names are accepted only as compatibility aliases.
    """

    # Lifecycle / maturity.
    confirmation_min_path_points: int = 2
    confirmation_hits: int | None = None
    detector_miss_tolerance_sec: float | None = None
    tentative_track_window_sec: float = 1.0
    tentative_tolerance_sec: float | None = None
    confirmed_track_window_sec: float | None = None
    confirmed_reassociation_window_sec: float = 2.0
    allow_weak_confirmed_matching: bool = True

    # Lifecycle-specific motion limits. Tentative tracks deliberately use
    # stricter defaults while still competing observation-by-observation when
    # they are believable explanations.
    confirmed_prediction_gate_px: float = 100.0
    tentative_prediction_gate_px: float = 40.0
    confirmed_latest_position_gate_px: float = 150.0
    tentative_latest_position_gate_px: float = 50.0
    confirmed_max_speed_px_per_sec: float = 500.0
    tentative_max_speed_px_per_sec: float = 250.0

    # Backwards-compatible lifecycle names.
    active_confirmation_min_path_points: int = 2
    tentative_confirmation_min_path_points: int = 2
    tentative_recency_window_frames: float = 3.0
    max_reassociation_gap_sec: float = 2.0

    # Motion / spatial plausibility.
    motion_tolerance_px: float | None = None
    motion_tolerance_growth_px_per_sec: float | None = None
    max_believable_speed_px_per_sec: float | None = None
    max_physical_speed_px_per_sec: float | None = None
    hard_speed_limit_px_per_sec: float | None = None
    prediction_gate_px: float | None = None
    prediction_gate_growth_px_per_sec: float | None = None
    latest_position_gate_px: float | None = None
    latest_position_gate_growth_px_per_sec: float | None = None
    localization_jitter_px: float | None = None
    jitter_tolerance_px: float | None = None
    location_history_window_frames: int = 5

    # Backwards-compatible motion names.
    max_speed_px_per_sec: float = 500.0
    base_motion_gate_px: float = 45.0

    # Candidate classification. A lower score is stronger; weak candidates are
    # still realistic continuations and are kept for continuity preservation.
    strong_motion_threshold: float = 0.35
    normal_motion_threshold: float = 1.0
    weak_motion_threshold: float = 1.75

    # Appearance remains secondary and never controls eligibility.
    appearance_tiebreak_enabled: bool = True

    # Deprecated compatibility valve. The eligibility model excludes impossible
    # candidates before assignment, so this should not be needed for normal use.
    forced_continuity_break_normalized_motion: float | None = 8.0

    # Behavioural continuity controls. Continuity strength is a same-scale
    # additive allowance on normalized motion for incumbent preservation;
    # takeover_margin is a relative margin challengers must beat before an
    # incumbent's deterministic first claim is considered clearly worse.
    continuity_strength: float = 0.005
    takeover_margin: float = 0.50

    # Birth suppression. At 1.0, births are overflow-only: if lifecycle-active
    # tracks are greater than or equal to observations, every observation must
    # be explained by an existing active identity.
    birth_suppression_strength: float = 1.0

    epsilon: float = 1e-9
