"""Deterministic Track V2 matching parameters."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrackV2Config:
    """Centralized constants that affect Track V2 eligibility and assignment.

    The public reducer contract is intentionally unchanged. These fields tune the
    internal observation-specific eligibility model while preserving older field
    names as compatibility aliases where possible.
    """

    # Lifecycle / maturity.
    confirmation_min_path_points: int = 2
    tentative_track_window_sec: float = 1.0
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
    active_recency_window_frames: float | None = None
    tentative_recency_window_frames: float = 3.0
    active_track_window_frames: int = 3
    max_reassociation_gap_sec: float = 2.0

    # Motion / spatial plausibility.
    max_believable_speed_px_per_sec: float | None = None
    hard_speed_limit_px_per_sec: float | None = None
    prediction_gate_px: float | None = None
    prediction_gate_growth_px_per_sec: float | None = None
    latest_position_gate_px: float | None = None
    latest_position_gate_growth_px_per_sec: float | None = None
    jitter_tolerance_px: float | None = None

    # Backwards-compatible motion names.
    max_speed_px_per_sec: float = 500.0
    base_motion_gate_px: float = 45.0

    # Candidate classification. A lower score is stronger; weak candidates are
    # still realistic continuations and are kept for continuity preservation.
    strong_motion_threshold: float = 0.35
    normal_motion_threshold: float = 1.0
    weak_motion_threshold: float = 1.75

    # Appearance remains secondary. Older combined-cost fields are retained for
    # compatibility but assignment remains lexicographic and motion-first.
    min_appearance_similarity: float = -1.0
    motion_weight: float = 0.65
    appearance_weight: float = 0.35
    max_combined_cost: float = 1.25

    # Deprecated compatibility valve. The eligibility model excludes impossible
    # candidates before assignment, so this should not be needed for normal use.
    forced_continuity_break_normalized_motion: float | None = 8.0

    # Optional diagnostics. If debug_diagnostics is a list, per-observation
    # diagnostics are appended. If debug_callback is callable, it receives the
    # full diagnostics list after assignment.
    debug_diagnostics: Any = None
    debug_callback: Any = None

    epsilon: float = 1e-9
