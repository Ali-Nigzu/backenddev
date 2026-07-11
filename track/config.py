"""Deterministic Track V2 matching parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackV2Config:
    """Centralized constants that only affect assignment/matching behaviour."""

    # Minimum path evidence required before a recent track receives confirmed
    # active priority over tentative tracks.
    active_confirmation_min_path_points: int = 2
    # Minimum path evidence a tentative track must accumulate to mature into the
    # confirmed active pool. Kept separate so confirmation policy stays tunable.
    tentative_confirmation_min_path_points: int = 2
    # Recency windows are measured against ObservationBatch.timestamp because
    # TrackState stores timestamps, not explicit frame counters.
    active_recency_window_frames: float | None = None
    tentative_recency_window_frames: float = 3.0
    # Tracks whose latest path timestamp is within this many timestamp units of
    # the current batch are active. Retained for compatibility with callers
    # using the previous continuity-window knob.
    active_track_window_frames: int = 3
    max_speed_px_per_sec: float = 500.0
    base_motion_gate_px: float = 45.0
    max_reassociation_gap_sec: float = 2.0
    min_appearance_similarity: float = -1.0
    motion_weight: float = 0.65
    appearance_weight: float = 0.35
    max_combined_cost: float = 1.25
    epsilon: float = 1e-9
