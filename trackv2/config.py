from dataclasses import dataclass


@dataclass(frozen=True)
class TrackV2Config:
    max_speed_px_per_sec: float = 500.0
    base_motion_gate: float = 40.0
    gate_multiplier: float = 1.0
    tentative_hits_to_activate: int = 2
    max_misses_active: int = 5
    max_misses_tentative: int = 1
    embedding_tie_threshold: float = 0.05
    motion_ambiguity_delta: float = 8.0
    velocity_smoothing: float = 0.6
    closed_track_cooldown_sec: float = 3.0
    unmatched_detection_buffer_frames: int = 3
    min_track_lifetime_sec: float = 0.7
    max_association_gap_frames: int = 2
    strict_motion_gate_multiplier: float = 1.2
