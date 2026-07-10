"""Deterministic Track V2 matching parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackV2Config:
    """Centralized constants that only affect assignment/matching behaviour."""

    max_speed_px_per_sec: float = 500.0
    base_motion_gate_px: float = 45.0
    max_reassociation_gap_sec: float = 2.0
    min_appearance_similarity: float = -1.0
    motion_weight: float = 0.65
    appearance_weight: float = 0.35
    max_combined_cost: float = 1.25
    epsilon: float = 1e-9
