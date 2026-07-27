"""Small deterministic Track V2 configuration."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TrackV2Config:
    """Behaviour settings for historical-location-only, frame-based tracking."""

    location_history_window_frames: int = 7
    max_anchor_distance_px: float = 100.0
    anchor_tie_distance_px: float = 20.0
    confirmation_hits: int = 3
    active_timeout_frames: int = 30
    tentative_timeout_frames: int = 15

    def __post_init__(self) -> None:
        _require_positive_int(self.location_history_window_frames, "location_history_window_frames")
        _require_positive_float(self.max_anchor_distance_px, "max_anchor_distance_px")
        _require_non_negative_float(self.anchor_tie_distance_px, "anchor_tie_distance_px")
        _require_positive_int(self.confirmation_hits, "confirmation_hits")
        _require_positive_int(self.active_timeout_frames, "active_timeout_frames")
        _require_positive_int(self.tentative_timeout_frames, "tentative_timeout_frames")


def _require_positive_int(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_float(value, name: str) -> None:
    if not isinstance(value, (float, int)) or not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_float(value, name: str) -> None:
    if not isinstance(value, (float, int)) or not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if float(value) < 0:
        raise ValueError(f"{name} must be non-negative")
