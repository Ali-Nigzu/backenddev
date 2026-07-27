"""Small deterministic Track V2 configuration."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TrackV2Config:
    """Configuration for historical-location-only tracking."""

    location_history_window_frames: int = 5
    max_anchor_distance_px: float = 100.0
    anchor_tie_distance_px: float = 5.0
    confirmation_hits: int = 2
    tentative_timeout_seconds: float = 1.0
    active_timeout_seconds: float = 2.0
    max_history_points: int | None = 30

    def __post_init__(self) -> None:
        if not isinstance(self.location_history_window_frames, int):
            raise ValueError("location_history_window_frames must be an integer")
        if self.location_history_window_frames <= 0:
            raise ValueError("location_history_window_frames must be positive")
        if not isfinite(float(self.max_anchor_distance_px)) or float(self.max_anchor_distance_px) < 0:
            raise ValueError("max_anchor_distance_px must be finite and non-negative")
        if not isfinite(float(self.anchor_tie_distance_px)) or float(self.anchor_tie_distance_px) < 0:
            raise ValueError("anchor_tie_distance_px must be finite and non-negative")
        if not isinstance(self.confirmation_hits, int):
            raise ValueError("confirmation_hits must be an integer")
        if self.confirmation_hits <= 0:
            raise ValueError("confirmation_hits must be positive")
        if not isfinite(float(self.tentative_timeout_seconds)) or float(self.tentative_timeout_seconds) < 0:
            raise ValueError("tentative_timeout_seconds must be finite and non-negative")
        if not isfinite(float(self.active_timeout_seconds)) or float(self.active_timeout_seconds) < 0:
            raise ValueError("active_timeout_seconds must be finite and non-negative")
        if self.max_history_points is not None:
            if not isinstance(self.max_history_points, int):
                raise ValueError("max_history_points must be an integer or None")
            if self.max_history_points <= 0:
                raise ValueError("max_history_points must be positive when configured")
