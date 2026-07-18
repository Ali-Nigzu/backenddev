"""Compatibility shim for private Track V2 policy construction."""

from track.policy import TrackerPolicy as _NormalizedTrackConfig
from track.policy import build_policy as _normalize_config

__all__ = ["_NormalizedTrackConfig", "_normalize_config"]
