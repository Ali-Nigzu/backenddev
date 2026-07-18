"""Compatibility exports for lifecycle facts."""

from track.lifecycle import (
    CONFIRMED_LIVE as PROTECTED,
    CONFIRMED_MISSING as REASSOCIATION,
    STALE,
    TENTATIVE,
    TrackStatus as TrackFacts,
    classify_track as derive_track_facts,
)

__all__ = [
    "PROTECTED",
    "REASSOCIATION",
    "STALE",
    "TENTATIVE",
    "TrackFacts",
    "derive_track_facts",
]
