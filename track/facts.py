"""Private lifecycle and motion facts derived from explicit Track state."""

from dataclasses import dataclass

from track.motion import derive_velocity, predict_center
from track.normalize import _NormalizedTrackConfig

PROTECTED = "protected"
REASSOCIATION = "reassociation"
TENTATIVE = "tentative"
STALE = "stale"


@dataclass(frozen=True)
class TrackFacts:
    confirmed: bool
    age_seconds: float
    missing_seconds: float
    eligible: bool
    ownership_class: str
    latest_position: dict
    predicted_position: dict
    velocity: dict


def derive_track_facts(track: dict, timestamp: float, config: _NormalizedTrackConfig) -> TrackFacts:
    latest = track["path"][-1]
    latest_timestamp = float(latest["timestamp"])
    age_seconds = float(timestamp) - latest_timestamp
    confirmed = len(track["path"]) >= int(config.confirmation_hits)

    if age_seconds < -config.epsilon:
        eligible = False
        ownership_class = STALE
    elif confirmed:
        eligible = age_seconds <= float(config.detector_miss_tolerance_sec)
        ownership_class = PROTECTED if eligible and age_seconds <= config.epsilon else REASSOCIATION
    else:
        eligible = age_seconds <= float(config.tentative_tolerance_sec)
        ownership_class = TENTATIVE if eligible else STALE

    velocity = derive_velocity(track["path"], config)
    predicted = predict_center(track, timestamp, config)
    return TrackFacts(
        confirmed=confirmed,
        age_seconds=age_seconds,
        missing_seconds=max(0.0, age_seconds),
        eligible=eligible,
        ownership_class=ownership_class,
        latest_position=latest["center"],
        predicted_position=predicted,
        velocity=velocity,
    )
