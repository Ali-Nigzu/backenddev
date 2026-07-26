"""Track V2 lifecycle classification and state mutation helpers."""

from dataclasses import dataclass

from track.policy import TrackerPolicy

TENTATIVE = "tentative"
CONFIRMED_LIVE = "confirmed_live"
CONFIRMED_MISSING = "confirmed_missing"
STALE = "stale"


@dataclass(frozen=True)
class TrackStatus:
    """Lifecycle facts derived exactly once for a track at a timestamp."""

    confirmed: bool
    age_seconds: float
    missing_seconds: float
    eligible: bool
    state: str
    latest_position: dict


def classify_track(track: dict, timestamp: float, policy: TrackerPolicy) -> TrackStatus:
    latest = track["path"][-1]
    latest_timestamp = float(latest["timestamp"])
    age_seconds = float(timestamp) - latest_timestamp
    if age_seconds < -policy.epsilon:
        raise ValueError("Track path contains a timestamp newer than the observation batch")

    confirmed = len(track["path"]) >= int(policy.confirmation_hits)
    missing_seconds = max(0.0, age_seconds)

    if confirmed:
        eligible = missing_seconds <= float(policy.confirmed_max_missed_sec)
        if not eligible:
            state = STALE
        elif missing_seconds <= policy.epsilon:
            state = CONFIRMED_LIVE
        else:
            state = CONFIRMED_MISSING
    else:
        eligible = missing_seconds <= float(policy.tentative_max_age_sec)
        state = TENTATIVE if eligible else STALE

    return TrackStatus(
        confirmed=confirmed,
        age_seconds=age_seconds,
        missing_seconds=missing_seconds,
        eligible=eligible,
        state=state,
        latest_position=latest["center"],
    )


def update_best_crop(track, observation, frame_id: str) -> None:
    confidence = float(observation["confidence"])
    if confidence > float(track["best_crop_confidence"]):
        track["best_crop"] = {
            "frame_id": frame_id,
            "bbox": dict(observation["bbox"]),
            "embedding": observation["embedding"],
        }
        track["best_crop_confidence"] = confidence


def append_observation(track, observation, frame_id: str, timestamp: float) -> None:
    track["path"].append(
        {
            "timestamp": float(timestamp),
            "center": {
                "x": float(observation["center"]["x"]),
                "y": float(observation["center"]["y"]),
            },
        }
    )
    update_best_crop(track, observation, frame_id)


def create_track(observation, frame_id: str, timestamp: float, track_id: str) -> dict:
    return {
        "track_id": str(track_id),
        "path": [
            {
                "timestamp": float(timestamp),
                "center": {
                    "x": float(observation["center"]["x"]),
                    "y": float(observation["center"]["y"]),
                },
            }
        ],
        "best_crop": {
            "frame_id": frame_id,
            "bbox": dict(observation["bbox"]),
            "embedding": observation["embedding"],
        },
        "best_crop_confidence": float(observation["confidence"]),
    }
