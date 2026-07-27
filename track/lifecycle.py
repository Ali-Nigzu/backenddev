"""Track V2 lifecycle classification and state mutation helpers."""

from dataclasses import dataclass

from track.config import TrackV2Config

ACTIVE = "active"
TENTATIVE = "tentative"
INACTIVE = "inactive"


@dataclass(frozen=True)
class TrackStatus:
    """Derived lifecycle state for one track at one timestamp."""

    state: str
    active: bool
    tentative: bool
    eligible: bool
    age_seconds: float
    history_length: int


def classify_track(track: dict, timestamp: float, config: TrackV2Config) -> TrackStatus:
    latest_timestamp = float(track["path"][-1]["timestamp"])
    age_seconds = float(timestamp) - latest_timestamp
    if age_seconds < 0:
        raise ValueError("Track path contains a timestamp newer than the observation batch")

    history_length = len(track["path"])
    if history_length >= int(config.confirmation_hits):
        active = age_seconds <= float(config.active_timeout_seconds)
        state = ACTIVE if active else INACTIVE
        return TrackStatus(state, active, False, active, age_seconds, history_length)

    tentative = age_seconds <= float(config.tentative_timeout_seconds)
    state = TENTATIVE if tentative else INACTIVE
    return TrackStatus(state, False, tentative, tentative, age_seconds, history_length)


def update_best_crop(track, observation, frame_id: str) -> None:
    confidence = float(observation["confidence"])
    if confidence > float(track["best_crop_confidence"]):
        track["best_crop"] = {
            "frame_id": frame_id,
            "bbox": dict(observation["bbox"]),
            "embedding": observation["embedding"],
        }
        track["best_crop_confidence"] = confidence


def _trim_history(track, config: TrackV2Config) -> None:
    if config.max_history_points is None:
        return
    overflow = len(track["path"]) - int(config.max_history_points)
    if overflow > 0:
        del track["path"][:overflow]


def append_observation(track, observation, frame_id: str, timestamp: float, config: TrackV2Config) -> None:
    track["path"].append(
        {
            "timestamp": float(timestamp),
            "center": {
                "x": float(observation["center"]["x"]),
                "y": float(observation["center"]["y"]),
            },
        }
    )
    _trim_history(track, config)
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
