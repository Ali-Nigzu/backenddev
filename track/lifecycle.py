"""Frame-based Track V2 lifecycle classification and state mutation helpers."""

from dataclasses import dataclass

from track.config import TrackV2Config

ACTIVE = "active"
TENTATIVE = "tentative"
INACTIVE = "inactive"


@dataclass(frozen=True)
class TrackStatus:
    """Derived lifecycle state for one track at one frame."""

    state: str
    active: bool
    tentative: bool
    eligible: bool
    age_frames: int
    history_length: int


def classify_track(track: dict, current_frame_number: float, config: TrackV2Config) -> TrackStatus:
    last_seen_frame = float(track["path"][-1]["timestamp"])
    frame_delta = float(current_frame_number) - last_seen_frame
    if frame_delta < 0:
        raise ValueError("Track path contains a frame newer than the observation batch")
    age_frames = int(frame_delta)

    history_length = len(track["path"])
    if history_length >= int(config.confirmation_hits):
        active = age_frames <= int(config.active_timeout_frames)
        state = ACTIVE if active else INACTIVE
        return TrackStatus(state, active, False, active, age_frames, history_length)

    tentative = age_frames <= int(config.tentative_timeout_frames)
    state = TENTATIVE if tentative else INACTIVE
    return TrackStatus(state, False, tentative, tentative, age_frames, history_length)


def update_best_crop(track, observation, frame_id: str) -> None:
    confidence = float(observation["confidence"])
    if confidence > float(track["best_crop_confidence"]):
        track["best_crop"] = {
            "frame_id": frame_id,
            "bbox": dict(observation["bbox"]),
            "embedding": observation["embedding"],
        }
        track["best_crop_confidence"] = confidence


def append_observation(track, observation, frame_id: str, frame_number: float) -> None:
    track["path"].append(
        {
            "timestamp": float(frame_number),
            "center": {
                "x": float(observation["center"]["x"]),
                "y": float(observation["center"]["y"]),
            },
        }
    )
    update_best_crop(track, observation, frame_id)


def create_track(observation, frame_id: str, frame_number: float, track_id: str) -> dict:
    return {
        "track_id": str(track_id),
        "path": [
            {
                "timestamp": float(frame_number),
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
