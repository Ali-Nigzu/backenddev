"""Lifecycle helpers derived only from path timestamps."""

from track.config import TrackV2Config


def update_best_crop(track, observation, frame_id: str) -> None:
    confidence = float(observation["confidence"])
    if confidence > float(track["best_crop_confidence"]):
        track["best_crop"] = {
            "frame_id": frame_id,
            "bbox": dict(observation["bbox"]),
            "embedding": observation["embedding"],
        }
        track["best_crop_confidence"] = confidence


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
    if config.max_path_length > 0 and len(track["path"]) > config.max_path_length:
        del track["path"][: len(track["path"]) - config.max_path_length]
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


def is_expired(track, current_timestamp: float, config: TrackV2Config) -> bool:
    latest_timestamp = float(track["path"][-1]["timestamp"])
    return float(current_timestamp) - latest_timestamp > config.stale_timeout_sec
