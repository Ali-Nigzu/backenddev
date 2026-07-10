"""Track V2 mutation helpers.

These helpers only create tracks, append matched observations, and update best
crop data. Track V2 does not own lifecycle, pruning, stale, or expiry policy.
"""


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
