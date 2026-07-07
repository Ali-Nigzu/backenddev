"""Straight-line black-box Embed contract validation."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embed import Embed

with Image.open(ROOT / "data" / "samples" / "1000040807.jpg") as image_file:
    image = np.ascontiguousarray(
        np.asarray(image_file.convert("RGB"), dtype=np.uint8)
    )

frame = {
    "frame_id": "frame-v2-001",
    "timestamp": 1234.5,
    "image": image,
}

detection_batch = {
    "frame_id": frame["frame_id"],
    "timestamp": frame["timestamp"],
    "frame": frame,
    "detections": [
        {
            "detection_id": "frame-v2-001:det:0",
            "bbox": {
                "x1": 0.0,
                "y1": 0.0,
                "x2": float(min(128, image.shape[1])),
                "y2": float(min(256, image.shape[0])),
            },
            "confidence": 1.0,
        }
    ],
}

embed = Embed()
result = embed(detection_batch)

print(result)

assert set(result.keys()) == {"frame_id", "timestamp", "embeddings"}
assert result["frame_id"] == detection_batch["frame_id"]
assert result["timestamp"] == detection_batch["timestamp"]
assert isinstance(result["embeddings"], list)
assert len(result["embeddings"]) == len(detection_batch["detections"])

for detection, embedding in zip(detection_batch["detections"], result["embeddings"]):
    assert set(embedding.keys()) == {"detection_id", "vector"}
    assert embedding["detection_id"] == detection["detection_id"]
    vector = embedding["vector"]
    assert set(vector.keys()) == {"dtype", "shape", "values"}
    assert vector["dtype"] == "float32"
    assert isinstance(vector["shape"], list)
    assert len(vector["shape"]) == 1
    assert isinstance(vector["values"], list)
    assert len(vector["values"]) == vector["shape"][0]
    assert all(isinstance(value, float) for value in vector["values"])
