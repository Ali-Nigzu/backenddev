"""Straight-line black-box V2 baseline pipeline validation."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection import Detect
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

detect = Detect()
detection_batch = detect(frame)

print(detection_batch)

assert set(detection_batch.keys()) == {"frame_id", "timestamp", "detections"}
assert detection_batch["frame_id"] == frame["frame_id"]
assert detection_batch["timestamp"] == frame["timestamp"]
assert isinstance(detection_batch["detections"], list)

for detection in detection_batch["detections"]:
    assert set(detection.keys()) == {"detection_id", "bbox", "confidence"}
    assert isinstance(detection["detection_id"], str)
    assert isinstance(detection["confidence"], float)
    assert set(detection["bbox"].keys()) == {"x1", "y1", "x2", "y2"}
    for coordinate in detection["bbox"].values():
        assert isinstance(coordinate, float)

embed = Embed()
embedding_batch = embed(detection_batch)

print(embedding_batch)

assert set(embedding_batch.keys()) == {"frame_id", "timestamp", "embeddings"}
assert embedding_batch["frame_id"] == detection_batch["frame_id"]
assert embedding_batch["timestamp"] == detection_batch["timestamp"]
assert isinstance(embedding_batch["embeddings"], list)
assert len(embedding_batch["embeddings"]) == len(detection_batch["detections"])

for detection, embedding in zip(detection_batch["detections"], embedding_batch["embeddings"]):
    assert set(embedding.keys()) == {"detection_id", "vector"}
    assert embedding["detection_id"] == detection["detection_id"]
    vector = embedding["vector"]
    assert set(vector.keys()) == {"dtype", "shape", "values"}
    assert vector["dtype"] == "float32"
    assert isinstance(vector["shape"], list)
    assert len(vector["shape"]) == 1
    assert isinstance(vector["values"], list)
    assert len(vector["values"]) == vector["shape"][0]
