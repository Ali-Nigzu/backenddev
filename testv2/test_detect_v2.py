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
from observe import Observe


def assert_has_fields(contract_name, payload, fields):
    assert payload is not None, f"{contract_name} must exist"
    for field in fields:
        assert field in payload, f"{contract_name}.{field} must exist"


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

assert_has_fields(
    "DetectionBatch",
    detection_batch,
    ("frame_id", "timestamp", "detections"),
)
assert set(detection_batch.keys()) == {"frame_id", "timestamp", "frame", "detections"}
assert detection_batch["frame_id"] == frame["frame_id"]
assert detection_batch["timestamp"] == frame["timestamp"]
assert detection_batch["frame"]["image"] is image
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

assert_has_fields(
    "EmbeddingBatch",
    embedding_batch,
    ("frame_id", "timestamp", "embeddings"),
)
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
    assert len(vector["values"]) > 0
    assert all(isinstance(value, float) for value in vector["values"])

observe = Observe()
observation_batch = observe(detection_batch, embedding_batch)

print(observation_batch)

assert_has_fields(
    "ObservationBatch",
    observation_batch,
    ("frame_id", "timestamp", "observations"),
)
assert set(observation_batch.keys()) == {"frame_id", "timestamp", "observations"}
assert observation_batch["frame_id"] == detection_batch["frame_id"]
assert observation_batch["timestamp"] == detection_batch["timestamp"]
assert observation_batch["timestamp"] == embedding_batch["timestamp"]
assert isinstance(observation_batch["observations"], list)
assert len(observation_batch["observations"]) == len(embedding_batch["embeddings"])
assert len(observation_batch["observations"]) == len(detection_batch["detections"])

detections_by_id = {
    detection["detection_id"]: detection
    for detection in detection_batch["detections"]
}
embeddings_by_id = {
    embedding["detection_id"]: embedding
    for embedding in embedding_batch["embeddings"]
}
observations_by_id = {}

for observation in observation_batch["observations"]:
    assert_has_fields(
        "Observation",
        observation,
        ("detection_id", "bbox", "center", "embedding", "confidence"),
    )
    assert set(observation.keys()) == {
        "detection_id",
        "bbox",
        "center",
        "embedding",
        "confidence",
    }
    detection_id = observation["detection_id"]
    assert detection_id not in observations_by_id
    observations_by_id[detection_id] = observation

    detection = detections_by_id[detection_id]
    embedding = embeddings_by_id[detection_id]
    bbox = detection["bbox"]

    assert observation["bbox"] == bbox
    assert observation["embedding"] == embedding["vector"]
    assert observation["confidence"] == detection["confidence"]
    assert set(observation["center"].keys()) == {"x", "y"}
    assert observation["center"]["x"] == (bbox["x1"] + bbox["x2"]) / 2
    assert observation["center"]["y"] == (bbox["y1"] + bbox["y2"]) / 2

for detection in detection_batch["detections"]:
    matches = [
        observation
        for observation in observation_batch["observations"]
        if observation["detection_id"] == detection["detection_id"]
    ]
    assert len(matches) == 1

for embedding in embedding_batch["embeddings"]:
    matches = [
        observation
        for observation in observation_batch["observations"]
        if observation["detection_id"] == embedding["detection_id"]
    ]
    assert len(matches) == 1


# Observe must reject DetectionBatch/EmbeddingBatch timestamp mismatches.
mismatched_embedding_batch = {
    "frame_id": embedding_batch["frame_id"],
    "timestamp": embedding_batch["timestamp"] + 1.0,
    "embeddings": embedding_batch["embeddings"],
}
try:
    observe(detection_batch, mismatched_embedding_batch)
except ValueError as exc:
    assert str(exc) == "DetectionBatch.timestamp must match EmbeddingBatch.timestamp"
else:
    raise AssertionError("Observe must reject mismatched batch timestamps")

# Observe must require EmbeddingBatch.timestamp.
missing_timestamp_embedding_batch = {
    "frame_id": embedding_batch["frame_id"],
    "embeddings": embedding_batch["embeddings"],
}
try:
    observe(detection_batch, missing_timestamp_embedding_batch)
except ValueError as exc:
    assert str(exc) == "Missing required EmbeddingBatch field: timestamp"
else:
    raise AssertionError("Observe must require EmbeddingBatch.timestamp")
