"""Straight-line black-box Detect contract validation."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection import Detect

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
result = detect(frame)

print(result)

assert set(result.keys()) == {"frame_id", "timestamp", "detections"}
assert result["frame_id"] == frame["frame_id"]
assert result["timestamp"] == frame["timestamp"]
assert isinstance(result["detections"], list)

for detection in result["detections"]:
    assert set(detection.keys()) == {"detection_id", "bbox", "confidence"}
    assert isinstance(detection["detection_id"], str)
    assert isinstance(detection["confidence"], float)
    assert set(detection["bbox"].keys()) == {"x1", "y1", "x2", "y2"}
    for coordinate in detection["bbox"].values():
        assert isinstance(coordinate, float)
