"""Minimal black-box TestV2 harness for the DetectV2 contract.

Run directly:
    python testv2/test_detect_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection import DetectV2

SAMPLE_IMAGE = ROOT / "data" / "samples" / "1000040807.jpg"


def load_image() -> Any:
    """Decode the repository JPEG sample into an RGB uint8 contiguous image."""

    with Image.open(SAMPLE_IMAGE) as image_file:
        image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(image)


def make_frame(image: Any) -> dict[str, object]:
    return {
        "frame_id": "frame-v2-001",
        "timestamp": 1234.5,
        "image": image,
    }


def assert_detection_schema(detection: object) -> None:
    assert isinstance(detection, dict)
    assert set(detection.keys()) == {"detection_id", "bbox", "confidence"}
    assert isinstance(detection["detection_id"], str)
    assert isinstance(detection["confidence"], float)

    bbox = detection["bbox"]
    assert isinstance(bbox, dict)
    assert set(bbox.keys()) == {"x1", "y1", "x2", "y2"}
    for coordinate in bbox.values():
        assert isinstance(coordinate, float)


def assert_detection_batch_contract(
    output: dict[str, object], frame: dict[str, object]
) -> None:
    assert set(output.keys()) == {"frame_id", "timestamp", "detections"}
    assert output["frame_id"] == frame["frame_id"]
    assert output["timestamp"] == frame["timestamp"]
    assert isinstance(output["detections"], list)

    for detection in output["detections"]:
        assert_detection_schema(detection)


def run_detect_v2_contract() -> dict[str, object]:
    image = load_image()
    frame = make_frame(image)

    detector = DetectV2()
    output = detector(frame)

    print("DetectV2 DetectionBatch:")
    print(json.dumps(output, indent=2, sort_keys=True))

    assert_detection_batch_contract(output, frame)

    return {
        "detections": len(output["detections"]),
        "schema": "valid",
    }


def test_detect_v2_contract() -> None:
    run_detect_v2_contract()


def main() -> None:
    diagnostics = {"contract": run_detect_v2_contract()}
    print("DetectV2 TestV2 diagnostics:")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
