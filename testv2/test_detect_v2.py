"""Black-box TestV2 harness for the DetectV2 contract.

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


def load_rgb_sample() -> Any:
    """Decode the repository JPEG sample into a locked RGB uint8 image."""

    with Image.open(SAMPLE_IMAGE) as image_file:
        image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(image)


def make_frame(image: Any, frame_id: str = "frame-v2-001") -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "timestamp": 1234.5,
        "image": image,
    }


def assert_detection_batch_contract(
    batch: dict[str, object], frame: dict[str, object]
) -> None:
    assert set(batch.keys()) == {"frame_id", "timestamp", "detections"}
    assert batch["frame_id"] == frame["frame_id"]
    assert batch["timestamp"] == frame["timestamp"]
    assert isinstance(batch["detections"], list)

    image = frame["image"]
    assert isinstance(image, np.ndarray)
    height, width = image.shape[:2]

    previous_order_key: tuple[float, float, float, float, float] | None = None
    for index, detection in enumerate(batch["detections"]):
        assert isinstance(detection, dict)
        assert set(detection.keys()) == {"detection_id", "bbox", "confidence"}
        assert detection["detection_id"] == f"{frame['frame_id']}:det:{index}"

        confidence = detection["confidence"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

        bbox = detection["bbox"]
        assert isinstance(bbox, dict)
        assert set(bbox.keys()) == {"x1", "y1", "x2", "y2"}
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]
        for value in (x1, y1, x2, y2):
            assert isinstance(value, float)
            assert np.isfinite(value)
        assert 0.0 <= x1 < x2 <= float(width)
        assert 0.0 <= y1 < y2 <= float(height)

        order_key = (-confidence, x1, y1, x2, y2)
        if previous_order_key is not None:
            assert previous_order_key <= order_key
        previous_order_key = order_key


def run_detect_v2_contract() -> dict[str, object]:
    image = load_rgb_sample()
    frame = make_frame(image)
    detector = DetectV2()

    first = detector(frame)
    second = detector(make_frame(image))

    print("DetectV2 DetectionBatch:")
    print(json.dumps(first, indent=2, sort_keys=True))

    assert first == second
    assert_detection_batch_contract(first, frame)

    return {
        "detections": len(first["detections"]),
        "deterministic": first == second,
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
