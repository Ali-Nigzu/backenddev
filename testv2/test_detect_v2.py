"""TestV2 harness for the DetectV2 contract.

Run directly:
    python testv2/test_detect_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection import DetectV2, DetectV2Config, DetectV2InputError
from detection.detection_engine import RawDetection

SAMPLE_IMAGE = ROOT / "data" / "samples" / "1000040807.jpg"


class StaticDetectBackend:
    """Deterministic backend fixture for contract validation."""

    def __init__(self, detections: Iterable[RawDetection]) -> None:
        self._detections = tuple(detections)

    def infer(self, image: np.ndarray) -> Iterable[RawDetection]:
        _ = image
        return self._detections


def load_rgb_sample() -> np.ndarray:
    """Load repository fixture bytes into a valid locked RGB uint8 image.

    The TestV2 harness validates the DetectV2 contract rather than image codec
    behaviour. Keeping this loader dependency-free avoids making the contract
    test depend on OpenCV/Pillow availability in minimal CI environments.
    """

    data = SAMPLE_IMAGE.read_bytes()
    required = 64 * 64 * 3
    if len(data) < required:
        repeats = (required + len(data) - 1) // len(data)
        data = (data * repeats)[:required]
    else:
        data = data[:required]
    image = np.frombuffer(data, dtype=np.uint8).reshape((64, 64, 3))
    return np.ascontiguousarray(image)


def make_frame(image: np.ndarray, frame_id: str = "frame-v2-001") -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "timestamp": 1234.5,
        "image": image,
    }


def make_detector(detections: Iterable[RawDetection]) -> DetectV2:
    return DetectV2(
        config=DetectV2Config(max_detections=300),
        backend=StaticDetectBackend(detections),
    )


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


def run_detect_v2_contract_and_determinism() -> dict[str, object]:
    image = load_rgb_sample()
    frame = make_frame(image)
    backend_detections = [
        RawDetection(30.0, 20.0, 110.0, 170.0, 0.81, 2),
        RawDetection(5.0, 10.0, 60.0, 100.0, 0.95, 1),
        RawDetection(5.0, 10.0, 50.0, 90.0, 0.95, 0),
        RawDetection(-10.0, -20.0, 12.0, 40.0, 0.60, 3),
    ]
    detector = make_detector(backend_detections)

    first = detector.detect(frame)
    second = detector.detect(make_frame(image))

    assert first == second
    assert_detection_batch_contract(first, frame)
    assert [d["confidence"] for d in first["detections"]] == sorted(
        [d["confidence"] for d in first["detections"]], reverse=True
    )

    return {
        "detections": len(first["detections"]),
        "first_detection_id": first["detections"][0]["detection_id"],
        "deterministic": first == second,
        "schema": "valid",
    }


def run_detect_v2_empty_detections() -> dict[str, object]:
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    frame = make_frame(image, frame_id="empty-frame")
    detector = make_detector(())

    batch = detector.detect(frame)

    assert_detection_batch_contract(batch, frame)
    assert batch["detections"] == []
    return {"detections": 0, "empty_batch": batch["detections"] == []}


def run_detect_v2_corrupt_input() -> dict[str, object]:
    corrupt_frame = make_frame(np.zeros((8, 8, 3), dtype=np.float32), "corrupt-frame")
    detector = make_detector(())

    try:
        detector.detect(corrupt_frame)
    except DetectV2InputError as exc:
        return {"raised": "DetectV2InputError", "message": str(exc)}

    raise AssertionError("DetectV2InputError was not raised for corrupt input")


def test_detect_v2_contract_and_determinism() -> None:
    run_detect_v2_contract_and_determinism()


def test_detect_v2_empty_detections() -> None:
    run_detect_v2_empty_detections()


def test_detect_v2_corrupt_input() -> None:
    run_detect_v2_corrupt_input()


def main() -> None:
    diagnostics = {
        "contract_and_determinism": run_detect_v2_contract_and_determinism(),
        "empty_detections": run_detect_v2_empty_detections(),
        "corrupt_input": run_detect_v2_corrupt_input(),
    }
    print("DetectV2 TestV2 diagnostics:")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
