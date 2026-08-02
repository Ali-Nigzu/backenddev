import numpy as np
import pytest

from demographics.demographic import Demographic
from demographics.exceptions import DemographicInputError
from test_tracking_v2_pipeline import (
    demographics_for_events,
    make_frame_record,
    validate_detection_batch_matches_frame_batch,
    validate_event_best_crop_frames,
)


def test_make_frame_record_converts_bgr_to_rgb_once_and_preserves_ids():
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)

    frame = make_frame_record(7, 14.0, bgr)

    assert frame["frame_id"] == "frame-7"
    assert frame["timestamp"] == 0.5
    assert frame["image"].tolist() == [[[30, 20, 10]]]
    assert frame["image"].flags.c_contiguous


def test_detection_batch_alignment_validation():
    frame_batch = {"frames": [{"frame_id": "frame-0", "timestamp": 0.0, "image": np.zeros((1, 1, 3), dtype=np.uint8)}]}
    validate_detection_batch_matches_frame_batch(
        {"detections": [{"frame_id": "frame-0", "timestamp": 0.0, "detections": []}]},
        frame_batch,
    )
    with pytest.raises(ValueError, match="DetectionBatch.detections length does not match FrameBatch.frames"):
        validate_detection_batch_matches_frame_batch({"detections": []}, frame_batch)


def test_event_best_crop_frame_validation():
    frame_batch = {"frames": [{"frame_id": "frame-0", "timestamp": 0.0, "image": np.zeros((2, 2, 3), dtype=np.uint8)}]}
    event_batch = {"events": [{"track_id": "1", "timestamp": 0.0, "event_type": 1, "best_crop": {"frame_id": "frame-0", "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}}]}
    validate_event_best_crop_frames(event_batch, frame_batch)
    event_batch["events"][0]["best_crop"]["frame_id"] = "missing"
    with pytest.raises(ValueError, match="track_id=1 references missing frame_id missing"):
        validate_event_best_crop_frames(event_batch, frame_batch)


def test_demographic_uses_supplied_frame_batch_without_video_reopen(monkeypatch):
    class FakeModel:
        def predict(self, descriptors, frames_by_id):
            assert list(frames_by_id) == ["frame-0"]
            return [{"age": 33, "sex": 1} for _ in descriptors]

    demographic = Demographic.__new__(Demographic)
    demographic._model = FakeModel()
    monkeypatch.setattr("test_tracking_v2_pipeline.Demographic", lambda: demographic, raising=False)

    event_batch = {"events": [{"track_id": "1", "timestamp": 0.0, "event_type": 1, "best_crop": {"frame_id": "frame-0", "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}}]}
    frame_batch = {"frames": [{"frame_id": "frame-0", "timestamp": 0.0, "image": np.zeros((2, 2, 3), dtype=np.uint8)}]}

    # Call Demographic directly to validate output schema without constructing MiVOLO.
    assert demographic(event_batch, frame_batch) == {"results": [{"track_id": "1", "age": 33, "sex": 1}]}


def test_demographic_missing_frame_error_contains_track_frame_and_bbox():
    demographic = Demographic.__new__(Demographic)
    demographic._model = object()
    event_batch = {"events": [{"track_id": "1", "timestamp": 0.0, "event_type": 1, "best_crop": {"frame_id": "missing", "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}}]}
    with pytest.raises(DemographicInputError) as exc_info:
        demographic(event_batch, {"frames": []})
    message = str(exc_info.value)
    assert "track_id=1" in message
    assert "frame_id=missing" in message
    assert "bbox={'x1': 0.0, 'y1': 0.0, 'x2': 1.0, 'y2': 1.0}" in message
