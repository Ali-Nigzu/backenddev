import math
from pathlib import Path

import numpy as np
import pytest

from demographics import Demographic, DemographicInputError, DemographicModelError
from demographics.model import EXPECTED_SHA256, MiVOLOBackend, convert_outputs
from demographics.preprocessing import body_crop_to_mivolo_input, crop_body


BBOX = {"x1": 1.2, "y1": 2.2, "x2": 8.7, "y2": 9.1}


class FakeBackend:
    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = outputs

    def predict(self, batch):
        self.calls.append(batch)
        count = batch.shape[0]
        if self.outputs is not None:
            return self.outputs
        return [{"age": 30 + i, "sex": i % 2} for i in range(count)]


def event(track_id="t1", timestamp=1.0, event_type=1, frame_id="frame-0", bbox=None):
    return {
        "track_id": track_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "best_crop": {"frame_id": frame_id, "bbox": dict(bbox or BBOX)},
    }


def frame(frame_id="frame-0", image=None):
    if image is None:
        image = np.full((16, 16, 3), [10, 20, 30], dtype=np.uint8)
    return {"frame_id": frame_id, "timestamp": 0.0, "image": image}


def test_empty_event_batch_returns_without_backend_or_checkpoint(tmp_path):
    backend = FakeBackend()
    assert Demographic(checkpoint_path=tmp_path / "missing.pth", backend=backend)({"events": []}, {"frames": []}) == {"results": []}
    assert backend.calls == []


def test_one_track_produces_one_result_and_no_race():
    backend = FakeBackend([{"age": 34, "sex": 1}])
    result = Demographic(backend=backend)({"events": [event()]}, {"frames": [frame()]})
    assert result == {"results": [{"track_id": "t1", "age": 34, "sex": 1}]}
    assert "race" not in result["results"][0]
    assert backend.calls[0].shape == (1, 6, 224, 224)


def test_two_events_for_one_track_infer_once_for_one_crop():
    backend = FakeBackend([{"age": 40, "sex": 0}])
    batch = {"events": [event("t1", 2.0, 1), event("t1", 5.0, 0)]}
    result = Demographic(backend=backend)(batch, {"frames": [frame()]})
    assert result == {"results": [{"track_id": "t1", "age": 40, "sex": 0}]}
    assert len(backend.calls) == 1
    assert backend.calls[0].shape[0] == 1


def test_two_tracks_order_by_earliest_timestamp_then_track_id():
    backend = FakeBackend([{"age": 20, "sex": 1}, {"age": 21, "sex": 0}])
    events = [event("b", 3.0), event("a", 1.0), event("a", 2.0)]
    result = Demographic(backend=backend)({"events": events}, {"frames": [frame()]})
    assert [row["track_id"] for row in result["results"]] == ["a", "b"]
    assert backend.calls[0].shape[0] == 2


def test_conflicting_crops_for_same_track_fail():
    events = [event("t1", 1.0), event("t1", 2.0, bbox={"x1": 2, "y1": 2, "x2": 9, "y2": 9})]
    with pytest.raises(DemographicInputError, match="Conflicting best_crop"):
        Demographic(backend=FakeBackend())({"events": events}, {"frames": [frame()]})


@pytest.mark.parametrize("bad_event", [{}, {"events": "bad"}, {"events": [dict(event(), event_type=2)]}])
def test_invalid_event_batch_fails(bad_event):
    with pytest.raises(DemographicInputError):
        Demographic(backend=FakeBackend())(bad_event, {"frames": [frame()]})


def test_duplicate_frame_ids_fail():
    with pytest.raises(DemographicInputError, match="Duplicate frame_id"):
        Demographic(backend=FakeBackend())({"events": [event()]}, {"frames": [frame(), frame()]})


def test_missing_frame_fails_with_context():
    with pytest.raises(DemographicInputError, match="track_id=t1.*frame_id=frame-0.*bbox"):
        Demographic(backend=FakeBackend())({"events": [event()]}, {"frames": []})


@pytest.mark.parametrize("image", ["bad", np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.float32)])
def test_invalid_image_fails(image):
    with pytest.raises(DemographicInputError):
        Demographic(backend=FakeBackend())({"events": [event()]}, {"frames": [frame(image=image)]})


def test_malformed_bbox_fails():
    with pytest.raises(DemographicInputError):
        Demographic(backend=FakeBackend())({"events": [event(bbox={"x1": 4, "y1": 1, "x2": 1, "y2": 2})]}, {"frames": [frame()]})


def test_out_of_bounds_bbox_clips_and_zero_area_fails():
    image = np.full((10, 10, 3), 255, dtype=np.uint8)
    crop = crop_body(image, {"x1": -2.1, "y1": -1, "x2": 5.2, "y2": 6}, "t", "f")
    assert crop.shape == (6, 6, 3)
    with pytest.raises(DemographicInputError):
        crop_body(image, {"x1": 20, "y1": 20, "x2": 25, "y2": 25}, "t", "f")


def test_rgb_preprocessing_shape_and_zero_face_channels():
    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    crop[..., 0] = 255
    tensor = body_crop_to_mivolo_input(crop)
    assert tensor.shape == (6, 224, 224)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert np.all(tensor[:3] == 0)
    assert tensor[3:].max() > tensor[4:].max()


def test_age_and_sex_conversion_rules():
    metadata = {"min_age": 1, "max_age": 95, "avg_age": 48.0}
    output = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, -10.0], [1.0, 1.0, 10.0], [1.0, 2.0, 0.00531915]], dtype=np.float32)
    result = convert_outputs(output, metadata)
    assert result[0] == {"age": 48, "sex": 1}
    assert result[1] == {"age": 1, "sex": 0}
    assert result[2] == {"age": 95, "sex": 1}
    assert result[3]["age"] == 49


@pytest.mark.parametrize("output", [np.array([[math.nan, 0, 0]], dtype=np.float32), np.array([[0, math.inf, 0]], dtype=np.float32), np.zeros((1, 2), dtype=np.float32)])
def test_invalid_model_outputs_fail(output):
    with pytest.raises(DemographicModelError):
        convert_outputs(output, {"min_age": 1, "max_age": 95, "avg_age": 48.0})


def test_missing_checkpoint_fails_for_non_empty_batch(tmp_path):
    with pytest.raises(DemographicModelError, match="assemble_weights"):
        Demographic(checkpoint_path=tmp_path / "missing.pth", device="cpu")({"events": [event()]}, {"frames": [frame()]})


def test_checksum_mismatch_fails(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    backend = MiVOLOBackend(checkpoint_path=bad, device="cpu")
    with pytest.raises(DemographicModelError, match="checksum"):
        backend.predict(np.zeros((1, 6, 224, 224), dtype=np.float32))


def test_explicit_unavailable_cuda_fails(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend = MiVOLOBackend(device="cuda")
    with pytest.raises(DemographicModelError, match="CUDA"):
        backend.predict(np.zeros((1, 6, 224, 224), dtype=np.float32))
