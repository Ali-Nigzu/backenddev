import numpy as np
import pytest

from contracts import FrameBatchError, build_frame_lookup, validate_frame_batch


def frame(frame_id="frame-0", timestamp=0.0, image=None):
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "image": np.zeros((2, 3, 3), dtype=np.uint8) if image is None else image,
    }


def test_valid_frame_batch_preserves_frame_references_and_order():
    first = frame("frame-1", 2.0)
    second = frame("frame-2", 1.0)
    batch = {"frames": [first, second]}

    assert validate_frame_batch(batch) == [first, second]
    assert build_frame_lookup(batch)["frame-1"] is first


def test_empty_frame_batch_is_valid():
    assert validate_frame_batch({"frames": []}) == []


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        (None, "FrameBatch must be a mapping"),
        ({}, "Missing required FrameBatch field: frames"),
        ({"frames": ()}, "FrameBatch.frames must be a list"),
        ({"frames": [], "metadata": {}}, "Unexpected FrameBatch field: metadata"),
        ({"frames": [None]}, "Frame at index 0 must be a mapping"),
        ({"frames": [{"frame_id": "frame-0", "timestamp": 0.0}]}, "Missing required FrameBatch.frames[0] field: image"),
        ({"frames": [frame("", 0.0)]}, "Frame at index 0 frame_id must be a non-empty string"),
        ({"frames": [frame("frame-0", True)]}, "Frame frame-0 timestamp must be finite"),
        ({"frames": [frame("frame-0", float("nan"))]}, "Frame frame-0 timestamp must be finite"),
        ({"frames": [frame("frame-0", 0.0, np.zeros((2, 3, 3), dtype=np.float32))]}, "Frame frame-0 image must have dtype uint8"),
        ({"frames": [frame("frame-0", 0.0, np.zeros((2, 3), dtype=np.uint8))]}, "Frame frame-0 image must have shape H x W x 3"),
        ({"frames": [frame("frame-0", 0.0, np.zeros((0, 3, 3), dtype=np.uint8))]}, "Frame frame-0 image must have positive dimensions"),
        ({"frames": [frame("frame-0", 0.0, np.zeros((2, 3, 3), dtype=np.uint8)[:, ::2, :])]}, "Frame frame-0 image must be C-contiguous"),
    ],
)
def test_invalid_frame_batch_errors_are_clear(batch, message):
    with pytest.raises(FrameBatchError, match=message.replace("[", r"\[").replace("]", r"\]")):
        validate_frame_batch(batch)


def test_duplicate_frame_id_fails_but_duplicate_timestamp_is_allowed():
    validate_frame_batch({"frames": [frame("frame-0", 1.0), frame("frame-1", 1.0)]})
    with pytest.raises(FrameBatchError, match="Duplicate frame_id: frame-0"):
        validate_frame_batch({"frames": [frame("frame-0", 0.0), frame("frame-0", 1.0)]})


def test_required_lookup_fails_for_missing_frame_id():
    with pytest.raises(FrameBatchError, match="Missing frame_id in FrameBatch: frame-2"):
        build_frame_lookup({"frames": [frame("frame-1")]}, required_ids={"frame-2"})
