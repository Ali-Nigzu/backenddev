"""Deterministic Observe contract validation."""

from copy import deepcopy

import pytest

from observe import Observe


def _detection(detection_id, x1=10, y1=20, x2=30, y2=60, confidence=0.9):
    return {
        "detection_id": detection_id,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": confidence,
    }


def _embedding(detection_id, values=None):
    if values is None:
        values = [0.1, 0.2]
    return {
        "detection_id": detection_id,
        "vector": {"dtype": "float32", "shape": [len(values)], "values": values},
    }


def _detection_batch(detections):
    return {
        "frame_id": "frame-001",
        "timestamp": 1234.5,
        "detections": detections,
    }


def _embedding_batch(embeddings):
    return {
        "frame_id": "frame-001",
        "embeddings": embeddings,
    }


def test_successful_observation_creation():
    detection_batch = _detection_batch([_detection("person_1")])
    embedding_batch = _embedding_batch([_embedding("person_1", [0.3, 0.4])])

    observation_batch = Observe()(detection_batch, embedding_batch)

    assert observation_batch["frame_id"] == "frame-001"
    assert observation_batch["timestamp"] == 1234.5
    assert len(observation_batch["observations"]) == 1
    observation = observation_batch["observations"][0]
    assert observation["detection_id"] == "person_1"
    assert observation["embedding"] == {
        "dtype": "float32",
        "shape": [2],
        "values": [0.3, 0.4],
    }
    assert observation["confidence"] == 0.9


def test_center_calculation():
    detection_batch = _detection_batch([_detection("person_1", 10, 20, 30, 60)])
    embedding_batch = _embedding_batch([_embedding("person_1")])

    observation_batch = Observe()(detection_batch, embedding_batch)

    assert observation_batch["observations"][0]["center"] == {"x": 20, "y": 40}


def test_embedding_order_independence():
    detection_batch = _detection_batch(
        [
            _detection("person_1"),
            _detection("person_2", 40, 50, 80, 90),
        ]
    )
    embedding_batch = _embedding_batch(
        [
            _embedding("person_2", [0.2, 0.8]),
            _embedding("person_1", [0.1, 0.9]),
        ]
    )

    observation_batch = Observe()(detection_batch, embedding_batch)

    observations = observation_batch["observations"]
    assert [observation["detection_id"] for observation in observations] == [
        "person_1",
        "person_2",
    ]
    assert observations[0]["embedding"]["values"] == [0.1, 0.9]
    assert observations[1]["embedding"]["values"] == [0.2, 0.8]


def test_missing_embedding_failure():
    detection_batch = _detection_batch([_detection("person_1")])
    embedding_batch = _embedding_batch([])

    with pytest.raises(ValueError):
        Observe()(detection_batch, embedding_batch)


def test_duplicate_detection_id_failure():
    detection_batch = _detection_batch([_detection("person_1"), _detection("person_1")])
    embedding_batch = _embedding_batch([_embedding("person_1")])

    with pytest.raises(ValueError):
        Observe()(detection_batch, embedding_batch)


def test_duplicate_embedding_id_failure():
    detection_batch = _detection_batch([_detection("person_1")])
    embedding_batch = _embedding_batch([_embedding("person_1"), _embedding("person_1")])

    with pytest.raises(ValueError):
        Observe()(detection_batch, embedding_batch)


def test_frame_mismatch_failure():
    detection_batch = _detection_batch([_detection("person_1")])
    embedding_batch = _embedding_batch([_embedding("person_1")])
    embedding_batch["frame_id"] = "frame-002"

    with pytest.raises(ValueError):
        Observe()(detection_batch, embedding_batch)


def test_empty_batch():
    observation_batch = Observe()(_detection_batch([]), _embedding_batch([]))

    assert observation_batch == {
        "frame_id": "frame-001",
        "timestamp": 1234.5,
        "observations": [],
    }


def test_input_immutability():
    detection_batch = _detection_batch([_detection("person_1")])
    embedding_batch = _embedding_batch([_embedding("person_1")])
    original_detection_batch = deepcopy(detection_batch)
    original_embedding_batch = deepcopy(embedding_batch)

    Observe()(detection_batch, embedding_batch)

    assert detection_batch == original_detection_batch
    assert embedding_batch == original_embedding_batch
