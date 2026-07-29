import numpy as np
import pytest

from demographics import Demographic
from test_tracking_v2_pipeline import build_enriched_events


class FakeBackend:
    def __init__(self):
        self.calls = []
    def predict(self, batch):
        self.calls.append(batch)
        return [{"age": 31, "sex": 1}]


def event_batch():
    return {"events": [
        {"track_id": "7", "timestamp": 10.4, "event_type": 1, "best_crop": {"frame_id": "frame-1", "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}},
        {"track_id": "7", "timestamp": 26.8, "event_type": 0, "best_crop": {"frame_id": "frame-1", "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}},
    ]}


def test_event_to_demographic_preserves_timestamps_and_reuses_result():
    backend = FakeBackend()
    frame_batch = {"frames": [{"frame_id": "frame-1", "timestamp": 1.0, "image": np.ones((8, 8, 3), dtype=np.uint8)}]}
    demographics = Demographic(backend=backend)(event_batch(), frame_batch)
    assert demographics == {"results": [{"track_id": "7", "age": 31, "sex": 1}]}
    assert len(backend.calls) == 1
    assert backend.calls[0].shape[0] == 1
    enriched = build_enriched_events(event_batch(), demographics)
    assert [event["timestamp"] for event in enriched["events"]] == [10.4, 26.8]
    assert [event["age"] for event in enriched["events"]] == [31, 31]


def test_join_independent_of_demographics_order_and_failures():
    events = {"events": [
        {"track_id": "b", "timestamp": 2.0, "event_type": 1},
        {"track_id": "a", "timestamp": 1.0, "event_type": 0},
    ]}
    demographics = {"results": [{"track_id": "a", "age": 20, "sex": 0}, {"track_id": "b", "age": 21, "sex": 1}]}
    enriched = build_enriched_events(events, demographics)
    assert [event["track_id"] for event in enriched["events"]] == ["a", "b"]
    with pytest.raises(ValueError, match="Duplicate"):
        build_enriched_events(events, {"results": demographics["results"] + [demographics["results"][0]]})
    with pytest.raises(ValueError, match="Missing"):
        build_enriched_events(events, {"results": [demographics["results"][0]]})
    with pytest.raises(ValueError, match="without matching"):
        build_enriched_events({"events": [events["events"][0]]}, demographics)
