import json
from pathlib import Path

import cv2
import numpy as np

from test_tracking_v2_pipeline import (
    build_enriched_events,
    build_minimal_frame_batch,
    demographics_for_events,
    required_frame_ids,
    write_enriched_events,
)


class RaisingDemographic:
    def __init__(self, *args, **kwargs):
        raise AssertionError("Demographic should not be instantiated for zero events")


class FakeDemographic:
    instances = []
    def __init__(self, device="auto"):
        self.device = device
        self.calls = []
        FakeDemographic.instances.append(self)
    def __call__(self, event_batch, frame_batch):
        self.calls.append((event_batch, frame_batch))
        return {"results": [{"track_id": event["track_id"], "age": 44, "sex": 1} for event in event_batch["events"] if event["event_type"] == 1]}


def make_video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (8, 8))
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((8, 8, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_zero_event_run_exports_empty_and_skips_demographic(tmp_path):
    event_batch = {"events": []}
    result = demographics_for_events(event_batch, tmp_path / "missing.mp4", 10.0, "cpu", RaisingDemographic)
    assert result == {"results": []}
    enriched = build_enriched_events(event_batch, result)
    out = tmp_path / "nested" / "events.json"
    write_enriched_events(enriched, out)
    assert json.loads(out.read_text()) == {"events": []}


def test_required_frame_reread_and_timestamp_seconds(tmp_path):
    video = tmp_path / "video.mp4"
    make_video(video)
    batch = build_minimal_frame_batch(video, 10.0, ["frame-2"])
    assert batch["frames"][0]["frame_id"] == "frame-2"
    assert batch["frames"][0]["timestamp"] == 0.2
    assert batch["frames"][0]["image"].shape == (8, 8, 3)


def test_enriched_json_schema_ordering_and_cli_device(tmp_path):
    events = {"events": [
        {"track_id": "7", "timestamp": 2.0, "event_type": 0, "best_crop": {"frame_id": "frame-1", "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}},
        {"track_id": "7", "timestamp": 1.0, "event_type": 1, "best_crop": {"frame_id": "frame-1", "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}},
    ]}
    assert required_frame_ids(events) == ["frame-1"]
    enriched = build_enriched_events(events, {"results": [{"track_id": "7", "age": 32, "sex": 1}]})
    assert enriched == {"events": [
        {"track_id": "7", "timestamp": 1.0, "event_type": 1, "age": 32, "sex": 1},
        {"track_id": "7", "timestamp": 2.0, "event_type": 0, "age": 32, "sex": 1},
    ]}
    out = tmp_path / "events.json"
    write_enriched_events(enriched, out)
    text = out.read_text()
    assert '"best_crop"' not in text
    assert '"race"' not in text
