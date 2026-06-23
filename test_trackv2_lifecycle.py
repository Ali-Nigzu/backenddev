from trackv2 import TrackV2, TrackV2Config


def observation(detection_id, timestamp, center=(100.0, 100.0)):
    return {
        "detection_id": detection_id,
        "timestamp": timestamp,
        "center": [float(center[0]), float(center[1])],
        "bbox": [float(center[0] - 10), float(center[1] - 20), float(center[0] + 10), float(center[1] + 20)],
        "confidence": 0.9,
        "embedding": [1.0, 0.0],
    }


def tracker_config():
    return TrackV2Config(
        tentative_hits_to_activate=1,
        unmatched_detection_buffer_frames=1,
        max_misses_active=2,
        max_misses_tentative=1,
        min_track_lifetime_sec=0.0,
    )


def create_active_track(tracker):
    _, assignment = tracker.update({0.0: [observation("initial", 0.0)]})
    runtime_track_id = assignment["initial"]
    track = tracker.tracks[0]
    assert track.runtime_track_id == runtime_track_id
    assert track.state == "ACTIVE"
    return track, runtime_track_id


def test_empty_frame_aging_increments_misses_and_closes():
    tracker = TrackV2(tracker_config())
    track, _ = create_active_track(tracker)

    tracker.update({}, current_timestamp=0.1)
    assert track.miss_count == 1
    assert track.state == "ACTIVE"
    assert tracker.frame_index == 2

    tracker.update({}, current_timestamp=0.2)
    assert track.miss_count == 2
    assert track.state == "ACTIVE"

    tracker.update({}, current_timestamp=0.3)
    assert track.miss_count == 3
    assert track.state == "CLOSED"


def test_long_absence_closes_track_with_empty_observation_lists():
    tracker = TrackV2(tracker_config())
    track, _ = create_active_track(tracker)

    tracker.update({0.1: []})
    tracker.update({0.2: []})
    tracker.update({5.0: []})

    assert track.state == "CLOSED"
    assert track.closed_timestamp == 5.0


def test_closed_track_is_not_resurrected_after_new_observation():
    tracker = TrackV2(tracker_config())
    old_track, old_runtime_track_id = create_active_track(tracker)

    tracker.update({}, current_timestamp=0.1)
    tracker.update({}, current_timestamp=0.2)
    tracker.update({}, current_timestamp=0.3)
    assert old_track.state == "CLOSED"

    _, assignment = tracker.update({1.0: [observation("return", 1.0)]})
    new_runtime_track_id = assignment["return"]

    assert new_runtime_track_id != old_runtime_track_id
    assert old_track.runtime_track_id not in assignment.values()
    assert old_track.state == "CLOSED"


def test_continuous_visibility_keeps_runtime_id_stable():
    tracker = TrackV2(tracker_config())
    observed_runtime_ids = []

    for frame_idx in range(5):
        timestamp = frame_idx * 0.1
        _, assignment = tracker.update({
            timestamp: [observation(f"visible-{frame_idx}", timestamp, center=(100 + frame_idx, 100))]
        })
        observed_runtime_ids.extend(assignment.values())

    assert len(set(observed_runtime_ids)) == 1
    assert tracker.tracks[0].state == "ACTIVE"
    assert tracker.tracks[0].miss_count == 0
