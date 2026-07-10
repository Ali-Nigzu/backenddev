from copy import deepcopy

from track import Track, TrackV2Config


def obs_batch(timestamp, observations):
    return {
        "frame_id": f"frame-{timestamp}",
        "timestamp": timestamp,
        "observations": observations,
    }


def obs(detection_id, x, y, confidence=0.5, emb=None):
    if emb is None:
        emb = [1.0, 0.0]
    return {
        "detection_id": detection_id,
        "bbox": {"x1": x - 1, "y1": y - 1, "x2": x + 1, "y2": y + 1},
        "center": {"x": x, "y": y},
        "embedding": {"dtype": "float32", "shape": [len(emb)], "values": emb},
        "confidence": confidence,
    }


def existing_track(track_id="1", timestamp=0.0, x=10.0, y=10.0):
    observation = obs("seed", x, y, confidence=0.7)
    return {
        "track_id": track_id,
        "path": [{"timestamp": timestamp, "center": {"x": x, "y": y}}],
        "best_crop": {
            "frame_id": "seed-frame",
            "bbox": observation["bbox"],
            "embedding": observation["embedding"],
        },
        "best_crop_confidence": observation["confidence"],
    }


def test_empty_state_creates_deterministic_numeric_ids():
    state = {"tracks": []}
    returned = Track(state, obs_batch(1.0, [obs("b", 20, 20), obs("a", 10, 10)]))

    assert returned is state
    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [track["path"][0]["center"]["x"] for track in state["tracks"]] == [
        10.0,
        20.0,
    ]


def test_existing_tracks_match_and_append_path():
    state = {"tracks": []}
    Track(state, obs_batch(0.0, [obs("a", 10, 10, emb=[1.0, 0.0])]))
    Track(state, obs_batch(1.0, [obs("b", 12, 10, emb=[1.0, 0.0])]))

    assert len(state["tracks"]) == 1
    assert state["tracks"][0]["track_id"] == "1"
    assert len(state["tracks"][0]["path"]) == 2
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 12.0, "y": 10.0}


def test_unmatched_observation_creates_next_max_id_without_filling_gaps():
    state = {"tracks": []}
    Track(state, obs_batch(0.0, [obs("a", 10, 10)]))
    state["tracks"][0]["track_id"] = "77"
    Track(state, obs_batch(1.0, [obs("far", 10000, 10000)]))

    assert [track["track_id"] for track in state["tracks"]] == ["77", "78"]


def test_track_never_prunes_tracks_with_empty_observations():
    state = {
        "tracks": [
            existing_track("1", timestamp=0.0),
            existing_track("2", timestamp=20.0),
        ]
    }

    returned = Track(state, obs_batch(100.0, []))

    assert returned is state
    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [1, 1]


def test_zero_observation_call_preserves_track_count_invariant():
    state = {
        "tracks": [
            existing_track("3", timestamp=0.0),
            existing_track("4", timestamp=5.0),
            existing_track("77", timestamp=10.0),
        ]
    }
    input_count = len(state["tracks"])

    Track(state, obs_batch(1000.0, []))

    assert len(state["tracks"]) == input_count


def test_best_crop_updates_only_on_higher_confidence():
    state = {"tracks": []}
    Track(
        state,
        obs_batch(0.0, [obs("a", 10, 10, confidence=0.5, emb=[1.0, 0.0])]),
    )
    Track(state, obs_batch(1.0, [obs("b", 12, 10, confidence=0.4, emb=[1.0, 0.0])]))
    assert state["tracks"][0]["best_crop"]["frame_id"] == "frame-0.0"
    assert state["tracks"][0]["best_crop_confidence"] == 0.5

    Track(state, obs_batch(2.0, [obs("c", 14, 10, confidence=0.9, emb=[1.0, 0.0])]))
    assert state["tracks"][0]["best_crop"]["frame_id"] == "frame-2.0"
    assert state["tracks"][0]["best_crop_confidence"] == 0.9


def test_deterministic_replay_from_same_inputs():
    initial_state = {"tracks": []}
    batch = obs_batch(1.0, [obs("b", 20, 20), obs("a", 10, 10)])

    state_a = deepcopy(initial_state)
    state_b = deepcopy(initial_state)

    assert Track(state_a, deepcopy(batch)) == Track(state_b, deepcopy(batch))


def test_new_track_id_uses_existing_max_id():
    state = {"tracks": [existing_track("156", timestamp=0.0)]}
    Track(state, obs_batch(100.0, [obs("new", 10000, 10000)]))

    assert [track["track_id"] for track in state["tracks"]] == ["156", "157"]
