from copy import deepcopy

import track.tracker as tracker_module
from track import Track
from track.config import _TrackConfig
from track.matcher import _Match, _best_for_observation, _continuity_key, _historical_anchor
from track.tracker import _ACTIVE, _INACTIVE, _TENTATIVE, _classify_track


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


def append_point(track, timestamp, x, y):
    track["path"].append({"timestamp": timestamp, "center": {"x": x, "y": y}})


def active_track(track_id="1", timestamp=2.0, x=10.0, y=10.0):
    track = existing_track(track_id, timestamp=timestamp - 2.0, x=x - 2.0, y=y)
    append_point(track, timestamp - 1.0, x - 1.0, y)
    append_point(track, timestamp, x, y)
    return track


def track_with_config(state, batch, config):
    original = tracker_module._CONFIG
    tracker_module._CONFIG = config
    try:
        return Track(state, batch)
    finally:
        tracker_module._CONFIG = original


def test_config_has_only_final_seven_fields():
    assert list(_TrackConfig.__dataclass_fields__) == [
        "location_history_window_frames",
        "anchor_weight_exponent",
        "max_anchor_distance_px",
        "anchor_tie_distance_px",
        "confirmation_hits",
        "active_timeout_frames",
        "tentative_timeout_frames",
    ]


def test_config_rejects_invalid_values():
    invalid_configs = [
        {"location_history_window_frames": 0},
        {"anchor_weight_exponent": -1.0},
        {"max_anchor_distance_px": 0.0},
        {"anchor_tie_distance_px": -1.0},
        {"confirmation_hits": 0},
        {"active_timeout_frames": 0},
        {"tentative_timeout_frames": 0},
        {"active_timeout_frames": 1.5},
    ]

    for kwargs in invalid_configs:
        try:
            _TrackConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config should fail: {kwargs}")


def test_empty_state_creates_deterministic_numeric_ids_from_sorted_observations():
    state = {"tracks": []}
    returned = Track(state, obs_batch(1.0, [obs("b", 20, 20), obs("a", 10, 10)]))

    assert returned is state
    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [track["path"][0]["center"]["x"] for track in state["tracks"]] == [10.0, 20.0]


def test_track_package_exports_only_track():
    import track

    assert track.__all__ == ["Track"]
    assert not hasattr(track, "TrackV2")
    assert not hasattr(track, "TrackV2Config")


def test_closest_anchor_wins():
    state = {
        "tracks": [
            active_track("1", timestamp=0.0, x=10.0, y=0.0),
            active_track("2", timestamp=0.0, x=100.0, y=0.0),
        ]
    }

    Track(state, obs_batch(1.0, [obs("near-two", 96.0, 0.0)]))

    assert [len(track["path"]) for track in state["tracks"]] == [3, 4]
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 96.0, "y": 0.0}


def test_active_beats_tentative_even_when_tentative_is_closer():
    active = active_track("1", timestamp=0.0, x=100.0, y=100.0)
    tentative = existing_track("2", timestamp=0.0, x=101.0, y=100.0)
    state = {"tracks": [active, tentative]}

    Track(state, obs_batch(1.0, [obs("could-fit-both", 101.0, 100.0)]))

    assert [len(track["path"]) for track in state["tracks"]] == [4, 1]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 101.0, "y": 100.0}


def test_one_to_one_matching_assigns_each_track_and_observation_once():
    state = {
        "tracks": [
            active_track("1", timestamp=0.0, x=0.0, y=0.0),
            active_track("2", timestamp=0.0, x=100.0, y=0.0),
        ]
    }

    Track(state, obs_batch(1.0, [obs("left", 1.0, 0.0), obs("right", 101.0, 0.0)]))

    assert [len(track["path"]) for track in state["tracks"]] == [4, 4]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 1.0, "y": 0.0}
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 101.0, "y": 0.0}


def test_tentative_matching_uses_remaining_observations_after_active_tier():
    state = {
        "tracks": [
            active_track("1", timestamp=0.0, x=0.0, y=0.0),
            existing_track("2", timestamp=0.0, x=50.0, y=0.0),
        ]
    }

    Track(state, obs_batch(1.0, [obs("for-active", 1.0, 0.0), obs("for-tentative", 51.0, 0.0)]))

    assert [len(track["path"]) for track in state["tracks"]] == [4, 2]
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 51.0, "y": 0.0}


def test_longer_history_wins_inside_tie_distance():
    short = active_track("1", timestamp=0.0, x=100.0, y=0.0)
    long = active_track("2", timestamp=0.0, x=104.0, y=0.0)
    append_point(long, 0.1, 104.0, 0.0)
    state = {"tracks": [short, long]}

    track_with_config(state, obs_batch(1.0, [obs("tie", 101.0, 0.0)]), _TrackConfig(anchor_tie_distance_px=5.0))

    assert [len(track["path"]) for track in state["tracks"]] == [3, 5]
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 101.0, "y": 0.0}


def test_older_id_wins_identical_tie_when_history_lengths_match():
    state = {
        "tracks": [
            active_track("1", timestamp=0.0, x=98.0, y=0.0),
            active_track("2", timestamp=0.0, x=102.0, y=0.0),
        ]
    }

    track_with_config(state, obs_batch(1.0, [obs("middle", 100.0, 0.0)]), _TrackConfig(anchor_tie_distance_px=5.0))

    assert [len(track["path"]) for track in state["tracks"]] == [4, 3]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 100.0, "y": 0.0}


def test_tentative_becomes_active_after_confirmation_hits():
    config = _TrackConfig(confirmation_hits=3)
    state = {"tracks": []}

    track_with_config(state, obs_batch(0.0, [obs("a", 10.0, 10.0)]), config)
    assert _classify_track(state["tracks"][0], 0.5, config).state == _TENTATIVE

    track_with_config(state, obs_batch(0.5, [obs("b", 11.0, 10.0)]), config)
    assert _classify_track(state["tracks"][0], 1.0, config).state == _TENTATIVE

    track_with_config(state, obs_batch(1.0, [obs("c", 12.0, 10.0)]), config)
    assert _classify_track(state["tracks"][0], 1.0, config).state == _ACTIVE


def test_inactive_track_stays_stored_but_cannot_match():
    config = _TrackConfig(active_timeout_frames=1)
    state = {"tracks": [active_track("1", timestamp=0.0, x=10.0, y=10.0)]}

    track_with_config(state, obs_batch(2.0, [obs("return", 11.0, 10.0)]), config)

    assert _classify_track(state["tracks"][0], 2.0, config).state == _INACTIVE
    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [3, 1]


def test_far_observation_creates_new_track_without_birth_suppression():
    state = {"tracks": [active_track("1", timestamp=0.0, x=0.0, y=0.0)]}

    track_with_config(state, obs_batch(1.0, [obs("far", 1000.0, 0.0)]), _TrackConfig(max_anchor_distance_px=100.0))

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [3, 1]


def test_embeddings_are_ignored_for_matching():
    state = {"tracks": [active_track("1", timestamp=0.0, x=10.0, y=10.0)]}

    Track(state, obs_batch(1.0, [obs("next", 12.0, 10.0, emb=[-1.0, 0.0])]))

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 4


def test_best_crop_updates_only_on_higher_confidence():
    state = {"tracks": []}
    Track(state, obs_batch(0.0, [obs("a", 10, 10, confidence=0.5)]))
    Track(state, obs_batch(1.0, [obs("b", 12, 10, confidence=0.4)]))
    assert state["tracks"][0]["best_crop"]["frame_id"] == "frame-0.0"
    assert state["tracks"][0]["best_crop_confidence"] == 0.5

    Track(state, obs_batch(2.0, [obs("c", 14, 10, confidence=0.9)]))
    assert state["tracks"][0]["best_crop"]["frame_id"] == "frame-2.0"
    assert state["tracks"][0]["best_crop_confidence"] == 0.9


def test_historical_anchor_uses_last_configured_path_points():
    track = existing_track("1", timestamp=0.0, x=0.0, y=0.0)
    append_point(track, 1.0, 10.0, 0.0)
    append_point(track, 2.0, 20.0, 0.0)
    append_point(track, 3.0, 30.0, 10.0)

    anchor = _historical_anchor(
        track["path"],
        _TrackConfig(location_history_window_frames=3, anchor_weight_exponent=0.0),
    )
    assert anchor == {"x": 20.0, "y": 10.0 / 3.0}


def test_historical_anchor_uses_recency_weighted_path_points():
    track = existing_track("1", timestamp=0.0, x=0.0, y=0.0)
    append_point(track, 1.0, 10.0, 0.0)
    append_point(track, 2.0, 20.0, 0.0)
    append_point(track, 3.0, 30.0, 10.0)

    anchor = _historical_anchor(
        track["path"],
        _TrackConfig(location_history_window_frames=3, anchor_weight_exponent=1.0),
    )
    assert anchor == {"x": (10.0 + 40.0 + 90.0) / 6.0, "y": 30.0 / 6.0}


def test_historical_anchor_weights_only_real_points_when_path_is_shorter_than_window():
    track = existing_track("1", timestamp=0.0, x=0.0, y=0.0)
    append_point(track, 1.0, 9.0, 6.0)

    anchor = _historical_anchor(
        track["path"],
        _TrackConfig(location_history_window_frames=5, anchor_weight_exponent=1.0),
    )
    assert anchor == {"x": 18.0 / 3.0, "y": 12.0 / 3.0}


def test_public_contract_shapes_do_not_gain_tracking_status_fields():
    state = {"tracks": []}

    returned = Track(state, obs_batch(0.0, [obs("first", 10.0, 10.0)]))

    assert returned is state
    assert set(state.keys()) == {"tracks"}
    assert set(state["tracks"][0].keys()) == {
        "track_id",
        "path",
        "best_crop",
        "best_crop_confidence",
    }
    assert set(state["tracks"][0]["path"][0].keys()) == {"timestamp", "center"}
    assert set(state["tracks"][0]["best_crop"].keys()) == {"frame_id", "bbox", "embedding"}


def test_repeated_identical_inputs_create_identical_outputs():
    initial_state = {
        "tracks": [
            active_track("1", timestamp=0.0, x=10.0, y=10.0),
            active_track("2", timestamp=0.0, x=100.0, y=100.0),
        ]
    }
    batch = obs_batch(1.0, [obs("b", 101.0, 100.0), obs("a", 11.0, 10.0)])

    outputs = []
    for _ in range(100):
        state = deepcopy(initial_state)
        outputs.append(deepcopy(Track(state, deepcopy(batch))))

    assert all(output == outputs[0] for output in outputs)


def test_matcher_ignores_exact_distance_inside_tie_window_for_continuity():
    long = active_track("1", timestamp=0.0, x=10.0, y=0.0)
    append_point(long, 0.1, 10.0, 0.0)
    short = active_track("2", timestamp=0.0, x=14.0, y=0.0)
    tracks = [long, short]
    config = _TrackConfig(anchor_tie_distance_px=5.0)
    statuses = [_classify_track(track, 1.0, config) for track in tracks]

    chosen = _best_for_observation(
        [
            _Match(track_index=0, observation_index=0, distance=5.0),
            _Match(track_index=1, observation_index=0, distance=1.0),
        ],
        [_continuity_key(track, status) for track, status in zip(tracks, statuses)],
        config,
    )

    assert chosen.track_index == 0


def test_matcher_closest_candidate_wins_outside_tie_window():
    long = active_track("1", timestamp=0.0, x=10.0, y=0.0)
    append_point(long, 0.1, 10.0, 0.0)
    short = active_track("2", timestamp=0.0, x=30.0, y=0.0)
    tracks = [long, short]
    config = _TrackConfig(anchor_tie_distance_px=5.0)
    statuses = [_classify_track(track, 1.0, config) for track in tracks]

    chosen = _best_for_observation(
        [
            _Match(track_index=0, observation_index=0, distance=7.0),
            _Match(track_index=1, observation_index=0, distance=1.0),
        ],
        [_continuity_key(track, status) for track, status in zip(tracks, statuses)],
        config,
    )

    assert chosen.track_index == 1
