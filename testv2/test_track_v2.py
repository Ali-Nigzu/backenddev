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


def append_point(track, timestamp, x, y):
    track["path"].append({"timestamp": timestamp, "center": {"x": x, "y": y}})


def confirmed_track(track_id="1", timestamp=1.0, x=10.0, y=10.0):
    track = existing_track(track_id, timestamp=timestamp - 1.0, x=x - 1.0, y=y)
    append_point(track, timestamp, x, y)
    return track


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
    Track(state, obs_batch(4.0, [obs("far", 10000, 10000)]))

    assert [track["track_id"] for track in state["tracks"]] == ["77", "78"]


def test_impossible_observations_birth_even_when_observations_do_not_exceed_active_tracks():
    state = {
        "tracks": [
            confirmed_track("1", timestamp=0.0, x=10.0, y=10.0),
            confirmed_track("2", timestamp=0.0, x=100.0, y=100.0),
        ]
    }
    config = TrackV2Config(forced_continuity_break_normalized_motion=None)

    Track(
        state,
        obs_batch(1.0, [obs("far-a", 10000.0, 10000.0), obs("far-b", 20000.0, 20000.0)]),
        config,
    )

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2", "3", "4"]
    assert [len(track["path"]) for track in state["tracks"]] == [2, 2, 1, 1]


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


def test_one_track_one_observation_prefers_continuity_despite_embedding_disagreement():
    state = {"tracks": [existing_track("1", timestamp=0.0, x=10.0, y=10.0)]}

    Track(state, obs_batch(1.0, [obs("next", 14.0, 10.0, emb=[-1.0, 0.0])]))

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 2
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 14.0, "y": 10.0}


def test_appearance_threshold_does_not_fragment_physically_plausible_continuation():
    state = {"tracks": [existing_track("1", timestamp=0.0, x=10.0, y=10.0)]}
    config = TrackV2Config(min_appearance_similarity=0.99)

    Track(state, obs_batch(1.0, [obs("next", 14.0, 10.0, emb=[-1.0, 0.0])]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 2


def test_many_tracks_one_observation_reuses_most_physically_plausible_track():
    state = {
        "tracks": [
            confirmed_track("1", timestamp=0.0, x=10.0, y=10.0),
            confirmed_track("2", timestamp=0.0, x=100.0, y=100.0),
            confirmed_track("3", timestamp=0.0, x=200.0, y=200.0),
        ]
    }

    Track(state, obs_batch(1.0, [obs("only", 104.0, 100.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2", "3"]
    assert [len(track["path"]) for track in state["tracks"]] == [2, 3, 2]
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 104.0, "y": 100.0}


def test_one_track_multiple_observations_extends_existing_before_births():
    state = {"tracks": [confirmed_track("5", timestamp=0.0, x=50.0, y=50.0)]}

    Track(state, obs_batch(1.0, [obs("new-object", 500.0, 500.0), obs("same", 55.0, 50.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["5", "6"]
    assert len(state["tracks"][0]["path"]) == 3
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 55.0, "y": 50.0}
    assert state["tracks"][1]["path"][0]["center"] == {"x": 500.0, "y": 500.0}


def test_stale_track_does_not_long_term_reidentify_returning_object():
    state = {"tracks": [existing_track("1", timestamp=0.0, x=10.0, y=10.0)]}

    Track(state, obs_batch(5.0, [obs("returning", 12.0, 10.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [1, 1]


def test_motion_dominates_when_appearance_conflicts_with_clear_spatial_match():
    track_1 = confirmed_track("1", timestamp=0.0, x=10.0, y=10.0)
    track_2 = confirmed_track("2", timestamp=0.0, x=100.0, y=100.0)
    track_2["best_crop"]["embedding"] = obs("seed-2", 100.0, 100.0, emb=[0.0, 1.0])[
        "embedding"
    ]
    state = {"tracks": [track_1, track_2]}

    Track(state, obs_batch(1.0, [obs("near-1", 12.0, 10.0, emb=[0.0, 1.0])]))

    assert [len(track["path"]) for track in state["tracks"]] == [3, 2]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 12.0, "y": 10.0}


def test_recent_path_velocity_is_smoothed_to_resist_detector_jitter():
    state = {"tracks": [existing_track("1", timestamp=0.0, x=0.0, y=0.0)]}
    append_point(state["tracks"][0], 1.0, 10.0, 0.0)
    append_point(state["tracks"][0], 2.0, 20.0, 0.0)
    append_point(state["tracks"][0], 3.0, 24.0, 0.0)

    Track(state, obs_batch(4.0, [obs("next", 40.0, 0.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 5
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 40.0, "y": 0.0}


def test_confirmed_active_track_rejects_huge_jump_even_when_escape_valve_disabled():
    state = {"tracks": [confirmed_track("7", timestamp=0.0, x=10.0, y=10.0)]}
    config = TrackV2Config(forced_continuity_break_normalized_motion=None)

    Track(state, obs_batch(1.0, [obs("jump", 2000.0, 2000.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["7", "8"]
    assert [len(track["path"]) for track in state["tracks"]] == [2, 1]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 10.0, "y": 10.0}
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 2000.0, "y": 2000.0}


def test_default_forced_continuity_break_rejects_extreme_jump_and_creates_tentative_track():
    state = {"tracks": [confirmed_track("7", timestamp=0.0, x=10.0, y=10.0)]}

    Track(state, obs_batch(1.0, [obs("jump", 2000.0, 2000.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["7", "8"]
    assert [len(track["path"]) for track in state["tracks"]] == [2, 1]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 10.0, "y": 10.0}
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 2000.0, "y": 2000.0}


def test_forced_continuity_break_preserves_normal_walking_continuity():
    state = {"tracks": [confirmed_track("1", timestamp=0.0, x=10.0, y=10.0)]}
    config = TrackV2Config(forced_continuity_break_normalized_motion=0.1)

    Track(state, obs_batch(1.0, [obs("walk", 14.0, 10.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 3
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 14.0, "y": 10.0}


def test_forced_continuity_break_preserves_brief_occlusion_continuity():
    state = {"tracks": [confirmed_track("1", timestamp=0.0, x=10.0, y=10.0)]}
    config = TrackV2Config(
        max_reassociation_gap_sec=2.0,
        forced_continuity_break_normalized_motion=0.1,
    )

    Track(state, obs_batch(1.0, []), config)
    Track(state, obs_batch(1.5, [obs("return", 12.0, 10.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 3
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 12.0, "y": 10.0}


def test_forced_continuity_break_ignores_appearance_disagreement_for_plausible_motion():
    state = {"tracks": [confirmed_track("1", timestamp=0.0, x=10.0, y=10.0)]}
    config = TrackV2Config(forced_continuity_break_normalized_motion=0.1)

    Track(state, obs_batch(1.0, [obs("appearance", 12.0, 10.0, emb=[-1.0, 0.0])]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 3
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 12.0, "y": 10.0}


def test_forced_continuity_break_multi_person_scene_is_deterministic():
    config = TrackV2Config(forced_continuity_break_normalized_motion=8.0)
    initial_state = {
        "tracks": [
            confirmed_track("1", timestamp=0.0, x=10.0, y=10.0),
            confirmed_track("2", timestamp=0.0, x=100.0, y=100.0),
        ]
    }
    batch = obs_batch(
        1.0,
        [
            obs("far", 2000.0, 2000.0),
            obs("near-2", 104.0, 100.0),
        ],
    )
    state_a = deepcopy(initial_state)
    state_b = deepcopy(initial_state)

    assert Track(state_a, deepcopy(batch), config) == Track(state_b, deepcopy(batch), config)
    assert [track["track_id"] for track in state_a["tracks"]] == ["1", "2", "3"]
    assert [len(track["path"]) for track in state_a["tracks"]] == [2, 3, 1]
    assert state_a["tracks"][1]["path"][-1]["center"] == {"x": 104.0, "y": 100.0}
    assert state_a["tracks"][2]["path"][-1]["center"] == {"x": 2000.0, "y": 2000.0}


def test_forced_continuity_break_does_not_reoptimize_remaining_selected_matches():
    config = TrackV2Config(forced_continuity_break_normalized_motion=15.0)
    state = {
        "tracks": [
            confirmed_track("1", timestamp=1.0, x=0.0, y=0.0),
            confirmed_track("2", timestamp=1.0, x=100.0, y=0.0),
        ]
    }

    Track(
        state,
        obs_batch(
            2.0,
            [
                obs("far-new", 800.0, 0.0),
                obs("near-2", 102.0, 0.0),
            ],
        ),
        config,
    )

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2", "3"]
    assert [len(track["path"]) for track in state["tracks"]] == [2, 3, 1]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 0.0, "y": 0.0}
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 102.0, "y": 0.0}
    assert state["tracks"][2]["path"][-1]["center"] == {"x": 800.0, "y": 0.0}


def test_five_confirmed_active_tracks_three_observations_zero_births():
    state = {
        "tracks": [
            confirmed_track(str(index), timestamp=0.0, x=float(index * 10), y=0.0)
            for index in range(1, 6)
        ]
    }

    Track(state, obs_batch(1.0, [obs("a", 10.0, 0.0), obs("b", 20.0, 0.0), obs("c", 30.0, 0.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2", "3", "4", "5"]
    assert sum(len(track["path"]) == 3 for track in state["tracks"]) == 3


def test_five_confirmed_active_tracks_five_observations_zero_births():
    state = {
        "tracks": [
            confirmed_track(str(index), timestamp=0.0, x=float(index * 10), y=0.0)
            for index in range(1, 6)
        ]
    }

    Track(
        state,
        obs_batch(
            1.0,
            [obs(str(index), float(index * 10), 0.0) for index in range(1, 6)],
        ),
    )

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2", "3", "4", "5"]
    assert [len(track["path"]) for track in state["tracks"]] == [3, 3, 3, 3, 3]


def test_five_confirmed_active_tracks_seven_observations_creates_two_births():
    state = {
        "tracks": [
            confirmed_track(str(index), timestamp=0.0, x=float(index * 10), y=0.0)
            for index in range(1, 6)
        ]
    }

    Track(
        state,
        obs_batch(
            1.0,
            [obs(str(index), float(index * 10), 0.0) for index in range(1, 8)],
        ),
    )

    assert [track["track_id"] for track in state["tracks"]] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ]
    assert [len(track["path"]) for track in state["tracks"][:5]] == [3, 3, 3, 3, 3]
    assert [len(track["path"]) for track in state["tracks"][5:]] == [1, 1]


def test_confirmed_track_continues_with_poor_detection_confidence():
    state = {"tracks": [confirmed_track("1", timestamp=0.0, x=10.0, y=10.0)]}

    Track(state, obs_batch(1.0, [obs("poor", 11.0, 10.0, confidence=0.01)]))

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 3


def test_confirmed_track_receives_first_claim_over_tentative_track():
    confirmed = confirmed_track("1", timestamp=0.0, x=100.0, y=100.0)
    tentative = existing_track("2", timestamp=0.0, x=101.0, y=100.0)
    state = {"tracks": [confirmed, tentative]}

    Track(state, obs_batch(1.0, [obs("could-fit-both", 101.0, 100.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [3, 1]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 101.0, "y": 100.0}


def test_only_tentative_track_can_continue_and_mature():
    state = {"tracks": []}

    Track(state, obs_batch(0.0, [obs("first", 10.0, 10.0)]))
    Track(state, obs_batch(1.0, [obs("second", 11.0, 10.0)]))
    Track(state, obs_batch(2.0, [obs("third", 12.0, 10.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 3
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 12.0, "y": 10.0}


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


def test_continuous_tracking_within_reassociation_gap_keeps_same_id():
    config = TrackV2Config(max_reassociation_gap_sec=0.5)
    state = {"tracks": []}

    for index, timestamp in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        Track(state, obs_batch(timestamp, [obs(f"a-{index}", 10.0 + index, 10.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 5
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 14.0, "y": 10.0}


def test_gap_within_reassociation_gap_keeps_confirmed_identity():
    config = TrackV2Config(max_reassociation_gap_sec=0.5)
    state = {"tracks": []}

    Track(state, obs_batch(0.0, [obs("a-0", 10.0, 10.0)]), config)
    Track(state, obs_batch(0.25, [obs("a-1", 11.0, 10.0)]), config)
    Track(state, obs_batch(0.5, []), config)
    Track(state, obs_batch(0.6, []), config)
    Track(state, obs_batch(0.7, [obs("a-2", 12.0, 10.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1"]
    assert len(state["tracks"][0]["path"]) == 3
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 12.0, "y": 10.0}


def test_gap_beyond_reassociation_gap_creates_new_id_for_confirmed_track():
    config = TrackV2Config(
        max_reassociation_gap_sec=0.5,
        active_recency_window_frames=10,
        active_track_window_frames=10,
    )
    state = {"tracks": []}

    Track(state, obs_batch(0.0, [obs("a-0", 10.0, 10.0)]), config)
    Track(state, obs_batch(0.25, [obs("a-1", 11.0, 10.0)]), config)
    Track(state, obs_batch(0.6, []), config)
    Track(state, obs_batch(0.8, []), config)
    Track(state, obs_batch(1.0, [obs("b-0", 12.0, 10.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [2, 1]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 11.0, "y": 10.0}
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 12.0, "y": 10.0}


def test_stale_confirmed_track_cannot_suppress_new_birth():
    config = TrackV2Config(
        max_reassociation_gap_sec=0.5,
        active_recency_window_frames=10,
        active_track_window_frames=10,
    )
    state = {"tracks": [confirmed_track("1", timestamp=0.0, x=10.0, y=10.0)]}

    Track(state, obs_batch(1.0, [obs("new-person", 20.0, 20.0)]), config)

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [2, 1]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 10.0, "y": 10.0}
    assert state["tracks"][1]["path"][-1]["center"] == {"x": 20.0, "y": 20.0}


def test_diagnostics_explain_eligible_match_and_birth():
    diagnostics = []
    config = TrackV2Config(debug_diagnostics=diagnostics)
    state = {"tracks": [confirmed_track("1", timestamp=0.0, x=10.0, y=10.0)]}

    Track(state, obs_batch(1.0, [obs("near", 12.0, 10.0), obs("teleport", 5000.0, 5000.0)]), config)

    by_id = {item["observation_id"]: item for item in diagnostics}
    assert by_id["near"]["final_assignment"]["track_id"] == "1"
    assert by_id["near"]["candidate_tracks"][0]["eligible"] is True
    assert by_id["near"]["candidate_tracks"][0]["classification"] in {"strong", "normal", "weak"}
    assert by_id["teleport"]["final_assignment"] == "birth"
    assert by_id["teleport"]["candidate_tracks"][0]["eligible"] is False


def test_observation_specific_eligibility_births_unexplained_new_person_with_active_tracks():
    state = {
        "tracks": [
            confirmed_track("1", timestamp=0.0, x=10.0, y=10.0),
            confirmed_track("2", timestamp=0.0, x=100.0, y=100.0),
        ]
    }

    Track(state, obs_batch(1.0, [obs("near-1", 12.0, 10.0), obs("new", 900.0, 900.0)]))

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2", "3"]
    assert [len(track["path"]) for track in state["tracks"]] == [3, 2, 1]
    assert state["tracks"][0]["path"][-1]["center"] == {"x": 12.0, "y": 10.0}
    assert state["tracks"][2]["path"][-1]["center"] == {"x": 900.0, "y": 900.0}


def test_configurable_hard_speed_limit_controls_eligibility():
    state = {"tracks": [confirmed_track("1", timestamp=0.0, x=0.0, y=0.0)]}
    strict = TrackV2Config(hard_speed_limit_px_per_sec=5.0)

    Track(state, obs_batch(1.0, [obs("too-fast", 20.0, 0.0)]), strict)

    assert [track["track_id"] for track in state["tracks"]] == ["1", "2"]
    assert [len(track["path"]) for track in state["tracks"]] == [2, 1]


def test_deterministic_replay_many_runs_with_eligibility():
    initial_state = {
        "tracks": [
            confirmed_track("1", timestamp=0.0, x=10.0, y=10.0),
            confirmed_track("2", timestamp=0.0, x=100.0, y=100.0),
        ]
    }
    batch = obs_batch(1.0, [obs("b", 101.0, 100.0), obs("a", 11.0, 10.0)])

    outputs = []
    for _ in range(100):
        state = deepcopy(initial_state)
        outputs.append(deepcopy(Track(state, deepcopy(batch))))

    assert all(output == outputs[0] for output in outputs)
