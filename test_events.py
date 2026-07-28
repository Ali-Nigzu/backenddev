import copy
import inspect
import math

import pytest


LINE = {"point_a": {"x": 0.0, "y": 0.0}, "point_b": {"x": 10.0, "y": 0.0}}
BEST_CROP = {
    "frame_id": "frame_2",
    "bbox": {"x1": 1.0, "y1": 1.0, "x2": 10.0, "y2": 20.0},
}


def point(timestamp, x, y):
    return {"timestamp": float(timestamp), "centre": {"x": float(x), "y": float(y)}}


def track(track_id, points, best_crop=None, confidence=0.9):
    return {
        "track_id": str(track_id),
        "path": points,
        "best_crop": copy.deepcopy(best_crop or BEST_CROP),
        "best_crop_confidence": confidence,
    }


def state(*tracks):
    return {"tracks": list(tracks)}


def event_types(result):
    return [event["event_type"] for event in result["events"]]


def test_public_surface_exports_only_event():
    import events
    from events import Event

    assert events.__all__ == ["Event"]
    assert callable(Event)
    assert len(inspect.signature(Event).parameters) == 2
    assert not hasattr(events, "detect" + "_events")
    assert not hasattr(events, "Runtime" + "EventCandidate")
    assert not hasattr(events, "ENTRY")
    assert not hasattr(events, "EXIT")
    assert not hasattr(events, "EventState")


@pytest.mark.parametrize(
    "tracking_state",
    [state(), state(track("1", [])), state(track("1", [point(1, 5, -1)]))],
)
def test_empty_and_short_paths_return_empty_event_batch(tracking_state):
    from events import Event

    assert Event(tracking_state, LINE) == {"events": []}


def test_event_batch_and_event_shapes_are_locked():
    from events import Event

    result = Event(state(track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, 2)])), LINE)
    assert set(result) == {"events"}
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert set(event) == {"track_id", "timestamp", "event_type", "best_crop"}
    assert "event_id" not in event
    assert "direction" not in event
    assert "supporting" + "_positions" not in event
    assert "candidate_count" not in event
    assert "best_crop_confidence" not in event
    assert event["event_type"] == 1
    assert not any(value in ("ENTRY", "EXIT") for value in event.values())


def test_negative_to_positive_continuous_crossing_returns_one_entry():
    from events import Event

    result = Event(state(track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, 2)])), LINE)
    assert event_types(result) == [1]
    assert result["events"][0]["timestamp"] == 2.0


def test_positive_to_negative_continuous_crossing_returns_one_exit():
    from events import Event

    result = Event(state(track("1", [point(1, 5, 1), point(2, 5, -1), point(3, 5, -2)])), LINE)
    assert event_types(result) == [0]
    assert result["events"][0]["timestamp"] == 2.0


def test_no_side_transition_returns_no_event():
    from events import Event

    assert Event(state(track("1", [point(1, 2, -1), point(2, 8, -2)])), LINE) == {
        "events": []
    }


def test_duplicate_stationary_points_do_not_create_events():
    from events import Event

    assert Event(state(track("1", [point(1, 5, -1), point(2, 5, -1)])), LINE) == {
        "events": []
    }
    result = Event(
        state(
            track(
                "1",
                [point(1, 5, -1), point(2, 5, -1), point(3, 5, 1), point(4, 5, 1)],
            )
        ),
        LINE,
    )
    assert event_types(result) == [1]
    assert Event(state(track("1", [point(1, 5, 0), point(2, 5, 0)])), LINE) == {
        "events": []
    }


def test_multiple_crossings_by_one_track():
    from events import Event

    result = Event(
        state(
            track(
                "1",
                [
                    point(1, 5, -1),
                    point(2, 5, 1),
                    point(3, 5, 2),
                    point(4, 5, -1),
                    point(5, 5, -2),
                    point(6, 5, 1),
                    point(7, 5, 2),
                ],
            )
        ),
        LINE,
    )
    assert event_types(result) == [1, 0, 1]
    assert [event["timestamp"] for event in result["events"]] == [2.0, 4.0, 6.0]


def test_multiple_tracks_and_existing_order_preserved():
    from events import Event

    tracking_state = state(
        track("z", [point(1, 5, -1), point(3, 5, 1), point(4, 5, 2)]),
        track("1", [point(1, 5, 1), point(2, 5, -1), point(3, 5, -2)]),
    )
    result = Event(tracking_state, LINE)
    assert [(event["track_id"], event["event_type"]) for event in result["events"]] == [
        ("z", 1),
        ("1", 0),
    ]


@pytest.mark.parametrize("x", [5.0, -5.0, 15.0])
def test_crossing_anywhere_on_continuous_line_counts(x):
    from events import Event

    assert event_types(
        Event(state(track("1", [point(1, x, -1), point(2, x, 1), point(3, x, 2)])), LINE)
    ) == [1]


@pytest.mark.parametrize("x", [0.0, 10.0])
def test_crossing_through_anchor_points_has_no_special_endpoint_rule(x):
    from events import Event

    assert event_types(
        Event(
            state(
                track(
                    "1",
                    [point(1, x - 1, -1), point(2, x, 0), point(3, x + 1, 1), point(4, x + 1, 2)],
                )
            ),
            LINE,
        )
    ) == [1]


@pytest.mark.parametrize("x", [0.0, 5.0, 10.0, 15.0])
def test_touching_line_and_returning_same_side_does_not_count(x):
    from events import Event

    assert Event(
        state(track("1", [point(1, x - 1, -1), point(2, x, 0), point(3, x + 1, -1)])),
        LINE,
    ) == {"events": []}


def test_isolated_one_point_jump_is_suppressed():
    from events import Event

    result = Event(
        state(track("1", [point(1, 5, -1), point(2, 5, -2), point(3, 5, 1), point(4, 5, -1), point(5, 5, -2)])),
        LINE,
    )
    assert result == {"events": []}


def test_alternating_jitter_is_suppressed():
    from events import Event

    result = Event(
        state(track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, -1), point(4, 5, 1), point(5, 5, -1)])),
        LINE,
    )
    assert result == {"events": []}


def test_candidate_rejection_preserves_confirmed_side_and_later_candidate_can_confirm():
    from events import Event

    result = Event(
        state(track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, -1), point(4, 5, 1), point(5, 5, 2)])),
        LINE,
    )
    assert event_types(result) == [1]
    assert result["events"][0]["timestamp"] == 4.0


def test_additional_same_side_points_do_not_duplicate_event():
    from events import Event

    result = Event(
        state(track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, 2), point(4, 5, 3), point(5, 5, 4)])),
        LINE,
    )
    assert event_types(result) == [1]


def test_on_line_sequences():
    from events import Event

    assert event_types(
        Event(
            state(track("1", [point(1, 5, -1), point(2, 5, 0), point(3, 5, 1), point(4, 5, 2)])), LINE
        )
    ) == [1]
    assert event_types(
        Event(
            state(track("1", [point(1, 5, 1), point(2, 5, 0), point(3, 5, -1), point(4, 5, -2)])), LINE
        )
    ) == [0]
    assert Event(
        state(track("1", [point(1, 5, -1), point(2, 5, 0), point(3, 5, -1)])), LINE
    ) == {"events": []}
    assert Event(
        state(track("1", [point(1, 5, 1), point(2, 5, 0), point(3, 5, 1)])), LINE
    ) == {"events": []}
    assert Event(state(track("1", [point(1, 5, 0), point(2, 5, 1), point(3, 5, 2)])), LINE) == {
        "events": []
    }
    assert Event(state(track("1", [point(1, 2, 0), point(2, 8, 0)])), LINE) == {
        "events": []
    }
    assert event_types(
        Event(
            state(
                track(
                    "1",
                    [point(1, 5, -1), point(2, 2, 0), point(3, 8, 0), point(4, 5, 1), point(5, 5, 2)],
                )
            ),
            LINE,
        )
    ) == [1]


def test_on_line_points_do_not_increment_or_reject_candidate():
    from events import Event

    result = Event(
        state(track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, 0), point(4, 5, 2)])),
        LINE,
    )
    assert event_types(result) == [1]
    assert result["events"][0]["timestamp"] == 2.0


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([point(1, 5, -1), point(2, 5, -2), point(3, 5, 1)], []),
        ([point(1, 5, 1), point(2, 5, 2), point(3, 5, -1)], []),
        ([point(1, 5, -1), point(2, 5, -2), point(3, 5, 1), point(4, 5, 2)], [1]),
        ([point(1, 5, 1), point(2, 5, 2), point(3, 5, -1), point(4, 5, -2)], [0]),
    ],
)
def test_track_end_candidate_behavior(points, expected):
    from events import Event

    assert event_types(Event(state(track("1", points)), LINE)) == expected


def test_swapping_line_endpoints_reverses_event_type():
    from events import Event

    reverse_line = {"point_a": {"x": 10.0, "y": 0.0}, "point_b": {"x": 0.0, "y": 0.0}}
    tracking_state = state(track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, 2)]))
    assert event_types(Event(tracking_state, LINE)) == [1]
    assert event_types(Event(tracking_state, reverse_line)) == [0]


def test_timestamp_uses_first_candidate_side_not_confirmation_or_final_timestamp():
    from events import Event

    result = Event(
        state(
            track(
                "1",
                [point(1, 5, -1), point(2, 5, 0), point(3, 5, 1), point(4, 5, 2), point(99, 6, 2)],
            )
        ),
        LINE,
    )
    assert [event["timestamp"] for event in result["events"]] == [3.0]


def test_multiple_crossings_use_each_candidate_start_timestamp():
    from events import Event

    result = Event(
        state(track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, 2), point(4, 5, -1), point(5, 5, -2)])),
        LINE,
    )
    assert [event["timestamp"] for event in result["events"]] == [2.0, 4.0]


def test_geometry_epsilon_uses_normalized_line_distance():
    from events import Event

    near_line_state = state(track("1", [point(1, 5, -1), point(2, 5, 5e-7), point(3, 5, 2)]))
    assert Event(near_line_state, LINE) == {"events": []}

    long_line = {"point_a": {"x": 0.0, "y": 0.0}, "point_b": {"x": 1000000.0, "y": 0.0}}
    clear_crossing = state(track("1", [point(1, 5, -1), point(2, 5, 1e-5), point(3, 5, 2e-5)]))
    assert event_types(Event(clear_crossing, LINE)) == [1]
    assert event_types(Event(clear_crossing, long_line)) == [1]


def test_best_crop_copied_and_confidence_excluded_and_ignored():
    from events import Event

    tracking_state = state(
        track("1", [point(1, 5, -1), point(2, 5, 1), point(3, 5, 2), point(4, 5, -1), point(5, 5, -2)], confidence=0.1)
    )
    result = Event(tracking_state, LINE)
    assert len(result["events"]) == 2
    assert result["events"][0]["best_crop"] == BEST_CROP
    assert result["events"][1]["best_crop"] == BEST_CROP
    assert "best_crop_confidence" not in result["events"][0]
    assert (
        result["events"][0]["best_crop"] is not tracking_state["tracks"][0]["best_crop"]
    )
    assert (
        result["events"][0]["best_crop"]["bbox"]
        is not tracking_state["tracks"][0]["best_crop"]["bbox"]
    )
    result["events"][0]["best_crop"]["bbox"]["x1"] = 999.0
    assert tracking_state["tracks"][0]["best_crop"]["bbox"]["x1"] == 1.0


def test_event_does_not_mutate_inputs_or_reorder():
    from events import Event

    tracking_state = state(
        track("b", [point(2, 5, -1), point(3, 5, 1), point(4, 5, 2)]),
        track("a", [point(1, 5, 1), point(4, 5, -1), point(5, 5, -2)]),
    )
    line_config = copy.deepcopy(LINE)
    original_state = copy.deepcopy(tracking_state)
    original_line = copy.deepcopy(line_config)
    first = Event(tracking_state, line_config)
    second = Event(tracking_state, line_config)
    assert tracking_state == original_state
    assert line_config == original_line
    assert first == second
    assert [track["track_id"] for track in tracking_state["tracks"]] == ["b", "a"]
    assert [(event["track_id"], event["event_type"]) for event in first["events"]] == [("b", 1), ("a", 0)]


@pytest.mark.parametrize(
    "bad_state",
    [
        {},
        {"tracks": "nope"},
        {"tracks": ["nope"]},
        state({"path": [], "best_crop": BEST_CROP}),
        state({"track_id": "1", "best_crop": BEST_CROP}),
        state(track("1", [{"timestamp": 1.0}])),
        state(track("1", [{"timestamp": math.inf, "centre": {"x": 1, "y": 2}}])),
        state(track("1", [point(1, math.nan, 2)])),
        state({"track_id": "1", "path": [], "best_crop": {}}),
        state(
            {
                "track_id": "1",
                "path": [],
                "best_crop": {"frame_id": "", "bbox": BEST_CROP["bbox"]},
            }
        ),
        state(
            {
                "track_id": "1",
                "path": [],
                "best_crop": {
                    "frame_id": "f",
                    "bbox": {"x1": 2, "y1": 1, "x2": 1, "y2": 2},
                },
            }
        ),
        state(track("1", [point(2, 5, -1), point(1, 5, 1)])),
    ],
)
def test_invalid_tracking_state_rejected(bad_state):
    from events import Event

    with pytest.raises(ValueError):
        Event(bad_state, LINE)


@pytest.mark.parametrize(
    "bad_line",
    [
        {},
        {"point_a": {"x": 0, "y": 0}},
        {"point_a": {"x": 0, "y": 0}, "point_b": {"x": 0}},
        {"point_a": {"x": 0, "y": 0}, "point_b": {"x": math.inf, "y": 0}},
    ],
)
def test_invalid_line_config_rejected(bad_line):
    from events import Event

    with pytest.raises(ValueError):
        Event(state(), bad_line)


def test_coincident_line_config_rejected_with_line_wording():
    from events import Event

    with pytest.raises(ValueError, match="non-zero line"):
        Event(state(), {"point_a": {"x": 1, "y": 1}, "point_b": {"x": 1, "y": 1}})
