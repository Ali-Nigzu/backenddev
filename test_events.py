import copy
import inspect
import math

import pytest


LINE = {"point_a": {"x": 0.0, "y": 0.0}, "point_b": {"x": 10.0, "y": 0.0}}
BEST_CROP = {
    "frame_id": "frame_2",
    "bbox": {"x1": 1.0, "y1": 1.0, "x2": 10.0, "y2": 20.0},
}
NEG_Y = -3.0
POS_Y = 3.0
ON_Y = 0.0


def point(timestamp, x, y):
    return {"timestamp": float(timestamp), "centre": {"x": float(x), "y": float(y)}}


def side_point(timestamp, side, x=5.0):
    y_by_side = {"A": NEG_Y, "B": POS_Y, "O": ON_Y}
    return point(timestamp, x, y_by_side[side])


def side_path(pattern, start=1, x=5.0):
    return [side_point(start + index, side, x) for index, side in enumerate(pattern)]


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


def event_timestamps(result):
    return [event["timestamp"] for event in result["events"]]


def test_public_surface_exports_only_event():
    import events
    from events import Event

    assert events.__all__ == ["Event"]
    assert callable(Event)
    assert len(inspect.signature(Event).parameters) == 2
    assert not hasattr(events, "_MIN_STABLE_SIDE_POINTS")
    assert not hasattr(events, "_MIN_EVENT_TRACK_POINTS")
    assert not hasattr(events, "_ON_LINE_DISTANCE_PIXELS")
    assert not hasattr(events, "detect" + "_events")
    assert not hasattr(events, "Runtime" + "EventCandidate")
    assert not hasattr(events, "ENTRY")
    assert not hasattr(events, "EXIT")
    assert not hasattr(events, "Event" + "State")


@pytest.mark.parametrize(
    ("pattern", "expected_count"),
    [
        ("AA", 0),
        ("AABB", 0),
        ("AABBAA", 0),
        ("AAA", 0),
        ("AAABB", 0),
        ("AABBBB", 0),
        ("AAABBAAA", 0),
        ("ABABAB", 0),
        ("AABBAB", 0),
        ("AAABBB", 1),
        ("AAAABBBB", 1),
        ("AAABBBAA", 1),
        ("AAABBBBBB", 1),
        ("AAABBBAAA", 2),
        ("AAAABBBBBBAAAA", 2),
        ("AAABBBAAABBB", 3),
    ],
)
def test_stable_side_run_sequence_matrix(pattern, expected_count):
    from events import Event

    result = Event(state(track("1", side_path(pattern))), LINE)
    assert len(result["events"]) == expected_count


def test_stable_side_run_event_types_follow_direction():
    from events import Event

    assert event_types(Event(state(track("entry", side_path("AAABBB"))), LINE)) == [1]
    assert event_types(Event(state(track("exit", side_path("BBBAAA"))), LINE)) == [0]
    assert event_types(Event(state(track("multi", side_path("AAABBBAAABBB"))), LINE)) == [1, 0, 1]


@pytest.mark.parametrize(
    "tracking_state",
    [
        state(),
        state(track("1", [])),
        state(track("1", [point(1, 5, -3)])),
        state(track("1", side_path("ABBAA"))),
    ],
)
def test_empty_and_short_paths_return_empty_event_batch(tracking_state):
    from events import Event

    assert Event(tracking_state, LINE) == {"events": []}


def test_event_batch_and_event_shapes_are_locked():
    from events import Event

    result = Event(state(track("1", side_path("AAABBB"))), LINE)
    assert set(result) == {"events"}
    assert len(result["events"]) == 1
    event = result["events"][0]
    assert set(event) == {"track_id", "timestamp", "event_type", "best_crop"}
    assert "event_id" not in event
    assert "direction" not in event
    assert "supporting" + "_positions" not in event
    assert "run_count" not in event
    assert "stable_side_points" not in event
    assert "track_length" not in event
    assert "on_line_width" not in event
    assert "best_crop_confidence" not in event
    assert event["event_type"] == 1
    assert not any(value in ("ENTRY", "EXIT") for value in event.values())


def test_minimum_track_length_boundaries_and_eligibility():
    from events import Event

    assert Event(state(track("below", side_path("ABBAA"))), LINE) == {"events": []}
    assert event_types(Event(state(track("exact", side_path("AAABBB"))), LINE)) == [1]
    assert event_types(Event(state(track("above", side_path("AAABBBB"))), LINE)) == [1]
    assert Event(state(track("eligible_no_stable", side_path("AABBAA"))), LINE) == {"events": []}


def test_mixed_track_eligibility_preserves_order_and_skips_do_not_affect_later_tracks():
    from events import Event

    tracking_state = state(
        track("short", side_path("ABBAA")),
        track("entry", side_path("AAABBB")),
        track("exit", side_path("BBBAAA")),
    )
    result = Event(tracking_state, LINE)
    assert [(event["track_id"], event["event_type"]) for event in result["events"]] == [
        ("entry", 1),
        ("exit", 0),
    ]


@pytest.mark.parametrize("x", [5.0, -5.0, 15.0])
def test_crossing_anywhere_on_continuous_line_counts(x):
    from events import Event

    assert event_types(Event(state(track("1", side_path("AAABBB", x=x))), LINE)) == [1]


@pytest.mark.parametrize("x", [0.0, 10.0])
def test_crossing_through_anchor_points_has_no_special_endpoint_rule(x):
    from events import Event

    points = [
        point(1, x - 1, NEG_Y),
        point(2, x - 0.5, NEG_Y),
        point(3, x - 0.25, NEG_Y),
        point(4, x, ON_Y),
        point(5, x + 0.25, POS_Y),
        point(6, x + 0.5, POS_Y),
        point(7, x + 1, POS_Y),
    ]
    assert event_types(Event(state(track("1", points)), LINE)) == [1]


@pytest.mark.parametrize("x", [0.0, 5.0, 10.0, 15.0])
def test_touching_line_and_returning_same_side_does_not_count(x):
    from events import Event

    points = [
        point(1, x - 1, NEG_Y),
        point(2, x - 0.5, NEG_Y),
        point(3, x - 0.25, NEG_Y),
        point(4, x, ON_Y),
        point(5, x + 0.25, NEG_Y),
        point(6, x + 0.5, NEG_Y),
        point(7, x + 1, NEG_Y),
    ]
    assert Event(state(track("1", points)), LINE) == {"events": []}


def test_short_opposite_runs_do_not_replace_established_side():
    from events import Event

    assert Event(state(track("1", side_path("AAABAAA"))), LINE) == {"events": []}
    assert Event(state(track("1", side_path("AAABBAAA"))), LINE) == {"events": []}
    result = Event(state(track("1", side_path("AAABBAAABBB", start=10))), LINE)
    assert event_types(result) == [1]
    assert event_timestamps(result) == [18.0]


def test_additional_same_side_points_do_not_duplicate_event():
    from events import Event

    result = Event(state(track("1", side_path("AAABBBBBB"))), LINE)
    assert event_types(result) == [1]
    assert event_timestamps(result) == [4.0]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("AAAOABBB", [1]),
        ("AAOABBB", [1]),
        ("AAOBBAA", []),
        ("OAAABBB", [1]),
        ("AAAOOBBB", [1]),
    ],
)
def test_on_line_points_pause_runs_without_incrementing_or_resetting(pattern, expected):
    from events import Event

    assert event_types(Event(state(track("1", side_path(pattern))), LINE)) == expected


def test_on_line_points_do_not_change_run_start_timestamp():
    from events import Event

    points = [
        point(1, 5, NEG_Y),
        point(2, 5, NEG_Y),
        point(3, 5, ON_Y),
        point(4, 5, NEG_Y),
        point(10, 5, POS_Y),
        point(11, 5, ON_Y),
        point(12, 5, POS_Y),
        point(13, 5, POS_Y),
    ]
    result = Event(state(track("1", points)), LINE)
    assert event_types(result) == [1]
    assert event_timestamps(result) == [10.0]


@pytest.mark.parametrize(
    ("initial_pattern", "y", "expected"),
    [
        ("AAA", 0.0, []),
        ("AAA", 1.9, []),
        ("BBB", -1.9, []),
        ("AAA", 2.0, []),
        ("BBB", -2.0, []),
        ("AAA", 2.1, [1]),
        ("BBB", -2.1, [0]),
    ],
)
def test_on_line_deadband_boundaries_through_event_behavior(initial_pattern, y, expected):
    from events import Event

    points = side_path(initial_pattern) + [
        point(10, 5, y),
        point(11, 5, y),
        point(12, 5, y),
    ]
    assert event_types(Event(state(track("1", points)), LINE)) == expected


def test_geometry_deadband_uses_normalized_line_distance_through_event_behavior():
    from events import Event

    short_line = {"point_a": {"x": 0.0, "y": 0.0}, "point_b": {"x": 10.0, "y": 0.0}}
    long_line = {"point_a": {"x": 0.0, "y": 0.0}, "point_b": {"x": 1000000.0, "y": 0.0}}
    near_line_state = state(
        track("1", [point(i, 5, y) for i, y in enumerate([-3, -3, -3, 1.9, 1.9, 1.9], 1)])
    )
    clear_crossing = state(
        track("1", [point(i, 5, y) for i, y in enumerate([-3, -3, -3, 2.1, 2.2, 2.3], 1)])
    )

    assert Event(near_line_state, short_line) == {"events": []}
    assert Event(near_line_state, long_line) == {"events": []}
    assert event_types(Event(clear_crossing, short_line)) == [1]
    assert event_types(Event(clear_crossing, long_line)) == [1]


def test_timestamp_uses_first_stable_run_point_not_confirmation_or_final_timestamp():
    from events import Event

    points = [
        point(1, 5, NEG_Y),
        point(2, 5, NEG_Y),
        point(3, 5, NEG_Y),
        point(10, 5, POS_Y),
        point(11, 5, POS_Y),
        point(12, 5, POS_Y),
        point(99, 5, POS_Y),
    ]
    result = Event(state(track("1", points)), LINE)
    assert event_timestamps(result) == [10.0]


def test_reverse_crossings_use_each_stable_run_start_timestamp():
    from events import Event

    points = [
        point(1, 5, NEG_Y),
        point(2, 5, NEG_Y),
        point(3, 5, NEG_Y),
        point(10, 5, POS_Y),
        point(11, 5, POS_Y),
        point(12, 5, POS_Y),
        point(20, 5, NEG_Y),
        point(21, 5, NEG_Y),
        point(22, 5, NEG_Y),
    ]
    result = Event(state(track("1", points)), LINE)
    assert event_types(result) == [1, 0]
    assert event_timestamps(result) == [10.0, 20.0]


def test_swapping_line_endpoints_reverses_event_type():
    from events import Event

    reverse_line = {"point_a": {"x": 10.0, "y": 0.0}, "point_b": {"x": 0.0, "y": 0.0}}
    tracking_state = state(track("1", side_path("AAABBB")))
    assert event_types(Event(tracking_state, LINE)) == [1]
    assert event_types(Event(tracking_state, reverse_line)) == [0]


def test_best_crop_copied_and_confidence_excluded_and_ignored():
    from events import Event

    tracking_state = state(track("1", side_path("AAABBBAAA"), confidence=0.1))
    result = Event(tracking_state, LINE)
    assert len(result["events"]) == 2
    assert result["events"][0]["best_crop"] == BEST_CROP
    assert result["events"][1]["best_crop"] == BEST_CROP
    assert "best_crop_confidence" not in result["events"][0]
    assert result["events"][0]["best_crop"] is not tracking_state["tracks"][0]["best_crop"]
    assert (
        result["events"][0]["best_crop"]["bbox"]
        is not tracking_state["tracks"][0]["best_crop"]["bbox"]
    )
    result["events"][0]["best_crop"]["bbox"]["x1"] = 999.0
    assert tracking_state["tracks"][0]["best_crop"]["bbox"]["x1"] == 1.0


def test_event_does_not_mutate_inputs_or_reorder():
    from events import Event

    tracking_state = state(
        track("b", side_path("AAABBB", start=2)),
        track("a", side_path("BBBAAA", start=1)),
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
        state(track("1", [point(2, 5, -3), point(1, 5, 3)])),
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
