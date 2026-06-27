from events import detect_events


LINE_CONFIG = {
    "point_a": [400.0, 0.0],
    "point_b": [400.0, 100.0],
}


def _track(points, track_id="track", timestamp=1.0):
    return {
        "runtime_track_id": track_id,
        "last_seen_timestamp": timestamp,
        "center_history": points,
    }


def _event_types(points):
    return [
        event["event_type"]
        for event in detect_events([_track(points)], LINE_CONFIG)
    ]


def test_single_crossing_entry_remains_supported():
    assert _event_types([[390, 0], [395, 0], [405, 0], [410, 0]]) == ["ENTRY"]


def test_single_crossing_exit_remains_supported():
    assert _event_types([[410, 0], [405, 0], [395, 0], [390, 0]]) == ["EXIT"]


def test_multiple_crossings_in_one_track_emit_chronologically():
    events = detect_events([
        _track([[390, 0], [395, 0], [405, 0], [410, 0], [395, 0], [390, 0], [405, 0], [410, 0]])
    ], LINE_CONFIG)

    assert [event["event_type"] for event in events] == ["ENTRY", "EXIT", "ENTRY"]
    assert [event["direction"] for event in events] == ["IN", "OUT", "IN"]
    assert len({event["event_id"] for event in events}) == 3


def test_no_crossing_on_one_side_emits_no_events():
    assert detect_events([_track([[410, 0], [420, 0], [430, 0]])], LINE_CONFIG) == []


def test_on_line_points_do_not_create_false_crossings():
    assert detect_events([_track([[400, 0], [400, 10], [400, 20]])], LINE_CONFIG) == []


def test_on_line_points_between_sides_are_preserved_in_supporting_positions():
    events = detect_events([_track([[390, 0], [400, 0], [405, 0], [400, 10], [410, 0]])], LINE_CONFIG)

    assert [event["event_type"] for event in events] == ["ENTRY"]
    assert events[0]["supporting_positions"] == [[390.0, 0.0], [400.0, 0.0], [405.0, 0.0], [400.0, 10.0], [410.0, 0.0]]


def test_oscillation_is_reported_as_deterministic_geometry_crossings():
    assert _event_types([[390, 0], [405, 0], [390, 0]]) == ["ENTRY", "EXIT"]


def test_deterministic_output_for_repeated_and_reversed_inputs():
    tracks = [
        _track([[390, 0], [405, 0]], "b", timestamp=1.0),
        _track([[410, 0], [395, 0]], "a", timestamp=1.0),
        _track([[390, 0], [405, 0], [390, 0]], "c", timestamp=2.0),
    ]

    first = detect_events(tracks, LINE_CONFIG)
    second = detect_events(tracks, LINE_CONFIG)
    reversed_input = detect_events(list(reversed(tracks)), LINE_CONFIG)

    assert first == second == reversed_input
    assert [event["runtime_track_id"] for event in first] == ["a", "b", "c", "c"]
    assert [event["event_type"] for event in first] == ["EXIT", "ENTRY", "ENTRY", "EXIT"]


def test_detector_is_silent_by_default(capsys):
    detect_events([_track([[390, 0], [405, 0], [390, 0]])], LINE_CONFIG)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
