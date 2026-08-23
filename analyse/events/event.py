from math import hypot

_MIN_STABLE_SIDE_POINTS = 3
_MIN_EVENT_TRACK_POINTS = 6
_ON_LINE_DISTANCE_PIXELS = 2.0
_GEOMETRY_EPSILON = 1e-6

def _signed_distance_to_line(point, line_a, dx, dy, line_length):
    return (dx * (point[1] - line_a[1]) - dy * (point[0] - line_a[0])) / line_length

def _side(point, line_a, dx, dy, line_length):
    distance = _signed_distance_to_line(point, line_a, dx, dy, line_length)
    if distance > _ON_LINE_DISTANCE_PIXELS:
        return 1
    if distance < -_ON_LINE_DISTANCE_PIXELS:
        return -1
    return 0

def _events_for_track(track, line_a, dx, dy, line_length):
    if len(track["path"]) < _MIN_EVENT_TRACK_POINTS:
        return []
    events = []
    established_side = run_side = run_start_timestamp = None
    run_count = 0
    for path_point in track["path"]:
        timestamp = path_point["timestamp"]
        centre = path_point["centre"]
        observed_side = _side(
            (centre["x"], centre["y"]), line_a, dx, dy, line_length
        )
        if observed_side == 0:
            continue
        if observed_side != run_side:
            run_side = observed_side
            run_count = 1
            run_start_timestamp = timestamp
        else:
            run_count += 1
        if run_count != _MIN_STABLE_SIDE_POINTS:
            continue
        if established_side is None:
            established_side = run_side
        elif run_side != established_side:
            crop = track["best_crop"]
            events.append({
                "track_id": track["track_id"],
                "timestamp": run_start_timestamp,
                "event_type": 1 if established_side == -1 and run_side == 1 else 0,
                "best_crop": {
                    "frame_id": crop["frame_id"],
                    "bbox": dict(crop["bbox"]),
                },
            })
            established_side = run_side
    return events

class Event:
    __slots__ = ()

    def __call__(self, track_batch, line_config):
        point_a = line_config["point_a"]
        point_b = line_config["point_b"]
        line_a = (float(point_a["x"]), float(point_a["y"]))
        line_b = (float(point_b["x"]), float(point_b["y"]))
        dx = line_b[0] - line_a[0]
        dy = line_b[1] - line_a[1]
        line_length = hypot(dx, dy)
        events = []
        for track in track_batch["tracks"]:
            if (
                len(track["path"]) >= _MIN_EVENT_TRACK_POINTS
                and line_length <= _GEOMETRY_EPSILON
            ):
                raise ValueError("Line endpoints must define a non-zero line")
            events.extend(_events_for_track(track, line_a, dx, dy, line_length))
        return {"events": events}
