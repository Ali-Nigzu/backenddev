import hashlib
from typing import Any, Sequence

from .geometry import compute_side
from .models import RuntimeEventCandidate


MIN_STABLE_POINTS_AFTER_TRANSITION = 2
MAX_EVENTS_PER_TRACK = 1


def _field(track: Any, name: str) -> Any:
    if isinstance(track, dict):
        return track[name]
    return getattr(track, name)


def _point(point: Sequence[float]) -> list[float]:
    if len(point) != 2:
        raise ValueError("center_history points must contain exactly two coordinates")
    return [float(point[0]), float(point[1])]


def _line_points(line_config: dict) -> tuple[list[float], list[float]]:
    point_a = _point(line_config["point_a"])
    point_b = _point(line_config["point_b"])
    if point_a == point_b:
        raise ValueError("LineConfig point_a and point_b must define a non-zero line")
    return point_a, point_b


def _canonical(value: float) -> str:
    return format(float(value), ".12g")


def _stable_event_id(
    runtime_track_id: str,
    point_a: Sequence[float],
    point_b: Sequence[float],
    transition_index: int,
    event_type: str,
    direction: str,
) -> str:
    payload = "|".join([
        runtime_track_id,
        f"{_canonical(point_a[0])},{_canonical(point_a[1])}",
        f"{_canonical(point_b[0])},{_canonical(point_b[1])}",
        str(int(transition_index)),
        event_type,
        direction,
    ])
    return "evt_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _event_for_track(track: Any, point_a: Sequence[float], point_b: Sequence[float]) -> RuntimeEventCandidate | None:
    runtime_track_id = str(_field(track, "runtime_track_id"))
    timestamp = float(_field(track, "last_seen_timestamp"))
    center_history = [_point(point) for point in _field(track, "center_history")]

    compressed = []
    for original_index, point in enumerate(center_history):
        side = compute_side(point_a, point_b, point)
        if side != "ON":
            compressed.append((original_index, side, point))

    if len(compressed) < MIN_STABLE_POINTS_AFTER_TRANSITION + 1:
        return None

    for compressed_index in range(1, len(compressed)):
        previous_index, previous_side, _ = compressed[compressed_index - 1]
        transition_index, transition_side, _ = compressed[compressed_index]
        if previous_side == transition_side:
            continue

        stable_end_compressed_index = compressed_index + MIN_STABLE_POINTS_AFTER_TRANSITION - 1
        if stable_end_compressed_index >= len(compressed):
            return None

        stable_window = compressed[compressed_index:stable_end_compressed_index + 1]
        if any(side != transition_side for _, side, _ in stable_window):
            return None

        event_type, direction = (
            ("ENTRY", "IN") if previous_side == "A" and transition_side == "B" else ("EXIT", "OUT")
        )
        stable_end_original_index = stable_window[-1][0]
        supporting_positions = [
            _point(point)
            for point in center_history[previous_index:stable_end_original_index + 1]
        ]

        return RuntimeEventCandidate(
            event_id=_stable_event_id(
                runtime_track_id,
                point_a,
                point_b,
                transition_index,
                event_type,
                direction,
            ),
            runtime_track_id=runtime_track_id,
            timestamp=timestamp,
            event_type=event_type,
            direction=direction,
            supporting_positions=supporting_positions,
        )

    return None


def detect_events(tracks: list[Any], line_config: dict) -> list[RuntimeEventCandidate]:
    point_a, point_b = _line_points(line_config)
    events = []

    for track in tracks:
        event = _event_for_track(track, point_a, point_b)
        if event is not None:
            events.append(event)

    return sorted(
        events,
        key=lambda event: (
            event["timestamp"],
            event["runtime_track_id"],
            event["event_id"],
        ),
    )
