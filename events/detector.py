import hashlib
from typing import Any, Sequence

from .geometry import compute_side, line_points_from_config
from .models import RuntimeEventCandidate


def _field(track: Any, name: str) -> Any:
    if isinstance(track, dict):
        return track[name]
    return getattr(track, name)


def _point(point: Sequence[float]) -> list[float]:
    if len(point) != 2:
        raise ValueError("center_history points must contain exactly two coordinates")
    return [float(point[0]), float(point[1])]


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
    payload = "|".join(
        [
            runtime_track_id,
            f"{_canonical(point_a[0])},{_canonical(point_a[1])}",
            f"{_canonical(point_b[0])},{_canonical(point_b[1])}",
            str(int(transition_index)),
            event_type,
            direction,
        ]
    )
    return "evt_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _make_event(
    runtime_track_id: str,
    timestamp: float,
    center_history: list[list[float]],
    point_a: Sequence[float],
    point_b: Sequence[float],
    previous_index: int,
    previous_side: str,
    transition_index: int,
    transition_side: str,
    support_end_index: int,
) -> RuntimeEventCandidate:
    event_type, direction = (
        ("ENTRY", "IN")
        if previous_side == "A" and transition_side == "B"
        else ("EXIT", "OUT")
    )
    supporting_positions = [
        _point(point)
        for point in center_history[previous_index : support_end_index + 1]
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


def _events_for_track(
    track: Any, point_a: Sequence[float], point_b: Sequence[float]
) -> list[RuntimeEventCandidate]:
    runtime_track_id = str(_field(track, "runtime_track_id"))
    timestamp = float(_field(track, "last_seen_timestamp"))
    center_history = [_point(point) for point in _field(track, "center_history")]

    compressed: list[tuple[int, str]] = []
    for original_index, point in enumerate(center_history):
        side = compute_side(point_a, point_b, point)
        if side != "ON":
            compressed.append((original_index, side))

    events: list[RuntimeEventCandidate] = []
    previous_index: int | None = None
    previous_side: str | None = None
    for compressed_index, (transition_index, transition_side) in enumerate(compressed):
        if previous_side is not None and previous_side != transition_side:
            if previous_index is None:
                raise AssertionError("previous transition index missing")
            support_end_index = transition_index
            next_compressed_index = compressed_index + 1
            if next_compressed_index < len(compressed):
                next_index, next_side = compressed[next_compressed_index]
                if next_side == transition_side:
                    support_end_index = next_index

            events.append(
                _make_event(
                    runtime_track_id,
                    timestamp,
                    center_history,
                    point_a,
                    point_b,
                    previous_index,
                    previous_side,
                    transition_index,
                    transition_side,
                    support_end_index,
                )
            )
        previous_index = transition_index
        previous_side = transition_side

    return events


def detect_events(tracks: list[Any], line_config: dict) -> list[RuntimeEventCandidate]:
    point_a, point_b = line_points_from_config(line_config)
    events = []

    for track in tracks:
        events.extend(_events_for_track(track, point_a, point_b))

    return sorted(
        events,
        key=lambda event: (
            event["timestamp"],
            event["runtime_track_id"],
        ),
    )
