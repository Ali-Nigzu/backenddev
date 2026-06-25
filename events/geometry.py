from typing import Sequence


LINE_SIDE_EPSILON = 1e-9


def _as_xy(point: Sequence[float], name: str) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    return float(point[0]), float(point[1])


def compute_side(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> str:
    ax, ay = _as_xy(a, "point_a")
    bx, by = _as_xy(b, "point_b")
    px, py = _as_xy(p, "point")

    if ax == bx and ay == by:
        raise ValueError("LineConfig point_a and point_b must define a non-zero line")

    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if cross < -LINE_SIDE_EPSILON:
        return "A"
    if cross > LINE_SIDE_EPSILON:
        return "B"
    return "ON"
