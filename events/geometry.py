from typing import Sequence

LINE_SIDE_EPSILON = 1e-9


def _as_xy(point: Sequence[float], name: str) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    return float(point[0]), float(point[1])


def signed_cross(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> float:
    ax, ay = _as_xy(a, "point_a")
    bx, by = _as_xy(b, "point_b")
    px, py = _as_xy(p, "point")

    if ax == bx and ay == by:
        raise ValueError("LineConfig point_a and point_b must define a non-zero line")

    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def side_from_cross(cross: float) -> str:
    if cross < -LINE_SIDE_EPSILON:
        return "A"
    if cross > LINE_SIDE_EPSILON:
        return "B"
    return "ON"


def compute_side(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> str:
    return side_from_cross(signed_cross(a, b, p))


def line_points_from_config(line_config: dict) -> tuple[list[float], list[float]]:
    point_a = list(_as_xy(line_config["point_a"], "point_a"))
    point_b = list(_as_xy(line_config["point_b"], "point_b"))
    if point_a == point_b:
        raise ValueError("LineConfig point_a and point_b must define a non-zero line")
    return point_a, point_b
