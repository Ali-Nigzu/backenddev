"""Private continuous-line geometry used by Event."""

from math import hypot

GEOMETRY_EPSILON = 1e-6

Point = tuple[float, float]


def _signed_distance_to_line(point: Point, line_a: Point, line_b: Point) -> float:
    dx = line_b[0] - line_a[0]
    dy = line_b[1] - line_a[1]
    line_length = hypot(dx, dy)
    return (dx * (point[1] - line_a[1]) - dy * (point[0] - line_a[0])) / line_length


def _side(point: Point, line_a: Point, line_b: Point) -> int:
    signed_distance = _signed_distance_to_line(point, line_a, line_b)
    if signed_distance > GEOMETRY_EPSILON:
        return 1
    if signed_distance < -GEOMETRY_EPSILON:
        return -1
    return 0
