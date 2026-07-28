"""Private finite-segment geometry used by Event."""

GEOMETRY_EPSILON = 1e-6

Point = tuple[float, float]


def _signed_area(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _sign(value: float) -> int:
    if value > GEOMETRY_EPSILON:
        return 1
    if value < -GEOMETRY_EPSILON:
        return -1
    return 0


def _side(point: Point, line_a: Point, line_b: Point) -> int:
    return _sign(_signed_area(line_a, line_b, point))


def _point_on_segment(point: Point, segment_a: Point, segment_b: Point) -> bool:
    if _sign(_signed_area(segment_a, segment_b, point)) != 0:
        return False
    return (
        min(segment_a[0], segment_b[0]) - GEOMETRY_EPSILON
        <= point[0]
        <= max(segment_a[0], segment_b[0]) + GEOMETRY_EPSILON
        and min(segment_a[1], segment_b[1]) - GEOMETRY_EPSILON
        <= point[1]
        <= max(segment_a[1], segment_b[1]) + GEOMETRY_EPSILON
    )


def _segments_intersect(
    first_a: Point, first_b: Point, second_a: Point, second_b: Point
) -> bool:
    if (
        abs(first_a[0] - first_b[0]) <= GEOMETRY_EPSILON
        and abs(first_a[1] - first_b[1]) <= GEOMETRY_EPSILON
    ):
        return _point_on_segment(first_a, second_a, second_b)

    first_second_a = _sign(_signed_area(first_a, first_b, second_a))
    first_second_b = _sign(_signed_area(first_a, first_b, second_b))
    second_first_a = _sign(_signed_area(second_a, second_b, first_a))
    second_first_b = _sign(_signed_area(second_a, second_b, first_b))

    if first_second_a * first_second_b < 0 and second_first_a * second_first_b < 0:
        return True

    return (
        (first_second_a == 0 and _point_on_segment(second_a, first_a, first_b))
        or (first_second_b == 0 and _point_on_segment(second_b, first_a, first_b))
        or (second_first_a == 0 and _point_on_segment(first_a, second_a, second_b))
        or (second_first_b == 0 and _point_on_segment(first_b, second_a, second_b))
    )
