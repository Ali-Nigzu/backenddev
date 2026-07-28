"""Continuous directed-line overlay for replay video frames."""

from collections.abc import Mapping
from math import isfinite

import cv2
import numpy as np

LINE_COLOR_BGR = (255, 255, 255)
POINT_A_COLOR_BGR = (0, 0, 255)
POINT_B_COLOR_BGR = (255, 0, 0)
LABEL_COLOR_BGR = (255, 255, 255)
LINE_THICKNESS = 2
POINT_RADIUS = 6


def _point_xy(point, name: str) -> tuple[float, float]:
    if not isinstance(point, Mapping):
        raise ValueError(f"{name} must be an object")
    for field in ("x", "y"):
        if field not in point:
            raise ValueError(f"Missing required {name} field: {field}")
    if isinstance(point["x"], bool) or isinstance(point["y"], bool):
        raise ValueError(f"{name} coordinates must be finite")
    x = float(point["x"])
    y = float(point["y"])
    if not isfinite(x) or not isfinite(y):
        raise ValueError(f"{name} coordinates must be finite")
    return x, y


def _draw_point(
    frame: np.ndarray, point: tuple[float, float], color: tuple[int, int, int]
) -> None:
    cv2.circle(
        frame,
        (int(round(point[0])), int(round(point[1]))),
        POINT_RADIUS,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def _clipped_line_points(
    point_a: tuple[float, float], point_b: tuple[float, float], width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    ax, ay = point_a
    bx, by = point_b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        raise ValueError("point_a and point_b must define a non-zero line")

    candidates: list[tuple[float, float]] = []
    max_x = float(width - 1)
    max_y = float(height - 1)

    if dx != 0:
        for x in (0.0, max_x):
            t = (x - ax) / dx
            y = ay + t * dy
            if 0.0 <= y <= max_y:
                candidates.append((x, y))

    if dy != 0:
        for y in (0.0, max_y):
            t = (y - ay) / dy
            x = ax + t * dx
            if 0.0 <= x <= max_x:
                candidates.append((x, y))

    unique: list[tuple[float, float]] = []
    for candidate in candidates:
        rounded = (round(candidate[0], 6), round(candidate[1], 6))
        if all(
            rounded != (round(point[0], 6), round(point[1], 6))
            for point in unique
        ):
            unique.append(candidate)

    if len(unique) < 2:
        return None

    return (
        (int(round(unique[0][0])), int(round(unique[0][1]))),
        (int(round(unique[1][0])), int(round(unique[1][1]))),
    )


def draw_line_overlay(frame: np.ndarray, line_config: dict) -> np.ndarray:
    point_a = _point_xy(line_config["point_a"], "point_a")
    point_b = _point_xy(line_config["point_b"], "point_b")
    height, width = frame.shape[:2]
    clipped_points = _clipped_line_points(point_a, point_b, width, height)
    if clipped_points is not None:
        cv2.line(
            frame,
            clipped_points[0],
            clipped_points[1],
            LINE_COLOR_BGR,
            thickness=LINE_THICKNESS,
            lineType=cv2.LINE_AA,
        )
    _draw_point(frame, point_a, POINT_A_COLOR_BGR)
    _draw_point(frame, point_b, POINT_B_COLOR_BGR)
    cv2.putText(
        frame,
        "A",
        (int(round(point_a[0])) + 8, int(round(point_a[1])) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        LABEL_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "B",
        (int(round(point_b[0])) + 8, int(round(point_b[1])) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        LABEL_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )
    return frame
