from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LINE_COLOR_BGR = (255, 255, 255)
POINT_A_COLOR_BGR = (0, 0, 255)
POINT_B_COLOR_BGR = (255, 0, 0)
LINE_THICKNESS = 2
POINT_RADIUS = 6

INPUT_PATH = "/workspaces/backenddev/videoplayback.mp4"
OUTPUT_PATH = "/workspaces/backenddev/line_debug.png"

CARD9_SYNTHETIC_LINE_CONFIG = {
    "point_a": [0.0, -10.0],
    "point_b": [0.0, 10.0],
}


def _point_xy(point: Sequence[float], name: str) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    return float(point[0]), float(point[1])


def _load_first_frame(input_path: str) -> np.ndarray:
    path = Path(input_path)
    extension = path.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {input_path}")
        return image

    if extension in VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {input_path}")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Cannot read first frame from video: {input_path}")
            return frame
        finally:
            capture.release()

    raise ValueError(
        "Unsupported input extension. Expected one of: "
        f"{sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)}"
    )


def _line_intersections_with_frame(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    ax, ay = point_a
    bx, by = point_b
    dx = bx - ax
    dy = by - ay

    if dx == 0.0 and dy == 0.0:
        raise ValueError("LineConfig point_a and point_b must define a non-zero line")

    max_x = float(width - 1)
    max_y = float(height - 1)
    intersections: list[tuple[int, int]] = []

    if dx != 0.0:
        for x in (0.0, max_x):
            t = (x - ax) / dx
            y = ay + t * dy
            if 0.0 <= y <= max_y:
                intersections.append((int(round(x)), int(round(y))))

    if dy != 0.0:
        for y in (0.0, max_y):
            t = (y - ay) / dy
            x = ax + t * dx
            if 0.0 <= x <= max_x:
                intersections.append((int(round(x)), int(round(y))))

    deduplicated = []
    for point in intersections:
        if point not in deduplicated:
            deduplicated.append(point)
    return deduplicated


def _extended_line_points(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    intersections = _line_intersections_with_frame(point_a, point_b, width, height)
    if len(intersections) >= 2:
        best_pair = (intersections[0], intersections[1])
        best_distance = -1
        for first_index, first in enumerate(intersections):
            for second in intersections[first_index + 1:]:
                distance = (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (first, second)
        return best_pair

    ax, ay = point_a
    bx, by = point_b
    dx = bx - ax
    dy = by - ay
    scale = float(max(width, height) * 4)
    return (
        (int(round(ax - dx * scale)), int(round(ay - dy * scale))),
        (int(round(ax + dx * scale)), int(round(ay + dy * scale))),
    )


def _draw_point(frame: np.ndarray, point: tuple[float, float], color: tuple[int, int, int]) -> None:
    cv2.circle(
        frame,
        (int(round(point[0])), int(round(point[1]))),
        POINT_RADIUS,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def render_line_overlay(
    input_path: str,
    line_config: dict,
    output_path: str | None = None,
) -> np.ndarray:
    frame = _load_first_frame(input_path).copy()
    height, width = frame.shape[:2]

    point_a = _point_xy(line_config["point_a"], "point_a")
    point_b = _point_xy(line_config["point_b"], "point_b")
    line_start, line_end = _extended_line_points(point_a, point_b, width, height)

    cv2.line(
        frame,
        line_start,
        line_end,
        LINE_COLOR_BGR,
        thickness=LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )
    _draw_point(frame, point_a, POINT_A_COLOR_BGR)
    _draw_point(frame, point_b, POINT_B_COLOR_BGR)

    if output_path is not None:
        if not cv2.imwrite(str(output_path), frame):
            raise IOError(f"Failed to write line overlay image: {output_path}")

    return frame


if __name__ == "__main__":
    frame = render_line_overlay(
        INPUT_PATH,
        CARD9_SYNTHETIC_LINE_CONFIG,
        OUTPUT_PATH,
    )
    print(f"saved -> {OUTPUT_PATH}")
