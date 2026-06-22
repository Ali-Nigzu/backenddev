import math
from typing import Sequence, List


def predict_center(center: Sequence[float], velocity: Sequence[float], dt: float) -> List[float]:
    return [
        float(center[0]) + float(velocity[0]) * float(dt),
        float(center[1]) + float(velocity[1]) * float(dt),
    ]


def compute_velocity(prev_center: Sequence[float], new_center: Sequence[float], dt: float) -> List[float]:
    if dt <= 1e-9:
        return [0.0, 0.0]

    return [
        (float(new_center[0]) - float(prev_center[0])) / float(dt),
        (float(new_center[1]) - float(prev_center[1])) / float(dt),
    ]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)
