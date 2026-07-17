"""Pure deterministic Track V2 motion maths."""

import math
from dataclasses import dataclass
from typing import Tuple

from track.normalize import _NormalizedTrackConfig


@dataclass(frozen=True)
class MotionAssessment:
    eligible: bool
    motion_score: float
    distance_prediction: float
    distance_latest: float
    speed_required: float
    rejection_reason: str


def _xy(center) -> Tuple[float, float]:
    return float(center["x"]), float(center["y"])


def distance(a, b) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def derive_velocity(path, config: _NormalizedTrackConfig) -> dict:
    if len(path) < 2:
        return {"x": 0.0, "y": 0.0}

    weighted_x = 0.0
    weighted_y = 0.0
    total_weight = 0.0
    segment_count = 0
    recent_start = max(0, len(path) - 5)
    previous = path[recent_start]
    for latest_index in range(recent_start + 1, len(path)):
        latest = path[latest_index]
        dt = float(latest["timestamp"]) - float(previous["timestamp"])
        if dt <= config.epsilon:
            previous = latest
            continue

        prev_x, prev_y = _xy(previous["center"])
        latest_x, latest_y = _xy(latest["center"])
        vx = (latest_x - prev_x) / dt
        vy = (latest_y - prev_y) / dt
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > config.max_physical_speed_px_per_sec and speed > config.epsilon:
            scale = config.max_physical_speed_px_per_sec / speed
            vx *= scale
            vy *= scale

        segment_count += 1
        weight = float(segment_count)
        weighted_x += vx * weight
        weighted_y += vy * weight
        total_weight += weight
        previous = latest

    if segment_count == 0:
        return {"x": 0.0, "y": 0.0}

    return {"x": weighted_x / total_weight, "y": weighted_y / total_weight}


def predict_center(track, timestamp: float, config: _NormalizedTrackConfig) -> dict:
    latest = track["path"][-1]
    latest_x, latest_y = _xy(latest["center"])
    dt = float(timestamp) - float(latest["timestamp"])
    if dt <= config.epsilon:
        return {"x": latest_x, "y": latest_y}

    velocity = derive_velocity(track["path"], config)
    return {
        "x": latest_x + float(velocity["x"]) * dt,
        "y": latest_y + float(velocity["y"]) * dt,
    }


def evaluate_motion(track_facts, observation: dict, config: _NormalizedTrackConfig) -> MotionAssessment:
    if track_facts.age_seconds < -config.epsilon:
        return MotionAssessment(False, float("inf"), float("inf"), float("inf"), float("inf"), "negative_time_gap")
    if not track_facts.eligible:
        return MotionAssessment(False, float("inf"), float("inf"), float("inf"), float("inf"), "outside_track_window")

    distance_prediction = distance(track_facts.predicted_position, observation["center"])
    distance_latest = distance(track_facts.latest_position, observation["center"])
    safe_gap = max(track_facts.age_seconds, config.epsilon)
    speed_required = distance_latest / safe_gap

    if speed_required > config.max_physical_speed_px_per_sec:
        return MotionAssessment(
            False,
            float("inf"),
            distance_prediction,
            distance_latest,
            speed_required,
            "speed_limit",
        )

    tolerance = (
        config.motion_tolerance_px
        + config.localization_jitter_px
        + config.motion_tolerance_growth_px_per_sec * track_facts.missing_seconds
    )
    normalized_prediction = distance_prediction / max(tolerance, config.epsilon)
    normalized_latest = distance_latest / max(tolerance, config.epsilon)
    normalized_speed = speed_required / max(config.max_physical_speed_px_per_sec, config.epsilon)
    motion_score = max(min(normalized_prediction, normalized_latest), normalized_speed)

    weak_limit = 1.0 + config.takeover_margin
    if motion_score > 1.0 and not (
        track_facts.confirmed and config.allow_weak_confirmed_matching and motion_score <= weak_limit
    ):
        return MotionAssessment(
            False,
            motion_score,
            distance_prediction,
            distance_latest,
            speed_required,
            "motion_tolerance",
        )

    return MotionAssessment(True, motion_score, distance_prediction, distance_latest, speed_required, "eligible")
