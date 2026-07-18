"""Pure deterministic Track V2 motion maths."""

import math
from dataclasses import dataclass
from typing import Tuple

from track.lifecycle import TrackStatus
from track.policy import TrackerPolicy


@dataclass(frozen=True)
class MotionAssessment:
    eligible: bool
    motion_score: float
    distance_prediction: float
    distance_latest: float
    speed_required: float
    predicted_position: dict
    velocity: dict
    rejection_reason: str


def _xy(center) -> Tuple[float, float]:
    return float(center["x"]), float(center["y"])


def distance(a, b) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def derive_velocity(path, policy: TrackerPolicy) -> dict:
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
        if dt <= policy.epsilon:
            previous = latest
            continue

        prev_x, prev_y = _xy(previous["center"])
        latest_x, latest_y = _xy(latest["center"])
        vx = (latest_x - prev_x) / dt
        vy = (latest_y - prev_y) / dt
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > policy.max_physical_speed_px_per_sec and speed > policy.epsilon:
            scale = policy.max_physical_speed_px_per_sec / speed
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


def predict_center(track, timestamp: float, policy: TrackerPolicy, velocity: dict | None = None) -> dict:
    latest = track["path"][-1]
    latest_x, latest_y = _xy(latest["center"])
    dt = float(timestamp) - float(latest["timestamp"])
    if dt <= policy.epsilon:
        return {"x": latest_x, "y": latest_y}

    velocity = velocity if velocity is not None else derive_velocity(track["path"], policy)
    return {
        "x": latest_x + float(velocity["x"]) * dt,
        "y": latest_y + float(velocity["y"]) * dt,
    }


def assess_motion(
    track: dict,
    status: TrackStatus,
    observation: dict,
    timestamp: float,
    policy: TrackerPolicy,
) -> MotionAssessment:
    velocity = derive_velocity(track["path"], policy)
    predicted_position = predict_center(track, timestamp, policy, velocity)

    if status.age_seconds < -policy.epsilon:
        return MotionAssessment(False, float("inf"), float("inf"), float("inf"), float("inf"), predicted_position, velocity, "negative_time_gap")
    if not status.eligible:
        return MotionAssessment(False, float("inf"), float("inf"), float("inf"), float("inf"), predicted_position, velocity, "outside_track_window")

    distance_prediction = distance(predicted_position, observation["center"])
    distance_latest = distance(status.latest_position, observation["center"])
    safe_gap = max(status.age_seconds, policy.epsilon)
    speed_required = distance_latest / safe_gap

    if speed_required > policy.max_physical_speed_px_per_sec:
        return MotionAssessment(
            False,
            float("inf"),
            distance_prediction,
            distance_latest,
            speed_required,
            predicted_position,
            velocity,
            "speed_limit",
        )

    tolerance = (
        policy.motion_tolerance_px
        + policy.localization_jitter_px
        + policy.motion_tolerance_growth_px_per_sec * status.missing_seconds
    )
    normalized_prediction = distance_prediction / max(tolerance, policy.epsilon)
    normalized_latest = distance_latest / max(tolerance, policy.epsilon)
    normalized_speed = speed_required / max(policy.max_physical_speed_px_per_sec, policy.epsilon)
    motion_score = max(min(normalized_prediction, normalized_latest), normalized_speed)

    if motion_score > 1.0 and not (
        status.confirmed
        and policy.allow_weak_confirmed_matching
        and motion_score <= policy.weak_match_max_motion_score
    ):
        return MotionAssessment(
            False,
            motion_score,
            distance_prediction,
            distance_latest,
            speed_required,
            predicted_position,
            velocity,
            "motion_tolerance",
        )

    return MotionAssessment(True, motion_score, distance_prediction, distance_latest, speed_required, predicted_position, velocity, "eligible")


# Backwards-compatible name for callers that still import the old helper.
def evaluate_motion(track_facts, observation: dict, config: TrackerPolicy) -> MotionAssessment:
    raise RuntimeError("evaluate_motion requires the track and precomputed TrackStatus; use assess_motion")
