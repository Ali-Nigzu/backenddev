"""Pure deterministic Track V2 motion maths.

Motion owns prediction, velocity, uncertainty, speed, eligibility, distance, and
rejection reasons. It does not know about assignment takeover rules or
appearance. The estimator is an alpha-beta style filter reconstructed from the
public path history for each assessment; no hidden state is persisted.
"""

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
    allowed_error: float


@dataclass(frozen=True)
class MotionEstimate:
    position: dict
    velocity: dict


def _xy(center) -> Tuple[float, float]:
    return float(center["x"]), float(center["y"])


def _point(x: float, y: float) -> dict:
    return {"x": float(x), "y": float(y)}


def distance(a, b) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def _clamp_velocity(vx: float, vy: float, policy: TrackerPolicy) -> tuple[float, float]:
    speed = math.sqrt(vx * vx + vy * vy)
    if speed > policy.max_speed_px_per_sec and speed > policy.epsilon:
        scale = policy.max_speed_px_per_sec / speed
        return vx * scale, vy * scale
    return vx, vy


def estimate_motion(path, policy: TrackerPolicy) -> MotionEstimate:
    """Reconstruct a smoothed deterministic alpha-beta estimate from path."""

    if not path:
        raise ValueError("motion estimation requires a non-empty path")

    start = max(0, len(path) - int(policy.max_history_points))
    first = path[start]
    x, y = _xy(first["center"])
    vx = 0.0
    vy = 0.0
    previous_timestamp = float(first["timestamp"])

    for point in path[start + 1 :]:
        timestamp = float(point["timestamp"])
        dt = timestamp - previous_timestamp
        observed_x, observed_y = _xy(point["center"])
        if dt <= policy.epsilon:
            # Same-frame duplicate path points are treated as measurement updates
            # without velocity acceleration.
            residual_x = observed_x - x
            residual_y = observed_y - y
            x += policy.alpha_beta_position_gain * residual_x
            y += policy.alpha_beta_position_gain * residual_y
            previous_timestamp = timestamp
            continue

        predicted_x = x + vx * dt
        predicted_y = y + vy * dt
        residual_x = observed_x - predicted_x
        residual_y = observed_y - predicted_y

        x = predicted_x + policy.alpha_beta_position_gain * residual_x
        y = predicted_y + policy.alpha_beta_position_gain * residual_y
        vx += policy.alpha_beta_velocity_gain * residual_x / dt
        vy += policy.alpha_beta_velocity_gain * residual_y / dt
        vx, vy = _clamp_velocity(vx, vy, policy)
        previous_timestamp = timestamp

    return MotionEstimate(position=_point(x, y), velocity=_point(vx, vy))


def derive_velocity(path, policy: TrackerPolicy) -> dict:
    """Compatibility helper returning the reconstructed velocity only."""

    return estimate_motion(path, policy).velocity


def predict_center(track, timestamp: float, policy: TrackerPolicy, velocity: dict | None = None) -> dict:
    latest_timestamp = float(track["path"][-1]["timestamp"])
    estimate = estimate_motion(track["path"], policy)
    dt = float(timestamp) - latest_timestamp
    if dt <= policy.epsilon:
        return dict(estimate.position)

    chosen_velocity = velocity if velocity is not None else estimate.velocity
    return _point(
        float(estimate.position["x"]) + float(chosen_velocity["x"]) * dt,
        float(estimate.position["y"]) + float(chosen_velocity["y"]) * dt,
    )


def _allowed_error(status: TrackStatus, policy: TrackerPolicy) -> float:
    return max(
        policy.epsilon,
        policy.base_position_uncertainty_px
        + policy.localization_jitter_px
        + policy.miss_uncertainty_growth_px_per_sec * status.missing_seconds,
    )


def _ineligible(
    reason: str,
    score: float,
    predicted_position: dict,
    velocity: dict,
    distance_prediction: float = float("inf"),
    distance_latest: float = float("inf"),
    speed_required: float = float("inf"),
    allowed_error: float = 0.0,
) -> MotionAssessment:
    return MotionAssessment(
        False,
        score,
        distance_prediction,
        distance_latest,
        speed_required,
        predicted_position,
        velocity,
        reason,
        allowed_error,
    )


def assess_motion(
    track: dict,
    status: TrackStatus,
    observation: dict,
    timestamp: float,
    policy: TrackerPolicy,
) -> MotionAssessment:
    estimate = estimate_motion(track["path"], policy)
    latest_timestamp = float(track["path"][-1]["timestamp"])
    dt = float(timestamp) - latest_timestamp
    predicted_position = predict_center(track, timestamp, policy, estimate.velocity)
    allowed_error = _allowed_error(status, policy)

    if status.age_seconds < -policy.epsilon or dt < -policy.epsilon:
        return _ineligible("negative_time_gap", float("inf"), predicted_position, estimate.velocity, allowed_error=allowed_error)
    if not status.eligible:
        return _ineligible("outside_track_window", float("inf"), predicted_position, estimate.velocity, allowed_error=allowed_error)

    distance_prediction = distance(predicted_position, observation["center"])
    distance_latest = distance(status.latest_position, observation["center"])

    if dt <= policy.epsilon:
        speed_required = 0.0 if distance_latest <= policy.localization_jitter_px else float("inf")
        if speed_required == float("inf"):
            return _ineligible(
                "same_timestamp_position_change",
                float("inf"),
                predicted_position,
                estimate.velocity,
                distance_prediction,
                distance_latest,
                speed_required,
                allowed_error,
            )
    else:
        speed_required = distance_latest / dt
        if speed_required > policy.max_speed_px_per_sec:
            return _ineligible(
                "speed_limit",
                float("inf"),
                predicted_position,
                estimate.velocity,
                distance_prediction,
                distance_latest,
                speed_required,
                allowed_error,
            )

    motion_score = distance_prediction / allowed_error
    weak_limit = policy.weak_confirmed_max_motion_score if status.confirmed else 1.0
    if motion_score > weak_limit:
        return _ineligible(
            "prediction_gate",
            motion_score,
            predicted_position,
            estimate.velocity,
            distance_prediction,
            distance_latest,
            speed_required,
            allowed_error,
        )

    return MotionAssessment(
        True,
        motion_score,
        distance_prediction,
        distance_latest,
        speed_required,
        predicted_position,
        estimate.velocity,
        "eligible",
        allowed_error,
    )


# Backwards-compatible name for callers that still import the old helper.
def evaluate_motion(track_facts, observation: dict, config: TrackerPolicy) -> MotionAssessment:
    raise RuntimeError("evaluate_motion requires the track and precomputed TrackStatus; use assess_motion")
