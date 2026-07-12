"""Pure motion helpers derived exclusively from Track.path."""

import math
from typing import Tuple

from track.config import TrackV2Config


def _xy(center) -> Tuple[float, float]:
    return float(center["x"]), float(center["y"])


def distance(a, b) -> float:
    ax, ay = _xy(a)
    bx, by = _xy(b)
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def derive_velocity(path, config: TrackV2Config) -> dict:
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
        if speed > config.max_speed_px_per_sec and speed > config.epsilon:
            scale = config.max_speed_px_per_sec / speed
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

    return {
        "x": weighted_x / total_weight,
        "y": weighted_y / total_weight,
    }


def predict_center(track, timestamp: float, config: TrackV2Config) -> dict:
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


def motion_gate(track, observation, timestamp: float, config: TrackV2Config):
    latest_timestamp = float(track["path"][-1]["timestamp"])
    dt = float(timestamp) - latest_timestamp
    if dt < -config.epsilon:
        return False, float("inf"), float("inf"), float("inf")
    if dt > config.max_reassociation_gap_sec:
        return False, float("inf"), float("inf"), dt

    safe_dt = max(dt, 0.0)
    predicted = predict_center(track, timestamp, config)
    predicted_distance = distance(predicted, observation["center"])
    latest_distance = distance(track["path"][-1]["center"], observation["center"])
    jitter_allowance = config.base_motion_gate_px
    motion_distance = min(predicted_distance, latest_distance + jitter_allowance)
    allowed_distance = config.base_motion_gate_px + config.max_speed_px_per_sec * safe_dt
    if allowed_distance <= config.epsilon:
        return False, motion_distance, float("inf"), dt

    normalized_motion = motion_distance / allowed_distance
    if motion_distance > allowed_distance:
        return False, motion_distance, normalized_motion, dt

    return True, motion_distance, normalized_motion, dt
