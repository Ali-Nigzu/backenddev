"""Stateless, deterministic DetectionBatch -> TrackBatch tracking stage."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import (
    ACTIVATION_HITS,
    ACTIVE_TIMEOUT_SECONDS,
    BOOTSTRAP_ACTIVATION_HITS,
    BOOTSTRAP_WINDOW_FRAMES,
    CENTRE_PROCESS_NOISE,
    HIGH_CONFIDENCE_THRESHOLD,
    INITIAL_VELOCITY_VARIANCE,
    IOU_COST_WEIGHT,
    LOW_CONFIDENCE_THRESHOLD,
    MAHALANOBIS_GATE,
    MEASUREMENT_CENTRE_NOISE,
    MEASUREMENT_SIZE_NOISE,
    MOTION_COST_WEIGHT,
    PRIMARY_MAX_COST,
    RECOVERY_MAX_COST,
    SIZE_PROCESS_NOISE,
    TENTATIVE_TIMEOUT_SECONDS,
)

_TENTATIVE = "tentative"
_ACTIVE = "active"
_CLOSED = "closed"
_MIN_BOX_SIZE = 1e-3
_IMPOSSIBLE_COST = 1e6


def _measurement(detection: dict) -> np.ndarray:
    bbox = detection["bbox"]
    return np.array(
        [
            float(detection["centre"]["x"]),
            float(detection["centre"]["y"]),
            max(float(bbox["x2"]) - float(bbox["x1"]), _MIN_BOX_SIZE),
            max(float(bbox["y2"]) - float(bbox["y1"]), _MIN_BOX_SIZE),
        ],
        dtype=np.float64,
    )


def _measurement_covariance(measurement: np.ndarray) -> np.ndarray:
    width = max(float(measurement[2]), _MIN_BOX_SIZE)
    height = max(float(measurement[3]), _MIN_BOX_SIZE)
    standard_deviations = np.array(
        [
            max(width * MEASUREMENT_CENTRE_NOISE, 1.0),
            max(height * MEASUREMENT_CENTRE_NOISE, 1.0),
            max(width * MEASUREMENT_SIZE_NOISE, 1.0),
            max(height * MEASUREMENT_SIZE_NOISE, 1.0),
        ],
        dtype=np.float64,
    )
    return np.diag(standard_deviations**2)


def _new_kalman_state(detection: dict) -> tuple[np.ndarray, np.ndarray]:
    measurement = _measurement(detection)
    state = np.zeros(8, dtype=np.float64)
    state[:4] = measurement
    covariance = np.zeros((8, 8), dtype=np.float64)
    covariance[:4, :4] = _measurement_covariance(measurement)
    covariance[4:, 4:] = np.eye(4, dtype=np.float64) * INITIAL_VELOCITY_VARIANCE
    return state, covariance


def _predict(track: dict, timestamp: float) -> None:
    dt = timestamp - track["last_prediction_timestamp"]
    transition = np.eye(8, dtype=np.float64)
    transition[:4, 4:] = np.eye(4, dtype=np.float64) * dt

    process_covariance = np.zeros((8, 8), dtype=np.float64)
    for index, noise in enumerate(
        (CENTRE_PROCESS_NOISE, CENTRE_PROCESS_NOISE, SIZE_PROCESS_NOISE, SIZE_PROCESS_NOISE)
    ):
        variance = noise**2
        process_covariance[index, index] = variance * dt**4 / 4.0
        process_covariance[index, index + 4] = variance * dt**3 / 2.0
        process_covariance[index + 4, index] = variance * dt**3 / 2.0
        process_covariance[index + 4, index + 4] = variance * dt**2

    track["state"] = transition @ track["state"]
    track["state"][2:4] = np.maximum(track["state"][2:4], _MIN_BOX_SIZE)
    track["covariance"] = (
        transition @ track["covariance"] @ transition.T + process_covariance
    )
    track["covariance"] = (track["covariance"] + track["covariance"].T) / 2.0
    track["last_prediction_timestamp"] = timestamp


def _innovation(track: dict, detection: dict) -> tuple[np.ndarray, np.ndarray, float]:
    measurement = _measurement(detection)
    residual = measurement - track["state"][:4]
    innovation_covariance = (
        track["covariance"][:4, :4] + _measurement_covariance(measurement)
    )
    solved = np.linalg.solve(innovation_covariance, residual)
    distance = float(residual @ solved)
    return residual, innovation_covariance, max(distance, 0.0)


def _predicted_bbox(track: dict) -> tuple[float, float, float, float]:
    centre_x, centre_y, width, height = track["state"][:4]
    width = max(float(width), _MIN_BOX_SIZE)
    height = max(float(height), _MIN_BOX_SIZE)
    return (
        float(centre_x) - width / 2.0,
        float(centre_y) - height / 2.0,
        float(centre_x) + width / 2.0,
        float(centre_y) + height / 2.0,
    )


def _iou(track: dict, detection: dict) -> float:
    left_a, top_a, right_a, bottom_a = _predicted_bbox(track)
    bbox = detection["bbox"]
    left_b, top_b, right_b, bottom_b = (
        float(bbox[key]) for key in ("x1", "y1", "x2", "y2")
    )
    intersection_width = max(0.0, min(right_a, right_b) - max(left_a, left_b))
    intersection_height = max(0.0, min(bottom_a, bottom_b) - max(top_a, top_b))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, right_a - left_a) * max(0.0, bottom_a - top_a)
    area_b = max(0.0, right_b - left_b) * max(0.0, bottom_b - top_b)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _associate(
    tracks: list[dict], detections: list[dict], maximum_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    costs = np.full((len(tracks), len(detections)), _IMPOSSIBLE_COST, dtype=np.float64)
    allowed = np.zeros(costs.shape, dtype=bool)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            _residual, _innovation_covariance, distance = _innovation(track, detection)
            if not math.isfinite(distance) or distance > MAHALANOBIS_GATE:
                continue
            motion_cost = min(distance / MAHALANOBIS_GATE, 1.0)
            iou_cost = 1.0 - _iou(track, detection)
            costs[track_index, detection_index] = (
                MOTION_COST_WEIGHT * motion_cost + IOU_COST_WEIGHT * iou_cost
            )
            allowed[track_index, detection_index] = True

    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    matches = []
    row_indices, column_indices = linear_sum_assignment(costs)
    for track_index, detection_index in zip(row_indices, column_indices, strict=True):
        if allowed[track_index, detection_index] and costs[track_index, detection_index] <= maximum_cost:
            matches.append((int(track_index), int(detection_index)))
            matched_tracks.add(int(track_index))
            matched_detections.add(int(detection_index))

    return (
        matches,
        [index for index in range(len(tracks)) if index not in matched_tracks],
        [index for index in range(len(detections)) if index not in matched_detections],
    )


def _update(track: dict, detection: dict, frame_id: str, timestamp: float) -> None:
    measurement = _measurement(detection)
    residual, innovation_covariance, _distance = _innovation(track, detection)
    cross_covariance = track["covariance"][:, :4]
    kalman_gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
    track["state"] = track["state"] + kalman_gain @ residual
    track["state"][2:4] = np.maximum(track["state"][2:4], _MIN_BOX_SIZE)

    identity = np.eye(8, dtype=np.float64)
    measurement_matrix = np.zeros((4, 8), dtype=np.float64)
    measurement_matrix[:, :4] = np.eye(4, dtype=np.float64)
    remainder = identity - kalman_gain @ measurement_matrix
    measurement_covariance = _measurement_covariance(measurement)
    track["covariance"] = (
        remainder @ track["covariance"] @ remainder.T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )
    track["covariance"] = (track["covariance"] + track["covariance"].T) / 2.0
    track["last_observed_timestamp"] = timestamp
    track["hits"] += 1
    track["path"].append(
        {"timestamp": timestamp, "centre": dict(detection["centre"])}
    )
    confidence = detection["confidence"]
    if confidence > track["best_crop_confidence"]:
        track["best_crop"] = {"frame_id": frame_id, "bbox": dict(detection["bbox"])}
        track["best_crop_confidence"] = confidence
    if track["status"] == _TENTATIVE and track["hits"] >= track["required_hits"]:
        track["status"] = _ACTIVE
        track["reached_active"] = True


def _create_track(
    detection: dict, frame_id: str, timestamp: float, frame_index: int, track_id: int
) -> dict:
    state, covariance = _new_kalman_state(detection)
    required_hits = (
        BOOTSTRAP_ACTIVATION_HITS
        if frame_index < BOOTSTRAP_WINDOW_FRAMES
        else ACTIVATION_HITS
    )
    reached_active = required_hits <= 1
    return {
        "track_id": str(track_id),
        "path": [{"timestamp": timestamp, "centre": dict(detection["centre"])}],
        "best_crop": {"frame_id": frame_id, "bbox": dict(detection["bbox"])},
        "best_crop_confidence": detection["confidence"],
        "state": state,
        "covariance": covariance,
        "status": _ACTIVE if reached_active else _TENTATIVE,
        "hits": 1,
        "last_observed_timestamp": timestamp,
        "last_prediction_timestamp": timestamp,
        "required_hits": required_hits,
        "reached_active": reached_active,
    }


def _expire(track: dict, timestamp: float) -> None:
    elapsed = timestamp - track["last_observed_timestamp"]
    timeout = (
        TENTATIVE_TIMEOUT_SECONDS if track["status"] == _TENTATIVE else ACTIVE_TIMEOUT_SECONDS
    )
    if elapsed > timeout:
        track["status"] = _CLOSED


def _public_track(track: dict) -> dict:
    return {
        "track_id": track["track_id"],
        "path": track["path"],
        "best_crop": track["best_crop"],
        "best_crop_confidence": track["best_crop_confidence"],
    }


class Track:
    """Process one complete DetectionBatch without retaining cross-call state."""

    __slots__ = ()

    def __call__(self, detection_batch: dict) -> dict:
        indexed_frames = []
        for original_index, frame in enumerate(detection_batch["detections"]):
            timestamp = float(frame["timestamp"])
            if not math.isfinite(timestamp):
                raise ValueError("Frame timestamps must be finite")
            indexed_frames.append((timestamp, str(frame["frame_id"]), original_index, frame))
        indexed_frames.sort(key=lambda item: item[:3])

        tracks: list[dict] = []
        next_track_id = 1
        for frame_index, (timestamp, _frame_key, _original_index, frame) in enumerate(indexed_frames):
            frame_id = frame["frame_id"]
            detections = [
                detection
                for _index, detection in sorted(
                    enumerate(frame["detections"]),
                    key=lambda item: (str(item[1]["detection_id"]), item[0]),
                )
            ]
            high_detections = [
                detection
                for detection in detections
                if detection["confidence"] >= HIGH_CONFIDENCE_THRESHOLD
            ]
            low_detections = [
                detection
                for detection in detections
                if LOW_CONFIDENCE_THRESHOLD
                <= detection["confidence"]
                < HIGH_CONFIDENCE_THRESHOLD
            ]

            eligible_tracks = [track for track in tracks if track["status"] != _CLOSED]
            for track in eligible_tracks:
                _predict(track, timestamp)
                _expire(track, timestamp)
            eligible_tracks = [track for track in eligible_tracks if track["status"] != _CLOSED]

            primary_matches, unmatched_track_indices, unmatched_high_indices = _associate(
                eligible_tracks, high_detections, PRIMARY_MAX_COST
            )
            for track_index, detection_index in primary_matches:
                _update(
                    eligible_tracks[track_index], high_detections[detection_index], frame_id, timestamp
                )

            recovery_tracks = [
                eligible_tracks[index]
                for index in unmatched_track_indices
                if eligible_tracks[index]["status"] == _ACTIVE
            ]
            recovery_matches, _unmatched_recovery, _unmatched_low = _associate(
                recovery_tracks, low_detections, RECOVERY_MAX_COST
            )
            for track_index, detection_index in recovery_matches:
                _update(
                    recovery_tracks[track_index], low_detections[detection_index], frame_id, timestamp
                )

            for detection_index in unmatched_high_indices:
                tracks.append(
                    _create_track(
                        high_detections[detection_index],
                        frame_id,
                        timestamp,
                        frame_index,
                        next_track_id,
                    )
                )
                next_track_id += 1

        return {
            "tracks": [
                _public_track(track)
                for track in tracks
                if track["reached_active"]
            ]
        }
