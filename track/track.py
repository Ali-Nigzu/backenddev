"""Stateless, deterministic DetectionBatch -> TrackBatch tracking stage."""

import bisect
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
    REFINE_ENABLED,
    REFINE_MAX_COST,
    REFINE_MAX_GAP_SECONDS,
    REFINE_MAX_ROUNDS,
    SIZE_PROCESS_NOISE,
    TENTATIVE_TIMEOUT_SECONDS,
)

_TENTATIVE = "tentative"
_ACTIVE = "active"
_CLOSED = "closed"
_MIN_BOX_SIZE = 1e-3
_IMPOSSIBLE_COST = 1e6
_ENDPOINT_OBSERVATIONS = 4
_REFINE_POSITION_WEIGHT = 0.55
_REFINE_DIRECTION_WEIGHT = 0.15
_REFINE_SPEED_WEIGHT = 0.10
_REFINE_BOX_WEIGHT = 0.10
_REFINE_GAP_WEIGHT = 0.10
_REFINE_POSITION_GATE = 13.28
_REFINE_MIN_BOX_RATIO = 0.4
_REFINE_MAX_BOX_RATIO = 2.5
_REFINE_SOFT_BOX_RATIO = 1.75
_REFINE_TIE_EPSILON = 1e-12


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


def _observation(detection: dict, timestamp: float) -> dict:
    return {
        "timestamp": timestamp,
        "centre": dict(detection["centre"]),
        "bbox": dict(detection["bbox"]),
        "confidence": detection["confidence"],
    }


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


def _transition_and_process_covariance(dt: float) -> tuple[np.ndarray, np.ndarray]:
    transition = np.eye(8, dtype=np.float64)
    transition[:4, 4:] = np.eye(4, dtype=np.float64) * dt
    process_covariance = np.zeros((8, 8), dtype=np.float64)
    for index, noise in enumerate(
        (
            CENTRE_PROCESS_NOISE,
            CENTRE_PROCESS_NOISE,
            SIZE_PROCESS_NOISE,
            SIZE_PROCESS_NOISE,
        )
    ):
        variance = noise**2
        process_covariance[index, index] = variance * dt**4 / 4.0
        process_covariance[index, index + 4] = variance * dt**3 / 2.0
        process_covariance[index + 4, index] = variance * dt**3 / 2.0
        process_covariance[index + 4, index + 4] = variance * dt**2
    return transition, process_covariance


def _predict(track: dict, timestamp: float) -> None:
    dt = timestamp - track["last_prediction_timestamp"]
    transition, process_covariance = _transition_and_process_covariance(dt)

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
    innovation_covariance = track["covariance"][:4, :4] + _measurement_covariance(
        measurement
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
        if (
            allowed[track_index, detection_index]
            and costs[track_index, detection_index] <= maximum_cost
        ):
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
    track["last_observed_state"] = track["state"].copy()
    track["last_observed_covariance"] = track["covariance"].copy()
    track["hits"] += 1
    track["observations"].append(_observation(detection, timestamp))
    track["path"].append({"timestamp": timestamp, "centre": dict(detection["centre"])})
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
        "observations": [_observation(detection, timestamp)],
        "state": state,
        "covariance": covariance,
        "last_observed_state": state.copy(),
        "last_observed_covariance": covariance.copy(),
        "status": _ACTIVE if reached_active else _TENTATIVE,
        "hits": 1,
        "last_observed_timestamp": timestamp,
        "last_prediction_timestamp": timestamp,
        "required_hits": required_hits,
        "reached_active": reached_active,
        "component_track_ids": [str(track_id)],
        "creation_order": track_id,
    }


def _expire(track: dict, timestamp: float) -> None:
    elapsed = timestamp - track["last_observed_timestamp"]
    timeout = (
        TENTATIVE_TIMEOUT_SECONDS
        if track["status"] == _TENTATIVE
        else ACTIVE_TIMEOUT_SECONDS
    )
    if elapsed > timeout:
        track["status"] = _CLOSED


def _box_size(observation: dict) -> tuple[float, float]:
    bbox = observation["bbox"]
    return (
        max(float(bbox["x2"]) - float(bbox["x1"]), _MIN_BOX_SIZE),
        max(float(bbox["y2"]) - float(bbox["y1"]), _MIN_BOX_SIZE),
    )


def _endpoint_fit(observations: list[dict]) -> dict:
    reference_timestamp = float(observations[0]["timestamp"])
    times = np.array(
        [
            float(observation["timestamp"]) - reference_timestamp
            for observation in observations
        ],
        dtype=np.float64,
    )
    values = []
    for observation in observations:
        width, height = _box_size(observation)
        values.append(
            [
                float(observation["centre"]["x"]),
                float(observation["centre"]["y"]),
                math.log(width),
                math.log(height),
            ]
        )
    matrix = np.asarray(values, dtype=np.float64)
    if len(observations) == 1 or float(np.ptp(times)) <= 1e-9:
        return {
            "reference_timestamp": reference_timestamp,
            "intercept": matrix[0],
            "velocity": np.zeros(4, dtype=np.float64),
            "residual_variance": np.zeros(2, dtype=np.float64),
            "reliability": 0.0,
        }

    centred_times = times - float(np.mean(times))
    design = np.column_stack((np.ones(len(times), dtype=np.float64), centred_times))
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design, matrix, rcond=None
    )
    fitted = design @ coefficients
    intercept = coefficients[0] - coefficients[1] * float(np.mean(times))
    residual_variance = np.mean((matrix[:, :2] - fitted[:, :2]) ** 2, axis=0)
    reliability = {2: 0.35, 3: 0.70}.get(len(observations), 1.0)
    return {
        "reference_timestamp": reference_timestamp,
        "intercept": intercept,
        "velocity": coefficients[1],
        "residual_variance": residual_variance,
        "reliability": reliability,
    }


def _fit_value(fit: dict, timestamp: float) -> np.ndarray:
    return fit["intercept"] + fit["velocity"] * (timestamp - fit["reference_timestamp"])


def _fragment_summary(fragment: dict) -> dict:
    observations = fragment["observations"]
    return {
        "fragment": fragment,
        "first_timestamp": float(observations[0]["timestamp"]),
        "last_timestamp": float(observations[-1]["timestamp"]),
        "first_observation": observations[0],
        "last_observation": observations[-1],
        "start_fit": _endpoint_fit(observations[:_ENDPOINT_OBSERVATIONS]),
        "end_fit": _endpoint_fit(observations[-_ENDPOINT_OBSERVATIONS:]),
        "reached_active": fragment["reached_active"],
        "creation_order": fragment["creation_order"],
    }


def _predict_observed_endpoint(
    fragment: dict, timestamp: float
) -> tuple[np.ndarray, np.ndarray]:
    dt = timestamp - float(fragment["last_observed_timestamp"])
    transition, process_covariance = _transition_and_process_covariance(dt)
    state = transition @ fragment["last_observed_state"]
    covariance = (
        transition @ fragment["last_observed_covariance"] @ transition.T
        + process_covariance
    )
    return state, covariance


def _mahalanobis_2d(residual: np.ndarray, covariance: np.ndarray) -> float:
    try:
        distance = float(residual @ np.linalg.solve(covariance, residual))
    except np.linalg.LinAlgError:
        return math.inf
    return max(distance, 0.0) if math.isfinite(distance) else math.inf


def _continuity_cost(earlier: dict, later: dict) -> float | None:
    gap = later["first_timestamp"] - earlier["last_timestamp"]
    if not 0.0 < gap <= REFINE_MAX_GAP_SECONDS:
        return None
    if not (earlier["reached_active"] or later["reached_active"]):
        return None

    width_a, height_a = _box_size(earlier["last_observation"])
    width_b, height_b = _box_size(later["first_observation"])
    width_ratio = width_b / width_a
    height_ratio = height_b / height_a
    if not (
        _REFINE_MIN_BOX_RATIO <= width_ratio <= _REFINE_MAX_BOX_RATIO
        and _REFINE_MIN_BOX_RATIO <= height_ratio <= _REFINE_MAX_BOX_RATIO
    ):
        return None

    centre_a = np.array(
        [earlier["last_observation"]["centre"][key] for key in ("x", "y")],
        dtype=np.float64,
    )
    centre_b = np.array(
        [later["first_observation"]["centre"][key] for key in ("x", "y")],
        dtype=np.float64,
    )
    pooled_diagonal = math.hypot((width_a + width_b) / 2.0, (height_a + height_b) / 2.0)
    endpoint_speed = max(
        float(np.linalg.norm(earlier["end_fit"]["velocity"][:2])),
        float(np.linalg.norm(later["start_fit"]["velocity"][:2])),
    )
    if (
        float(np.linalg.norm(centre_b - centre_a))
        > 4.0 * pooled_diagonal + 1.5 * endpoint_speed * gap
    ):
        return None

    predicted_state, predicted_covariance = _predict_observed_endpoint(
        earlier["fragment"], later["first_timestamp"]
    )
    first_measurement = _measurement(later["first_observation"])
    forward_covariance = (
        predicted_covariance[:2, :2]
        + _measurement_covariance(first_measurement)[:2, :2]
    )
    forward_distance = _mahalanobis_2d(
        centre_b - predicted_state[:2], forward_covariance
    )

    backward_centre = _fit_value(later["start_fit"], earlier["last_timestamp"])[:2]
    centre_noise = np.array(
        [
            max(width_a * MEASUREMENT_CENTRE_NOISE, 1.0),
            max(height_a * MEASUREMENT_CENTRE_NOISE, 1.0),
        ],
        dtype=np.float64,
    )
    process_standard_deviation = CENTRE_PROCESS_NOISE * gap**2 / 2.0
    backward_variance = (
        centre_noise**2
        + later["start_fit"]["residual_variance"]
        + process_standard_deviation**2
    )
    if later["start_fit"]["reliability"] < 0.5:
        backward_variance += (pooled_diagonal * 0.25) ** 2
    backward_distance = _mahalanobis_2d(
        centre_a - backward_centre, np.diag(backward_variance)
    )
    if (
        forward_distance > _REFINE_POSITION_GATE
        or backward_distance > _REFINE_POSITION_GATE
    ):
        return None

    position_cost = 0.5 * (
        min(forward_distance / _REFINE_POSITION_GATE, 1.0)
        + min(backward_distance / _REFINE_POSITION_GATE, 1.0)
    )
    velocity_a = earlier["end_fit"]["velocity"][:2]
    velocity_b = later["start_fit"]["velocity"][:2]
    reliability = earlier["end_fit"]["reliability"] * later["start_fit"]["reliability"]
    speed_a = float(np.linalg.norm(velocity_a))
    speed_b = float(np.linalg.norm(velocity_b))
    jitter_speed = pooled_diagonal / max(gap, 1.0 / 3.0)
    if reliability > 0.0 and speed_a > 1e-9 and speed_b > 1e-9:
        cosine = float(np.dot(velocity_a, velocity_b) / (speed_a * speed_b))
        direction_cost = reliability * (1.0 - max(-1.0, min(cosine, 1.0))) / 2.0
        speed_cost = reliability * min(
            float(np.linalg.norm(velocity_a - velocity_b))
            / (speed_a + speed_b + jitter_speed),
            1.0,
        )
    else:
        direction_cost = 0.0
        speed_cost = 0.0
    box_cost = 0.5 * (
        min(abs(math.log(width_ratio)) / math.log(_REFINE_SOFT_BOX_RATIO), 1.0)
        + min(abs(math.log(height_ratio)) / math.log(_REFINE_SOFT_BOX_RATIO), 1.0)
    )
    gap_cost = (gap / REFINE_MAX_GAP_SECONDS) ** 2
    cost = (
        _REFINE_POSITION_WEIGHT * position_cost
        + _REFINE_DIRECTION_WEIGHT * direction_cost
        + _REFINE_SPEED_WEIGHT * speed_cost
        + _REFINE_BOX_WEIGHT * box_cost
        + _REFINE_GAP_WEIGHT * gap_cost
    )
    return cost if math.isfinite(cost) and cost <= REFINE_MAX_COST else None


def _candidate_costs(summaries: list[dict]) -> dict[tuple[int, int], float]:
    starts = sorted(
        (
            (summary["first_timestamp"], index)
            for index, summary in enumerate(summaries)
        ),
        key=lambda item: (item[0], summaries[item[1]]["creation_order"]),
    )
    start_timestamps = [item[0] for item in starts]
    costs = {}
    for earlier_index, earlier in enumerate(summaries):
        begin = bisect.bisect_right(start_timestamps, earlier["last_timestamp"])
        end = bisect.bisect_right(
            start_timestamps, earlier["last_timestamp"] + REFINE_MAX_GAP_SECONDS
        )
        for _timestamp, later_index in starts[begin:end]:
            if earlier_index == later_index:
                continue
            cost = _continuity_cost(earlier, summaries[later_index])
            if cost is not None:
                costs[(earlier_index, later_index)] = cost
    return costs


def _assign_fragment_links(summaries: list[dict]) -> dict[int, int]:
    candidate_costs = _candidate_costs(summaries)
    if not candidate_costs:
        return {}
    count = len(summaries)
    costs = np.full((count, count * 2), _IMPOSSIBLE_COST, dtype=np.float64)
    for (earlier_index, later_index), cost in candidate_costs.items():
        rank = earlier_index * count + later_index + 1
        costs[earlier_index, later_index] = cost + rank * _REFINE_TIE_EPSILON
    for index in range(count):
        costs[index, count + index] = REFINE_MAX_COST + _REFINE_TIE_EPSILON * (
            index + 1
        )
    rows, columns = linear_sum_assignment(costs)
    return {
        int(row): int(column)
        for row, column in zip(rows, columns, strict=True)
        if column < count and (int(row), int(column)) in candidate_costs
    }


def _merge_chain(chain: list[dict]) -> dict:
    observations = []
    seen = set()
    for fragment in chain:
        for observation in fragment["observations"]:
            key = (
                float(observation["timestamp"]),
                float(observation["centre"]["x"]),
                float(observation["centre"]["y"]),
            )
            if key not in seen:
                seen.add(key)
                observations.append(observation)
    observations.sort(
        key=lambda observation: (
            float(observation["timestamp"]),
            float(observation["centre"]["x"]),
            float(observation["centre"]["y"]),
        )
    )
    earliest = min(chain, key=lambda fragment: fragment["creation_order"])
    latest = max(chain, key=lambda fragment: fragment["last_observed_timestamp"])
    best = min(chain, key=lambda fragment: fragment["creation_order"])
    for fragment in sorted(chain, key=lambda item: item["creation_order"]):
        if fragment["best_crop_confidence"] > best["best_crop_confidence"]:
            best = fragment
    return {
        **earliest,
        "track_id": earliest["track_id"],
        "path": [
            {
                "timestamp": observation["timestamp"],
                "centre": dict(observation["centre"]),
            }
            for observation in observations
        ],
        "observations": observations,
        "best_crop": best["best_crop"],
        "best_crop_confidence": best["best_crop_confidence"],
        "last_observed_timestamp": latest["last_observed_timestamp"],
        "last_prediction_timestamp": latest["last_observed_timestamp"],
        "state": latest["last_observed_state"].copy(),
        "covariance": latest["last_observed_covariance"].copy(),
        "last_observed_state": latest["last_observed_state"].copy(),
        "last_observed_covariance": latest["last_observed_covariance"].copy(),
        "reached_active": any(fragment["reached_active"] for fragment in chain),
        "component_track_ids": [
            component_id
            for fragment in chain
            for component_id in fragment["component_track_ids"]
        ],
    }


def _merge_linked_fragments(fragments: list[dict], links: dict[int, int]) -> list[dict]:
    predecessors = {successor: predecessor for predecessor, successor in links.items()}
    visited = set()
    merged = []
    for root in range(len(fragments)):
        if root in predecessors or root in visited:
            continue
        chain_indices = []
        current = root
        while current not in visited:
            visited.add(current)
            chain_indices.append(current)
            if current not in links:
                break
            current = links[current]
        chain = [fragments[index] for index in chain_indices]
        merged.append(_merge_chain(chain) if len(chain) > 1 else chain[0])
    merged.extend(
        fragments[index] for index in range(len(fragments)) if index not in visited
    )
    return sorted(merged, key=lambda fragment: fragment["creation_order"])


def _refine_fragments(fragments: list[dict]) -> list[dict]:
    refined = list(fragments)
    for _round in range(REFINE_MAX_ROUNDS):
        summaries = [_fragment_summary(fragment) for fragment in refined]
        links = _assign_fragment_links(summaries)
        if not links:
            break
        refined = _merge_linked_fragments(refined, links)
    return refined


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
            indexed_frames.append(
                (timestamp, str(frame["frame_id"]), original_index, frame)
            )
        indexed_frames.sort(key=lambda item: item[:3])

        tracks: list[dict] = []
        next_track_id = 1
        for frame_index, (timestamp, _frame_key, _original_index, frame) in enumerate(
            indexed_frames
        ):
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
            eligible_tracks = [
                track for track in eligible_tracks if track["status"] != _CLOSED
            ]

            primary_matches, unmatched_track_indices, unmatched_high_indices = (
                _associate(eligible_tracks, high_detections, PRIMARY_MAX_COST)
            )
            for track_index, detection_index in primary_matches:
                _update(
                    eligible_tracks[track_index],
                    high_detections[detection_index],
                    frame_id,
                    timestamp,
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
                    recovery_tracks[track_index],
                    low_detections[detection_index],
                    frame_id,
                    timestamp,
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

        if REFINE_ENABLED:
            tracks = _refine_fragments(tracks)
        return {
            "tracks": [
                _public_track(track) for track in tracks if track["reached_active"]
            ]
        }
