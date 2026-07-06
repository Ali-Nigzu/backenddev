"""DetectV2 compute primitive: Frame -> DetectionBatch."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = ["DetectV2"]

_MODEL_PATH = Path(__file__).with_name("yolov10n.pt")
_CONFIDENCE_THRESHOLD = 0.25
_IOU_THRESHOLD = 0.70
_MAX_DETECTIONS = 300
_CLASSES = (0,)
_DEVICE = "cpu"

_Candidate = tuple[float, float, float, float, float, int]


class DetectV2:
    """Callable V2 detector with the public contract Frame -> DetectionBatch."""

    __slots__ = ("_model",)

    def __init__(self) -> None:
        _enable_deterministic_torch()
        try:
            from ultralytics import YOLO
        except ImportError:
            self._model = None
        else:
            self._model = YOLO(str(_MODEL_PATH))

    def __call__(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        frame_id, timestamp, image = _validate_frame(frame)
        height, width = image.shape[:2]

        candidates = self._infer(image)
        detections = _build_detections(frame_id, candidates, width, height)

        del candidates
        del image
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "detections": detections,
        }

    def _infer(self, image: np.ndarray) -> Iterable[_Candidate]:
        if self._model is None or _MAX_DETECTIONS == 0:
            return ()

        results = self._model(
            image,
            classes=_CLASSES,
            conf=_CONFIDENCE_THRESHOLD,
            iou=_IOU_THRESHOLD,
            max_det=_MAX_DETECTIONS,
            device=_DEVICE,
            verbose=False,
        )

        candidates: list[_Candidate] = []
        candidate_index = 0
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy_values = boxes.xyxy.detach().cpu().numpy()
            confidence_values = boxes.conf.detach().cpu().numpy()
            for row_index in range(len(confidence_values)):
                xyxy = xyxy_values[row_index]
                candidates.append(
                    (
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                        float(confidence_values[row_index]),
                        candidate_index,
                    )
                )
                candidate_index += 1

        del results
        return candidates


def _validate_frame(frame: Mapping[str, Any]) -> tuple[str, float, np.ndarray]:
    if not isinstance(frame, Mapping):
        raise ValueError("Frame must be a mapping")

    for field in ("frame_id", "timestamp", "image"):
        if field not in frame:
            raise ValueError(f"Missing required Frame field: {field}")

    frame_id = frame["frame_id"]
    timestamp = frame["timestamp"]
    image = frame["image"]

    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("Frame.frame_id must be a non-empty string")
    if not isinstance(timestamp, (float, int)) or not isfinite(float(timestamp)):
        raise ValueError("Frame.timestamp must be finite float64-compatible")
    if not isinstance(image, np.ndarray):
        raise ValueError("Frame.image must be a numpy ndarray")
    if image.dtype != np.uint8:
        raise ValueError("Frame.image dtype must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Frame.image shape must be [H, W, 3]")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Frame.image dimensions must be non-empty")
    if not image.flags.c_contiguous:
        raise ValueError("Frame.image must be contiguous row-major")

    return frame_id, float(timestamp), image


def _build_detections(
    frame_id: str, raw_candidates: Iterable[_Candidate], width: int, height: int
) -> list[dict[str, Any]]:
    if _MAX_DETECTIONS == 0:
        return []

    valid_candidates = _valid_candidates(raw_candidates, width, height)
    kept_candidates = _deterministic_nms(valid_candidates)
    if len(kept_candidates) > _MAX_DETECTIONS:
        del kept_candidates[_MAX_DETECTIONS:]

    detections: list[dict[str, Any]] = []
    for index, candidate in enumerate(kept_candidates):
        x1, y1, x2, y2, confidence, _ = candidate
        detections.append(
            {
                "detection_id": f"{frame_id}:det:{index}",
                "bbox": {
                    "x1": float(np.float32(x1)),
                    "y1": float(np.float32(y1)),
                    "x2": float(np.float32(x2)),
                    "y2": float(np.float32(y2)),
                },
                "confidence": float(np.float32(confidence)),
            }
        )

    return detections


def _valid_candidates(
    raw_candidates: Iterable[_Candidate], width: int, height: int
) -> list[_Candidate]:
    valid: list[_Candidate] = []
    width_f = float(width)
    height_f = float(height)

    for candidate in raw_candidates:
        x1_raw, y1_raw, x2_raw, y2_raw, confidence, candidate_index = candidate
        if not (
            isfinite(x1_raw)
            and isfinite(y1_raw)
            and isfinite(x2_raw)
            and isfinite(y2_raw)
            and isfinite(confidence)
        ):
            continue
        if confidence < _CONFIDENCE_THRESHOLD or confidence > 1.0:
            continue

        x1 = min(max(x1_raw, 0.0), width_f)
        y1 = min(max(y1_raw, 0.0), height_f)
        x2 = min(max(x2_raw, 0.0), width_f)
        y2 = min(max(y2_raw, 0.0), height_f)
        if x1 >= x2 or y1 >= y2:
            continue

        valid.append((x1, y1, x2, y2, confidence, int(candidate_index)))

    return valid


def _deterministic_nms(candidates: list[_Candidate]) -> list[_Candidate]:
    candidates.sort(key=_ordering_key)
    kept: list[_Candidate] = []

    for candidate in candidates:
        if all(
            _intersection_over_union(candidate, existing) <= _IOU_THRESHOLD
            for existing in kept
        ):
            kept.append(candidate)

    return kept


def _ordering_key(
    candidate: _Candidate,
) -> tuple[float, float, float, float, float, int]:
    x1, y1, x2, y2, confidence, candidate_index = candidate
    return (-confidence, x1, y1, x2, y2, candidate_index)


def _intersection_over_union(first: _Candidate, second: _Candidate) -> float:
    first_x1, first_y1, first_x2, first_y2 = first[:4]
    second_x1, second_y1, second_x2, second_y2 = second[:4]

    x_left = max(first_x1, second_x1)
    y_top = max(first_y1, second_y1)
    x_right = min(first_x2, second_x2)
    y_bottom = min(first_y2, second_y2)

    intersection_width = max(0.0, x_right - x_left)
    intersection_height = max(0.0, y_bottom - y_top)
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0

    first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
    second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _enable_deterministic_torch() -> None:
    try:
        import torch
    except ImportError:
        return

    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
