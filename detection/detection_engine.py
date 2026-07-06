"""Production DetectV2 implementation for the Analytics Engine V2 contract.

Public contract:
    Frame -> DetectV2 -> DetectionBatch

The module intentionally returns plain dictionaries whose keys exactly match the
locked V2 DetectionBatch contract. It does not expose crops, class labels,
centres, embeddings, tracking identifiers, or backend metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np


Frame = Mapping[str, Any]
DetectionBatch = dict[str, Any]


class DetectV2InputError(ValueError):
    """Raised when a Frame cannot satisfy the locked DetectV2 input contract."""


@dataclass(frozen=True, slots=True)
class DetectV2Config:
    """Immutable DetectV2 runtime configuration."""

    model_path: Path = Path(__file__).with_name("yolov10n.pt")
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.70
    max_detections: int = 300
    classes: tuple[int, ...] = (0,)
    device: str | None = "cpu"

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0.0, 1.0]")
        if self.max_detections < 0:
            raise ValueError("max_detections must be non-negative")


@dataclass(frozen=True, slots=True)
class RawDetection:
    """Backend-normalised candidate detection before final contract mapping."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    candidate_index: int


class DetectBackend(Protocol):
    """Minimal backend protocol used by DetectV2.

    Implementations may own model weights and runtime handles, but must not keep
    frame-specific analytics state after ``infer`` returns.
    """

    def infer(self, image: np.ndarray) -> Iterable[RawDetection]:
        """Return candidate detections in original image coordinates."""


class UltralyticsYoloBackend:
    """YOLO backend wrapper used by the production DetectV2 runtime.

    Imports are intentionally lazy so contract tests can run in environments that
    do not have the inference stack installed. The loaded model object is the
    only persistent resource retained by this backend.
    """

    __slots__ = ("_classes_arg", "_config", "_model")

    def __init__(self, config: DetectV2Config) -> None:
        self._config = config
        self._classes_arg = list(config.classes)
        self._enable_deterministic_torch()

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised by deployment env
            raise RuntimeError(
                "ultralytics is required for the production DetectV2 YOLO backend"
            ) from exc

        self._model = YOLO(str(config.model_path))

    @staticmethod
    def _enable_deterministic_torch() -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch arrives with deployment deps
            return

        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def infer(self, image: np.ndarray) -> Iterable[RawDetection]:
        cfg = self._config
        if cfg.max_detections == 0:
            return ()

        results = self._model(
            image,
            classes=self._classes_arg,
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            max_det=cfg.max_detections,
            device=cfg.device,
            verbose=False,
        )

        candidates: list[RawDetection] = []
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
                    RawDetection(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=float(confidence_values[row_index]),
                        candidate_index=candidate_index,
                    )
                )
                candidate_index += 1

        del results
        return candidates


class DetectV2:
    """Stateless V2 detector that returns exactly one DetectionBatch per Frame."""

    __slots__ = ("_config", "_backend")

    def __init__(
        self,
        config: DetectV2Config | None = None,
        backend: DetectBackend | None = None,
    ) -> None:
        self._config = config or DetectV2Config()
        self._backend = backend or UltralyticsYoloBackend(self._config)

    def detect(self, frame: Frame) -> DetectionBatch:
        """Run object detection for one Frame and return one DetectionBatch."""

        frame_id, timestamp, image = self._validate_frame(frame)
        height, width = image.shape[:2]

        raw_candidates = self._backend.infer(image)
        detections = self._build_detections(
            frame_id=frame_id,
            raw_candidates=raw_candidates,
            width=width,
            height=height,
        )

        batch: DetectionBatch = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "detections": detections,
        }

        del raw_candidates
        del image
        return batch

    @staticmethod
    def _validate_frame(frame: Frame) -> tuple[str, float, np.ndarray]:
        if not isinstance(frame, Mapping):
            raise DetectV2InputError("Frame must be a mapping")

        for field in ("frame_id", "timestamp", "image"):
            if field not in frame:
                raise DetectV2InputError(f"Missing required Frame field: {field}")

        frame_id = frame["frame_id"]
        timestamp = frame["timestamp"]
        image = frame["image"]

        if not isinstance(frame_id, str) or not frame_id:
            raise DetectV2InputError("Frame.frame_id must be a non-empty string")
        if not isinstance(timestamp, (float, int)) or not isfinite(float(timestamp)):
            raise DetectV2InputError("Frame.timestamp must be finite float64-compatible")
        if not isinstance(image, np.ndarray):
            raise DetectV2InputError("Frame.image must be a numpy ndarray")
        if image.dtype != np.uint8:
            raise DetectV2InputError("Frame.image dtype must be uint8")
        if image.ndim != 3 or image.shape[2] != 3:
            raise DetectV2InputError("Frame.image shape must be [H, W, 3]")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise DetectV2InputError("Frame.image dimensions must be non-empty")
        if not image.flags.c_contiguous:
            raise DetectV2InputError("Frame.image must be contiguous row-major")

        return frame_id, float(timestamp), image

    def _build_detections(
        self,
        *,
        frame_id: str,
        raw_candidates: Iterable[RawDetection],
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        max_detections = self._config.max_detections
        if max_detections == 0:
            return []

        valid_candidates = self._valid_candidates(raw_candidates, width, height)
        nms_candidates = self._deterministic_nms(valid_candidates)
        if len(nms_candidates) > max_detections:
            del nms_candidates[max_detections:]

        detections: list[dict[str, Any]] = [None] * len(  # type: ignore[list-item]
            nms_candidates
        )
        for index, candidate in enumerate(nms_candidates):
            detections[index] = {
                "detection_id": f"{frame_id}:det:{index}",
                "bbox": {
                    "x1": float(np.float32(candidate.x1)),
                    "y1": float(np.float32(candidate.y1)),
                    "x2": float(np.float32(candidate.x2)),
                    "y2": float(np.float32(candidate.y2)),
                },
                "confidence": float(np.float32(candidate.confidence)),
            }

        return detections

    def _valid_candidates(
        self, raw_candidates: Iterable[RawDetection], width: int, height: int
    ) -> list[RawDetection]:
        valid: list[RawDetection] = []
        width_f = float(width)
        height_f = float(height)

        confidence_threshold = self._config.confidence_threshold

        for candidate in raw_candidates:
            x1_raw = float(candidate.x1)
            y1_raw = float(candidate.y1)
            x2_raw = float(candidate.x2)
            y2_raw = float(candidate.y2)
            confidence = float(candidate.confidence)

            if not (
                isfinite(x1_raw)
                and isfinite(y1_raw)
                and isfinite(x2_raw)
                and isfinite(y2_raw)
                and isfinite(confidence)
            ):
                continue

            if confidence < confidence_threshold or confidence > 1.0:
                continue

            x1 = min(max(x1_raw, 0.0), width_f)
            y1 = min(max(y1_raw, 0.0), height_f)
            x2 = min(max(x2_raw, 0.0), width_f)
            y2 = min(max(y2_raw, 0.0), height_f)

            if x1 >= x2 or y1 >= y2:
                continue

            valid.append(
                RawDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    candidate_index=int(candidate.candidate_index),
                )
            )

        return valid

    def _deterministic_nms(
        self, candidates: Sequence[RawDetection]
    ) -> list[RawDetection]:
        ordered = sorted(candidates, key=self._ordering_key)
        kept: list[RawDetection] = []

        for candidate in ordered:
            if all(
                self._intersection_over_union(candidate, existing)
                <= self._config.iou_threshold
                for existing in kept
            ):
                kept.append(candidate)

        return kept

    @staticmethod
    def _ordering_key(
        candidate: RawDetection,
    ) -> tuple[float, float, float, float, float, int]:
        return (
            -candidate.confidence,
            candidate.x1,
            candidate.y1,
            candidate.x2,
            candidate.y2,
            candidate.candidate_index,
        )

    @staticmethod
    def _intersection_over_union(first: RawDetection, second: RawDetection) -> float:
        x_left = max(first.x1, second.x1)
        y_top = max(first.y1, second.y1)
        x_right = min(first.x2, second.x2)
        y_bottom = min(first.y2, second.y2)

        intersection_width = max(0.0, x_right - x_left)
        intersection_height = max(0.0, y_bottom - y_top)
        intersection = intersection_width * intersection_height
        if intersection <= 0.0:
            return 0.0

        first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
        second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
        union = first_area + second_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union
