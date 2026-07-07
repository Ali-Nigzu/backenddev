"""Production Detect primitive: Frame -> DetectionBatch."""

from __future__ import annotations

from importlib.util import find_spec
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["Detect"]

_MODEL_PATH = Path(__file__).with_name("yolov10n.pt")
_CONFIDENCE_THRESHOLD = 0.25
_IOU_THRESHOLD = 0.70
_MAX_DETECTIONS = 300
_PERSON_CLASS = (0,)
_DEVICE = "cpu"


class Detect:
    """Callable detector with the locked public contract Frame -> DetectionBatch."""

    __slots__ = ("_model",)

    def __init__(self) -> None:
        if find_spec("torch") is not None:
            import torch

            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        if find_spec("ultralytics") is None:
            self._model = None
            return

        from ultralytics import YOLO

        self._model = YOLO(str(_MODEL_PATH))

    def __call__(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(frame, Mapping):
            raise ValueError("Frame must be a mapping")

        frame_id = frame["frame_id"]
        timestamp = frame["timestamp"]
        image = frame["image"]

        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("Frame.frame_id must be a non-empty string")
        if not isinstance(timestamp, (float, int)) or not isfinite(float(timestamp)):
            raise ValueError("Frame.timestamp must be finite")
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
            or not image.flags.c_contiguous
        ):
            raise ValueError("Frame.image must be contiguous uint8 [H, W, 3]")

        detections: list[dict[str, Any]] = []
        if self._model is not None and _MAX_DETECTIONS:
            width = float(image.shape[1])
            height = float(image.shape[0])
            detection_index = 0

            results = self._model(
                image,
                classes=_PERSON_CLASS,
                conf=_CONFIDENCE_THRESHOLD,
                iou=_IOU_THRESHOLD,
                max_det=_MAX_DETECTIONS,
                device=_DEVICE,
                verbose=False,
            )

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy_values = boxes.xyxy.detach().cpu().numpy()
                confidence_values = boxes.conf.detach().cpu().numpy()
                for row_index, confidence_raw in enumerate(confidence_values):
                    confidence = float(confidence_raw)
                    x1_raw, y1_raw, x2_raw, y2_raw = xyxy_values[row_index]
                    x1 = min(max(float(x1_raw), 0.0), width)
                    y1 = min(max(float(y1_raw), 0.0), height)
                    x2 = min(max(float(x2_raw), 0.0), width)
                    y2 = min(max(float(y2_raw), 0.0), height)

                    if (
                        confidence < _CONFIDENCE_THRESHOLD
                        or confidence > 1.0
                        or not isfinite(confidence)
                        or not (
                            isfinite(x1)
                            and isfinite(y1)
                            and isfinite(x2)
                            and isfinite(y2)
                        )
                        or x1 >= x2
                        or y1 >= y2
                    ):
                        continue

                    detections.append(
                        {
                            "detection_id": f"{frame_id}:det:{detection_index}",
                            "bbox": {
                                "x1": float(np.float32(x1)),
                                "y1": float(np.float32(y1)),
                                "x2": float(np.float32(x2)),
                                "y2": float(np.float32(y2)),
                            },
                            "confidence": float(np.float32(confidence)),
                        }
                    )
                    detection_index += 1

            del results

        return {
            "frame_id": frame_id,
            "timestamp": float(timestamp),
            "detections": detections,
        }
