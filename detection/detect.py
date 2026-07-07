"""Frame -> DetectionBatch."""

from math import isfinite
from pathlib import Path

import numpy as np
from ultralytics import YOLO

__all__ = ["Detect"]


class Detect:
    __slots__ = ("_model",)

    def __init__(self) -> None:
        self._model = YOLO(str(Path(__file__).with_name("yolov10n.pt")))

    def __call__(self, frame):
        for field in ("frame_id", "timestamp", "image"):
            if field not in frame:
                raise ValueError(f"Missing required Frame field: {field}")

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

        detections = []
        width = float(image.shape[1])
        height = float(image.shape[0])
        detection_index = 0

        for result in self._model(
            image,
            classes=(0,),
            conf=0.25,
            iou=0.70,
            max_det=300,
            device="cpu",
            verbose=False,
        ):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            for row_index, confidence in enumerate(conf):
                x1_raw, y1_raw, x2_raw, y2_raw = xyxy[row_index]
                x1 = min(max(float(x1_raw), 0.0), width)
                y1 = min(max(float(y1_raw), 0.0), height)
                x2 = min(max(float(x2_raw), 0.0), width)
                y2 = min(max(float(y2_raw), 0.0), height)
                if x1 >= x2 or y1 >= y2:
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

        return {
            "frame_id": frame_id,
            "timestamp": float(timestamp),
            "detections": detections,
        }
