"""FrameBatch -> DetectionBatch."""

from pathlib import Path

import numpy as np
from contracts import validate_frame_batch
from ultralytics import YOLO

__all__ = ["Detect"]


class Detect:
    __slots__ = ("_model",)

    def __init__(self) -> None:
        self._model = YOLO(str(Path(__file__).with_name("yolov10n.pt")))

    def __call__(self, frame_batch):
        frames = validate_frame_batch(frame_batch)
        return {"detections": [self._detect_frame(frame) for frame in frames]}

    def _detect_frame(self, frame):
        frame_id = frame["frame_id"]
        timestamp = frame["timestamp"]
        image = frame["image"]

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

                bbox = {
                    "x1": float(np.float32(x1)),
                    "y1": float(np.float32(y1)),
                    "x2": float(np.float32(x2)),
                    "y2": float(np.float32(y2)),
                }
                detections.append(
                    {
                        "detection_id": f"{frame_id}:det:{detection_index}",
                        "bbox": bbox,
                        "centre": {
                            "x": float(np.float32((bbox["x1"] + bbox["x2"]) / 2.0)),
                            "y": float(np.float32((bbox["y1"] + bbox["y2"]) / 2.0)),
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
