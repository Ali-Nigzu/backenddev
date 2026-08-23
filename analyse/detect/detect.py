from pathlib import Path

import numpy as np
from ultralytics import YOLO

__all__ = ["Detect"]

MODEL_PATH = Path(__file__).with_name("yolov10n.pt")
PERSON_CLASSES = (0,)
CONFIDENCE = 0.25
IOU = 0.70
MAX_DETECTIONS = 300
DEVICE = "cpu"
VERBOSE = False

class Detect:
    __slots__ = ("_model",)

    def __init__(self) -> None:
        self._model = YOLO(str(MODEL_PATH))

    def __call__(self, frame_batch):
        return {"detections": [self._detect_frame(frame) for frame in frame_batch["frames"]]}

    def _detect_frame(self, frame):
        frame_id = frame["frame_id"]
        timestamp = frame["timestamp"]
        image = frame["image"]

        detections = []
        width = image.shape[1]
        height = image.shape[0]
        detection_index = 0

        for result in self._model(
            image,
            classes=PERSON_CLASSES,
            conf=CONFIDENCE,
            iou=IOU,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            verbose=VERBOSE,
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

                x1 = np.float32(x1).item()
                y1 = np.float32(y1).item()
                x2 = np.float32(x2).item()
                y2 = np.float32(y2).item()
                centre_x = np.float32((x1 + x2) / 2.0).item()
                centre_y = np.float32((y1 + y2) / 2.0).item()
                confidence = np.float32(confidence).item()
                detections.append(
                    {
                        "detection_id": f"{frame_id}:det:{detection_index}",
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "centre": {"x": centre_x, "y": centre_y},
                        "confidence": confidence,
                    }
                )
                detection_index += 1

        return {
            "frame_id": frame_id,
            "timestamp": float(timestamp),
            "detections": detections,
        }
