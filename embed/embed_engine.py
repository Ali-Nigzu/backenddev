"""DetectionBatch -> EmbeddingBatch."""

from math import isfinite
from pathlib import Path

import numpy as np
import torch

from ._osnet import osnet_x0_25

__all__ = ["Embed"]


class Embed:
    __slots__ = ("_device", "_model")

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = osnet_x0_25()

        checkpoint = torch.load(
            Path(__file__).with_name("osnet_x0_25_msmt17.pth"),
            map_location=self._device,
        )
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        model_dict = self._model.state_dict()
        model_dict.update(
            {
                key.replace("module.", ""): value
                for key, value in state_dict.items()
                if "classifier" not in key
                and key.replace("module.", "") in model_dict
                and model_dict[key.replace("module.", "")].shape == value.shape
            }
        )
        self._model.load_state_dict(model_dict, strict=False)
        self._model.to(self._device)
        self._model.eval()

    def __call__(self, detection_batch):
        for field in ("frame_id", "timestamp", "detections"):
            if field not in detection_batch:
                raise ValueError(f"Missing required DetectionBatch field: {field}")

        frame_id = detection_batch["frame_id"]
        timestamp = detection_batch["timestamp"]
        detections = detection_batch["detections"]

        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("DetectionBatch.frame_id must be a non-empty string")
        if not isinstance(timestamp, (float, int)) or not isfinite(float(timestamp)):
            raise ValueError("DetectionBatch.timestamp must be finite")
        if not isinstance(detections, list):
            raise ValueError("DetectionBatch.detections must be a list")

        image = None
        if detections:
            frame = detection_batch.get("frame")
            if not isinstance(frame, dict) or "image" not in frame:
                raise ValueError("DetectionBatch frame image is required when detections are present")
            image = frame["image"]
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Frame.image must be an image array with shape [H, W, 3]")

        embeddings = []
        for detection in detections:
            detection_id = detection["detection_id"]
            bbox = detection["bbox"]
            crop = image[
                max(0, int(bbox["y1"])) : min(image.shape[0], int(bbox["y2"])),
                max(0, int(bbox["x1"])) : min(image.shape[1], int(bbox["x2"])),
            ]
            if crop.size == 0:
                raise ValueError("Detection bbox produced an empty crop")

            tensor = (
                torch.from_numpy(np.ascontiguousarray(crop))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self._device)
            )
            with torch.inference_mode():
                vector = (
                    self._model(tensor)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

            norm = np.linalg.norm(vector)
            if norm > 0.0:
                vector = (vector / norm).astype(np.float32, copy=False)

            embeddings.append(
                {
                    "detection_id": detection_id,
                    "vector": {
                        "dtype": "float32",
                        "shape": [int(vector.shape[0])],
                        "values": vector.tolist(),
                    },
                }
            )

        return {
            "frame_id": frame_id,
            "timestamp": float(timestamp),
            "embeddings": embeddings,
        }
