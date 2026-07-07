"""DetectionBatch -> EmbeddingBatch."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

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
        state_dict = checkpoint.get("state_dict", checkpoint)
        self._model.load_state_dict(
            {
                key.replace("module.", ""): value
                for key, value in state_dict.items()
                if not key.replace("module.", "").startswith("classifier")
            },
            strict=False,
        )
        self._model.to(self._device)
        self._model.eval()

    def __call__(self, detection_batch):
        for field in ("frame_id", "timestamp", "frame", "detections"):
            if field not in detection_batch:
                raise ValueError(f"Missing required DetectionBatch field: {field}")

        image = detection_batch["frame"].get("image")
        if image is None:
            raise ValueError("Missing required DetectionBatch frame image")

        embeddings = []
        for detection in detection_batch["detections"]:
            detection_id = detection["detection_id"]
            bbox = detection["bbox"]
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(image.shape[1], int(bbox["x2"]))
            y2 = min(image.shape[0], int(bbox["y2"]))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError("Detection bbox produced an empty crop")

            tensor = (
                torch.from_numpy(np.ascontiguousarray(crop))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self._device)
                / 255.0
            )
            tensor = F.interpolate(tensor, size=(256, 128), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std

            with torch.inference_mode():
                vector = (
                    self._model(tensor)
                    .squeeze(0)
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
            "frame_id": detection_batch["frame_id"],
            "timestamp": detection_batch["timestamp"],
            "embeddings": embeddings,
        }
