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
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self._model.load_state_dict(
            {
                key.replace("module.", ""): value
                for key, value in state_dict.items()
                if "classifier" not in key
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

        detections = detection_batch["detections"]
        if detections is None:
            raise ValueError("Missing required DetectionBatch detections")

        tensors = []
        detection_ids = []
        for detection in detections:
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
                .div(255.0)
            )
            tensors.append(
                F.interpolate(
                    tensor,
                    size=(256, 128),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            )
            detection_ids.append(detection["detection_id"])

        embeddings = []
        if tensors:
            batch = torch.stack(tensors).to(self._device)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(
                1, 3, 1, 1
            )
            std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(
                1, 3, 1, 1
            )
            batch = (batch - mean) / std

            with torch.inference_mode():
                vectors = (
                    self._model(batch).cpu().numpy().astype(np.float32, copy=False)
                )

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.divide(vectors, norms, out=vectors, where=norms > 0.0)

            for detection_id, vector in zip(detection_ids, vectors):
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
