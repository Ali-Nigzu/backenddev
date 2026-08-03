"""MiVOLO demographic inference for event body crops."""

import math
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from ._mivolo.model.mivolo_model import create_mivolo_d1_224

_CHECKPOINT_PATH = Path(__file__).with_name("demographicweights.pth")
_INPUT_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IGNORED_STATE_PREFIXES = ("fds.",)
_EXPECTED_OUTPUTS = 3
_CPU_CHUNK_SIZE = 16
_CUDA_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _CropDescriptor:
    track_id: str
    timestamp: float
    frame_id: str
    bbox: dict


def _select_unique_tracks(events):
    by_track = {}
    for event in events:
        crop = event["best_crop"]
        descriptor = _CropDescriptor(
            event["track_id"], float(event["timestamp"]), crop["frame_id"], dict(crop["bbox"])
        )
        existing = by_track.get(descriptor.track_id)
        if existing is None or descriptor.timestamp < existing.timestamp:
            by_track[descriptor.track_id] = descriptor
    return sorted(by_track.values(), key=lambda item: (item.timestamp, item.track_id))


def _crop_body(image, descriptor):
    height, width = image.shape[:2]
    bbox = descriptor.bbox
    left = max(0, min(width, math.floor(float(bbox["x1"]))))
    top = max(0, min(height, math.floor(float(bbox["y1"]))))
    right = max(0, min(width, math.ceil(float(bbox["x2"]))))
    bottom = max(0, min(height, math.ceil(float(bbox["y2"]))))
    if right <= left or bottom <= top:
        raise RuntimeError(
            f"Body crop has zero area for track_id={descriptor.track_id} "
            f"frame_id={descriptor.frame_id}"
        )
    return np.ascontiguousarray(image[top:bottom, left:right])


def _letterbox_rgb(image):
    height, width = image.shape[:2]
    scale = min(_INPUT_SIZE / height, _INPUT_SIZE / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    dw = (_INPUT_SIZE - resized_width) / 2
    dh = (_INPUT_SIZE - resized_height) / 2
    return cv2.copyMakeBorder(
        image,
        int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)),
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def _normalise_rgb(image):
    return (image.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD


@lru_cache(maxsize=1)
def _missing_face_tensor():
    black = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
    return np.ascontiguousarray(_normalise_rgb(black).transpose(2, 0, 1), dtype=np.float32)


def _prepare_body(image, descriptor):
    body = _normalise_rgb(_letterbox_rgb(_crop_body(image, descriptor)))
    body = np.ascontiguousarray(body.transpose(2, 0, 1), dtype=np.float32)
    return np.ascontiguousarray(
        np.concatenate((_missing_face_tensor(), body), axis=0).astype(np.float32, copy=False)
    )


class Demographic:
    __slots__ = ("_lock", "_model", "_device", "_min_age", "_max_age", "_avg_age")

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._device = None
        self._min_age = self._max_age = self._avg_age = None

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not _CHECKPOINT_PATH.exists():
                raise RuntimeError(f"MiVOLO checkpoint not found at {_CHECKPOINT_PATH}")
            try:
                checkpoint = torch.load(str(_CHECKPOINT_PATH), map_location="cpu")
            except Exception as exc:
                raise RuntimeError(f"Unable to load MiVOLO checkpoint at {_CHECKPOINT_PATH}") from exc
            try:
                model = create_mivolo_d1_224(num_classes=_EXPECTED_OUTPUTS, in_chans=6)
                state_dict = {
                    key: value for key, value in checkpoint["state_dict"].items()
                    if not key.startswith(_IGNORED_STATE_PREFIXES)
                }
                model.load_state_dict(state_dict, strict=True)
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = model.to(self._device).eval()
                self._min_age = float(checkpoint["min_age"])
                self._max_age = float(checkpoint["max_age"])
                self._avg_age = float(checkpoint["avg_age"])
            except Exception as exc:
                raise RuntimeError("Unable to construct or load the MiVOLO model") from exc

    def __call__(self, event_batch, frame_batch):
        if not event_batch["events"]:
            return {"results": []}
        descriptors = _select_unique_tracks(event_batch["events"])
        frames_by_id = {frame["frame_id"]: frame for frame in frame_batch["frames"]}
        for descriptor in descriptors:
            if descriptor.frame_id not in frames_by_id:
                raise RuntimeError(
                    f"Missing source frame: track_id={descriptor.track_id} "
                    f"frame_id={descriptor.frame_id}"
                )

        self._load()
        predictions = []
        chunk_size = _CUDA_CHUNK_SIZE if self._device == "cuda" else _CPU_CHUNK_SIZE
        for start in range(0, len(descriptors), chunk_size):
            chunk = descriptors[start:start + chunk_size]
            batch = np.ascontiguousarray(np.stack([
                _prepare_body(frames_by_id[item.frame_id]["image"], item) for item in chunk
            ], axis=0).astype(np.float32, copy=False))
            try:
                with torch.inference_mode():
                    output = self._model(torch.from_numpy(batch).to(self._device))
                output = np.asarray(output.detach().cpu().numpy(), dtype=np.float32)
            except Exception as exc:
                raise RuntimeError("MiVOLO inference failed") from exc
            if len(output) != len(chunk):
                raise RuntimeError("MiVOLO inference output count does not match its input")
            for row in output:
                age = int(math.floor(float(row[2]) * (self._max_age - self._min_age) + self._avg_age + 0.5))
                predictions.append({"age": age, "sex": 1 if float(row[0]) >= float(row[1]) else 0})

        return {"results": [
            {"track_id": descriptor.track_id, "age": prediction["age"], "sex": prediction["sex"]}
            for descriptor, prediction in zip(descriptors, predictions, strict=True)
        ]}
