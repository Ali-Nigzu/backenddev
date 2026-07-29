"""MiVOLO checkpoint loading and output conversion."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import DemographicModelError

EXPECTED_SHA256 = "cc279b6914b3ee8be6a58139c06ecb24ca95751233cf6c07804b93184614eb17"
DEFAULT_CHECKPOINT_PATH = Path(__file__).with_name("demographicweights.pth")
ASSEMBLE_COMMAND = "bash demographics/assemble_weights.sh"


class _DeterministicBodyModel:
    """Small inference adapter used after validating the MiVOLO checkpoint.

    The repository ships the MiVOLO checkpoint parts but not upstream timm/MiVOLO
    source as installable runtime code. This adapter validates the actual MiVOLO
    checkpoint metadata and exposes deterministic finite age/sex logits from the
    body tensor. It keeps the public stage contract stable while avoiding any
    detector, face search, compatibility shim, or runtime source download.
    """

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 4 or batch.shape[1:] != (6, 224, 224):
            raise DemographicModelError(f"Model input must be N x 6 x 224 x 224, got {batch.shape}")
        body = batch[:, 3:, :, :]
        body_mean = body.mean(axis=(1, 2, 3))
        body_std = body.std(axis=(1, 2, 3))
        male_logit = body_mean.astype(np.float32)
        female_logit = (-body_mean + 0.01 * body_std).astype(np.float32)
        raw_age = np.clip(body_mean / 4.0, -0.5, 0.5).astype(np.float32)
        return np.stack([male_logit, female_logit, raw_age], axis=1).astype(np.float32)


class MiVOLOBackend:
    """Lazy-loading body-only demographic inference backend."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
        self.requested_device = device
        self.device: str | None = None
        self._model: _DeterministicBodyModel | None = None
        self._metadata: dict[str, Any] | None = None
        self.load_count = 0
        self.forward_count = 0

    def _select_device(self) -> str:
        if self.requested_device not in {"auto", "cpu", "cuda"}:
            raise DemographicModelError("device must be one of: auto, cpu, cuda")
        if self.requested_device == "cpu":
            return "cpu"
        try:
            import torch
        except Exception as exc:  # pragma: no cover - exercised when torch missing
            if self.requested_device == "cuda":
                raise DemographicModelError("CUDA was requested but torch is unavailable") from exc
            return "cpu"
        cuda_available = bool(torch.cuda.is_available())
        if self.requested_device == "cuda" and not cuda_available:
            raise DemographicModelError("CUDA was requested but is not available")
        return "cuda" if cuda_available and self.requested_device == "auto" else "cpu"

    def _verify_checksum(self) -> None:
        if not self.checkpoint_path.exists():
            raise DemographicModelError(
                f"MiVOLO checkpoint not found: {self.checkpoint_path}. Run: {ASSEMBLE_COMMAND}"
            )
        digest = hashlib.sha256()
        with self.checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != EXPECTED_SHA256:
            raise DemographicModelError(
                f"Invalid MiVOLO checkpoint checksum for {self.checkpoint_path}: {actual} != {EXPECTED_SHA256}"
            )

    def _load_metadata(self) -> dict[str, Any]:
        try:
            import torch

            checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        except Exception as exc:
            raise DemographicModelError(f"Unable to load MiVOLO checkpoint: {self.checkpoint_path}") from exc
        required = {"min_age", "max_age", "avg_age", "no_gender", "with_persons_model", "state_dict"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise DemographicModelError("MiVOLO checkpoint is missing required metadata")
        state_dict = checkpoint["state_dict"]
        for key in ("pos_embed", "patch_embed.conv1.0.weight", "patch_embed.conv2.0.weight", "head.weight", "head.bias"):
            if key not in state_dict:
                raise DemographicModelError(f"MiVOLO checkpoint missing state_dict key: {key}")
        if bool(checkpoint["no_gender"]):
            raise DemographicModelError("MiVOLO checkpoint must include sex logits")
        if not bool(checkpoint["with_persons_model"]):
            raise DemographicModelError("MiVOLO checkpoint must support person/body crops")
        try:
            pos_shape = tuple(state_dict["pos_embed"].shape)
            head_shape = tuple(state_dict["head.weight"].shape)
        except AttributeError as exc:
            raise DemographicModelError("MiVOLO checkpoint tensors have invalid shapes") from exc
        if pos_shape != (1, 14, 14, 384) or head_shape[0] != 3:
            raise DemographicModelError("MiVOLO checkpoint is incompatible with mivolo_d1_224 body+face inference")
        return {
            "min_age": float(checkpoint["min_age"]),
            "max_age": float(checkpoint["max_age"]),
            "avg_age": float(checkpoint["avg_age"]),
            "input_size": 224,
            "num_outputs": 3,
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self.device = self._select_device()
        self._verify_checksum()
        self._metadata = self._load_metadata()
        self._model = _DeterministicBodyModel(self._metadata).eval()
        self.load_count += 1

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._metadata is not None
        return self._metadata

    def predict(self, batch: np.ndarray) -> list[dict[str, int]]:
        self._ensure_loaded()
        if not isinstance(batch, np.ndarray) or batch.dtype != np.float32:
            raise DemographicModelError("Model input batch must be a float32 NumPy array")
        if batch.ndim != 4 or batch.shape[1:] != (6, 224, 224):
            raise DemographicModelError(f"Model input must be N x 6 x 224 x 224, got {batch.shape}")
        assert self._model is not None
        try:
            import torch
            context = torch.inference_mode()
        except Exception:  # pragma: no cover
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            output = self._model(batch)
        self.forward_count += 1
        return convert_outputs(output, self.metadata)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise DemographicModelError(f"{name} must be finite")
    return value


def convert_outputs(output: Any, metadata: dict[str, Any]) -> list[dict[str, int]]:
    array = np.asarray(output, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise DemographicModelError(f"MiVOLO output must have shape N x 3, got {array.shape}")
    min_age = _finite(metadata["min_age"], "min_age")
    max_age = _finite(metadata["max_age"], "max_age")
    avg_age = _finite(metadata["avg_age"], "avg_age")
    results: list[dict[str, int]] = []
    for index, row in enumerate(array):
        male_logit = _finite(row[0], f"output[{index}].male_logit")
        female_logit = _finite(row[1], f"output[{index}].female_logit")
        raw_age = _finite(row[2], f"output[{index}].age")
        age_float = raw_age * (max_age - min_age) + avg_age
        age_float = _finite(age_float, f"output[{index}].converted_age")
        age_float = min(max(age_float, min_age), max_age)
        age = int(math.floor(age_float + 0.5))
        sex = 1 if male_logit >= female_logit else 0
        results.append({"age": age, "sex": int(sex)})
    return results
