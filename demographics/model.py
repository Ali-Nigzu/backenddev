"""MiVOLO checkpoint loading and output conversion."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import DemographicModelError

DEFAULT_CHECKPOINT_PATH = Path(__file__).with_name("demographicweights.pth")
REQUIRED_CHECKPOINT_FIELDS = {"state_dict", "min_age", "max_age", "avg_age", "no_gender", "with_persons_model"}
IGNORED_UNEXPECTED_PREFIXES = ("fds.",)
MIN_REALISTIC_PARAMETER_COUNT = 1_000_000


@dataclass(frozen=True)
class MiVOLOMetadata:
    min_age: float
    max_age: float
    avg_age: float
    input_size: int
    num_outputs: int
    model_name: str
    parameter_count: int


@dataclass(frozen=True)
class LoadDiagnostics:
    model_class: str
    parameter_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    matched_parameter_name: str
    checkpoint_path: str
    device: str


class MiVOLOBackend:
    """Lazy-loading body-only demographic inference backend using real MiVOLO."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH
        self.requested_device = device
        self.device: str | None = None
        self._torch: Any | None = None
        self._model: Any | None = None
        self._metadata: MiVOLOMetadata | None = None
        self._diagnostics: LoadDiagnostics | None = None
        self.load_count = 0
        self.forward_count = 0

    def _select_device(self, torch: Any) -> str:
        if self.requested_device not in {"auto", "cpu", "cuda"}:
            raise DemographicModelError("device must be one of: auto, cpu, cuda")
        if self.requested_device == "cpu":
            return "cpu"
        cuda_available = bool(torch.cuda.is_available())
        if self.requested_device == "cuda" and not cuda_available:
            raise DemographicModelError("CUDA was requested but is not available")
        return "cuda" if cuda_available and self.requested_device == "auto" else "cpu"

    def _load_torch(self) -> Any:
        try:
            return importlib.import_module("torch")
        except Exception as exc:
            raise DemographicModelError("Unable to import torch for MiVOLO inference") from exc

    def _load_create_model(self) -> Any:
        try:
            module = importlib.import_module("demographics._mivolo.model.create_timm_model")
        except Exception as exc:
            raise DemographicModelError("Unable to import vendored MiVOLO model implementation") from exc
        return module.create_model

    def _verify_checkpoint_exists(self) -> None:
        if not self.checkpoint_path.exists():
            raise DemographicModelError(
                f"MiVOLO checkpoint not found at {self.checkpoint_path}. "
                "Place the full checkpoint at that path before running demographic inference."
            )

    def _read_checkpoint(self, torch: Any) -> dict[str, Any]:
        self._verify_checkpoint_exists()
        try:
            checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        except Exception as exc:
            raise DemographicModelError(f"Unable to load MiVOLO checkpoint at {self.checkpoint_path}") from exc
        if not isinstance(checkpoint, dict):
            raise DemographicModelError("MiVOLO checkpoint must be a dictionary")
        missing = REQUIRED_CHECKPOINT_FIELDS - set(checkpoint)
        if missing:
            raise DemographicModelError(f"MiVOLO checkpoint missing required fields: {sorted(missing)}")
        if bool(checkpoint["no_gender"]):
            raise DemographicModelError("MiVOLO checkpoint must include gender logits")
        if not bool(checkpoint["with_persons_model"]):
            raise DemographicModelError("MiVOLO checkpoint must support person/body crops")
        if not isinstance(checkpoint["state_dict"], dict):
            raise DemographicModelError("MiVOLO checkpoint state_dict must be a dictionary")
        return checkpoint

    def _metadata_from_checkpoint(self, checkpoint: dict[str, Any]) -> tuple[MiVOLOMetadata, dict[str, Any]]:
        state_dict = checkpoint["state_dict"]
        for key in ("pos_embed", "head.weight", "head.bias", "patch_embed.conv1.0.weight", "patch_embed.conv2.0.weight"):
            if key not in state_dict:
                raise DemographicModelError(f"MiVOLO checkpoint missing state_dict key: {key}")
        try:
            pos_shape = tuple(state_dict["pos_embed"].shape)
            head_shape = tuple(state_dict["head.weight"].shape)
        except AttributeError as exc:
            raise DemographicModelError("MiVOLO checkpoint tensors have invalid shapes") from exc
        if len(pos_shape) != 4:
            raise DemographicModelError(f"MiVOLO pos_embed must have rank 4, got {pos_shape}")
        input_size = int(pos_shape[1]) * 16
        if input_size not in {224, 384, 448, 512}:
            raise DemographicModelError(f"Unsupported MiVOLO input size inferred from checkpoint: {input_size}")
        if len(head_shape) != 2 or int(head_shape[0]) != 3:
            raise DemographicModelError(f"MiVOLO head.weight must produce 3 outputs, got {head_shape}")
        model_name = f"mivolo_d1_{input_size}"
        metadata = MiVOLOMetadata(
            min_age=_finite(checkpoint["min_age"], "min_age"),
            max_age=_finite(checkpoint["max_age"], "max_age"),
            avg_age=_finite(checkpoint["avg_age"], "avg_age"),
            input_size=input_size,
            num_outputs=3,
            model_name=model_name,
            parameter_count=0,
        )
        return metadata, state_dict

    def _instantiate_model(self, create_model: Any, metadata: MiVOLOMetadata) -> Any:
        try:
            return create_model(
                model_name=metadata.model_name,
                num_classes=metadata.num_outputs,
                in_chans=6,
                pretrained=False,
            )
        except Exception as exc:
            raise DemographicModelError(f"Unable to instantiate MiVOLO model {metadata.model_name}") from exc

    def _load_state_dict(self, torch: Any, model: Any, metadata: MiVOLOMetadata, state_dict: dict[str, Any]) -> MiVOLOMetadata:
        loadable_state_dict = {key: value for key, value in state_dict.items() if not key.startswith(IGNORED_UNEXPECTED_PREFIXES)}
        try:
            incompatible = model.load_state_dict(loadable_state_dict, strict=True)
        except Exception as exc:
            raise DemographicModelError("Unable to load MiVOLO checkpoint state_dict into the real model") from exc
        missing_keys = tuple(getattr(incompatible, "missing_keys", ()))
        unexpected_keys = tuple(getattr(incompatible, "unexpected_keys", ()))
        if missing_keys:
            raise DemographicModelError(f"MiVOLO checkpoint missing model parameters: {list(missing_keys)}")
        if unexpected_keys:
            raise DemographicModelError(f"MiVOLO checkpoint has unexpected model parameters: {list(unexpected_keys)}")
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        if parameter_count < MIN_REALISTIC_PARAMETER_COUNT:
            raise DemographicModelError(f"MiVOLO parameter count is not realistic: {parameter_count}")
        model_state = model.state_dict()
        matched_name = "head.weight"
        if matched_name not in model_state:
            raise DemographicModelError(f"MiVOLO model is missing expected parameter: {matched_name}")
        if not torch.equal(model_state[matched_name].detach().cpu(), state_dict[matched_name].detach().cpu()):
            raise DemographicModelError(f"Loaded MiVOLO parameter does not match checkpoint tensor: {matched_name}")
        self._diagnostics = LoadDiagnostics(
            model_class=f"{model.__class__.__module__}.{model.__class__.__name__}",
            parameter_count=parameter_count,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            matched_parameter_name=matched_name,
            checkpoint_path=str(self.checkpoint_path),
            device=str(self.device),
        )
        return MiVOLOMetadata(
            min_age=metadata.min_age,
            max_age=metadata.max_age,
            avg_age=metadata.avg_age,
            input_size=metadata.input_size,
            num_outputs=metadata.num_outputs,
            model_name=metadata.model_name,
            parameter_count=parameter_count,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._verify_checkpoint_exists()
        torch = self._load_torch()
        self.device = self._select_device(torch)
        create_model = self._load_create_model()
        checkpoint = self._read_checkpoint(torch)
        metadata, state_dict = self._metadata_from_checkpoint(checkpoint)
        model = self._instantiate_model(create_model, metadata)
        metadata = self._load_state_dict(torch, model, metadata, state_dict)
        try:
            model = model.to(self.device)
            model.eval()
        except Exception as exc:
            raise DemographicModelError(f"Unable to move MiVOLO model to device {self.device}") from exc
        self._torch = torch
        self._model = model
        self._metadata = metadata
        self.load_count += 1

    def load(self) -> None:
        """Load and validate the real MiVOLO model for non-empty inference."""

        self._ensure_loaded()

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._metadata is not None
        return {
            "min_age": self._metadata.min_age,
            "max_age": self._metadata.max_age,
            "avg_age": self._metadata.avg_age,
            "input_size": self._metadata.input_size,
            "num_outputs": self._metadata.num_outputs,
            "model_name": self._metadata.model_name,
            "parameter_count": self._metadata.parameter_count,
        }

    @property
    def diagnostics(self) -> LoadDiagnostics:
        self._ensure_loaded()
        assert self._diagnostics is not None
        return self._diagnostics

    def predict(self, batch: np.ndarray) -> list[dict[str, int]]:
        if not isinstance(batch, np.ndarray) or batch.dtype != np.float32:
            raise DemographicModelError("Model input batch must be a float32 NumPy array")
        if batch.ndim != 4 or batch.shape[1:] != (6, 224, 224):
            raise DemographicModelError(f"Model input must be N x 6 x 224 x 224, got {batch.shape}")
        if batch.shape[0] == 0:
            return []
        self._ensure_loaded()
        assert self._torch is not None
        assert self._model is not None
        torch = self._torch
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            with torch.inference_mode():
                output = self._model(tensor)
        except Exception as exc:
            raise DemographicModelError("MiVOLO inference failed") from exc
        self.forward_count += 1
        return convert_outputs(output, self.metadata)


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise DemographicModelError(f"{name} must be finite")
    return value


def convert_outputs(output: Any, metadata: dict[str, Any]) -> list[dict[str, int]]:
    try:
        array = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output, dtype=np.float32)
    except Exception as exc:
        raise DemographicModelError("Unable to convert MiVOLO output to NumPy") from exc
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise DemographicModelError(f"MiVOLO output must have shape N x 3, got {array.shape}")
    min_age = _finite(metadata["min_age"], "min_age")
    max_age = _finite(metadata["max_age"], "max_age")
    avg_age = _finite(metadata["avg_age"], "avg_age")
    if max_age <= min_age:
        raise DemographicModelError("max_age must be greater than min_age")
    results: list[dict[str, int]] = []
    for index, row in enumerate(array):
        male_logit = _finite(row[0], f"output[{index}].male_logit")
        female_logit = _finite(row[1], f"output[{index}].female_logit")
        raw_age = _finite(row[2], f"output[{index}].age")
        age_float = raw_age * (max_age - min_age) + avg_age
        age_float = _finite(age_float, f"output[{index}].converted_age")
        age = int(math.floor(age_float + 0.5))
        sex = 1 if male_logit >= female_logit else 0
        results.append({"age": age, "sex": int(sex)})
    return results
