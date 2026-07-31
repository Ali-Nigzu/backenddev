"""Private MiVOLO checkpoint loading, inference, and output conversion."""

from __future__ import annotations

import importlib
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import DemographicModelError

_CHECKPOINT_PATH = Path(__file__).with_name("demographicweights.pth")
_REQUIRED_CHECKPOINT_FIELDS = {"state_dict", "min_age", "max_age", "avg_age", "no_gender", "with_persons_model"}
_IGNORED_STATE_PREFIXES = ("fds.",)
_EXPECTED_INPUT_SIZE = 224
_EXPECTED_OUTPUTS = 3
_MIN_REALISTIC_PARAMETER_COUNT = 1_000_000
_CPU_CHUNK_SIZE = 16
_CUDA_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _ModelMetadata:
    min_age: float
    max_age: float
    avg_age: float
    parameter_count: int


class _MiVOLOModelRunner:
    """Lazy, process-local runner for the fixed body/person MiVOLO D1-224 model."""

    def __init__(self) -> None:
        self._init_lock = threading.Lock()
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._metadata: _ModelMetadata | None = None

    def predict(self, descriptors: Sequence[Any], frames_by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, int]]:
        if not descriptors:
            return []

        from .preprocessing import crop_body, mivolo_input_from_body_crop, stack_mivolo_inputs

        self._ensure_loaded()
        assert self._metadata is not None
        assert self._device is not None

        results: list[dict[str, int]] = []
        chunk_size = _CUDA_CHUNK_SIZE if self._device == "cuda" else _CPU_CHUNK_SIZE
        for start in range(0, len(descriptors), chunk_size):
            chunk = descriptors[start : start + chunk_size]
            prepared = []
            for descriptor in chunk:
                frame = frames_by_id[descriptor.frame_id]
                crop = crop_body(frame["image"], descriptor.bbox, descriptor.track_id, descriptor.frame_id)
                prepared.append(mivolo_input_from_body_crop(crop))
            batch = stack_mivolo_inputs(prepared)
            output = self._forward(batch)
            results.extend(_convert_outputs(output, self._metadata))
        return results

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return
            self._load()

    def _load(self) -> None:
        if not _CHECKPOINT_PATH.exists():
            raise DemographicModelError(
                f"MiVOLO checkpoint not found at {_CHECKPOINT_PATH}. "
                "Production artifacts must include demographics/demographicweights.pth."
            )
        torch = _import_module("torch", "Unable to import torch for MiVOLO inference")
        device = _select_device(torch)
        create_model = _load_model_factory()
        checkpoint = _read_checkpoint(torch)
        metadata, state_dict = _metadata_and_state_dict(checkpoint)
        model = _instantiate_model(create_model)
        metadata = _strict_load_state_dict(model, metadata, state_dict)
        try:
            model = model.to(device)
            model.eval()
        except Exception as exc:  # pragma: no cover - depends on torch runtime/device
            raise DemographicModelError(f"Unable to move MiVOLO model to device {device}") from exc
        self._torch = torch
        self._device = device
        self._metadata = metadata
        self._model = model

    def _forward(self, batch: Any) -> Any:
        assert self._torch is not None
        assert self._model is not None
        assert self._device is not None
        torch = self._torch
        try:
            tensor = torch.from_numpy(batch).to(self._device)
            with torch.inference_mode():
                output = self._model(tensor)
        except Exception as exc:
            raise DemographicModelError("MiVOLO inference failed") from exc
        return output


def _import_module(module_name: str, error_message: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise DemographicModelError(error_message) from exc


def _select_device(torch: Any) -> str:
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _load_model_factory() -> Any:
    try:
        module = importlib.import_module("demographics._mivolo.model.mivolo_model")
        return module.create_mivolo_d1_224
    except Exception as exc:
        raise DemographicModelError("Unable to import minimal MiVOLO D1-224 architecture") from exc


def _read_checkpoint(torch: Any) -> dict[str, Any]:
    try:
        checkpoint = torch.load(str(_CHECKPOINT_PATH), map_location="cpu")
    except Exception as exc:
        raise DemographicModelError(f"Unable to load MiVOLO checkpoint at {_CHECKPOINT_PATH}") from exc
    if not isinstance(checkpoint, dict):
        raise DemographicModelError("MiVOLO checkpoint must be a dictionary")
    missing = _REQUIRED_CHECKPOINT_FIELDS - set(checkpoint)
    if missing:
        raise DemographicModelError(f"MiVOLO checkpoint missing required fields: {sorted(missing)}")
    return checkpoint


def _metadata_and_state_dict(checkpoint: dict[str, Any]) -> tuple[_ModelMetadata, dict[str, Any]]:
    if bool(checkpoint["no_gender"]):
        raise DemographicModelError("MiVOLO checkpoint must include gender logits")
    if not bool(checkpoint["with_persons_model"]):
        raise DemographicModelError("MiVOLO checkpoint must support person/body crops")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict):
        raise DemographicModelError("MiVOLO checkpoint state_dict must be a dictionary")
    for key in ("pos_embed", "head.weight", "head.bias", "patch_embed.conv1.0.weight", "patch_embed.conv2.0.weight"):
        if key not in state_dict:
            raise DemographicModelError(f"MiVOLO checkpoint missing state_dict key: {key}")
    try:
        pos_shape = tuple(state_dict["pos_embed"].shape)
        head_shape = tuple(state_dict["head.weight"].shape)
    except AttributeError as exc:
        raise DemographicModelError("MiVOLO checkpoint tensors have invalid shapes") from exc
    if len(pos_shape) != 4 or int(pos_shape[1]) * 16 != _EXPECTED_INPUT_SIZE:
        raise DemographicModelError(f"MiVOLO checkpoint must be D1-224 compatible, got pos_embed shape {pos_shape}")
    if len(head_shape) != 2 or int(head_shape[0]) != _EXPECTED_OUTPUTS:
        raise DemographicModelError(f"MiVOLO head.weight must produce 3 outputs, got {head_shape}")
    min_age = _finite(checkpoint["min_age"], "min_age")
    max_age = _finite(checkpoint["max_age"], "max_age")
    avg_age = _finite(checkpoint["avg_age"], "avg_age")
    if max_age <= min_age:
        raise DemographicModelError("max_age must be greater than min_age")
    return _ModelMetadata(min_age=min_age, max_age=max_age, avg_age=avg_age, parameter_count=0), state_dict


def _instantiate_model(create_model: Any) -> Any:
    try:
        return create_model(num_classes=_EXPECTED_OUTPUTS, in_chans=6)
    except Exception as exc:
        raise DemographicModelError("Unable to instantiate MiVOLO D1-224 model") from exc


def _strict_load_state_dict(model: Any, metadata: _ModelMetadata, state_dict: dict[str, Any]) -> _ModelMetadata:
    loadable = {key: value for key, value in state_dict.items() if not key.startswith(_IGNORED_STATE_PREFIXES)}
    try:
        incompatible = model.load_state_dict(loadable, strict=True)
    except Exception as exc:
        raise DemographicModelError("Unable to strictly load MiVOLO checkpoint state_dict") from exc
    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if missing:
        raise DemographicModelError(f"MiVOLO checkpoint missing model parameters: {list(missing)}")
    if unexpected:
        raise DemographicModelError(f"MiVOLO checkpoint has unexpected model parameters: {list(unexpected)}")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameter_count < _MIN_REALISTIC_PARAMETER_COUNT:
        raise DemographicModelError(f"MiVOLO parameter count is not realistic: {parameter_count}")
    return _ModelMetadata(
        min_age=metadata.min_age,
        max_age=metadata.max_age,
        avg_age=metadata.avg_age,
        parameter_count=parameter_count,
    )


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise DemographicModelError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise DemographicModelError(f"{name} must be finite")
    return number


def _convert_outputs(output: Any, metadata: _ModelMetadata) -> list[dict[str, int]]:
    try:
        array = output.detach().cpu().numpy() if hasattr(output, "detach") else output
        numpy = importlib.import_module("numpy")
        array = numpy.asarray(array, dtype=numpy.float32)
    except Exception as exc:
        raise DemographicModelError("Unable to convert MiVOLO output to NumPy") from exc
    if array.ndim != 2 or array.shape[1] != _EXPECTED_OUTPUTS:
        raise DemographicModelError(f"MiVOLO output must have shape N x 3, got {array.shape}")
    results: list[dict[str, int]] = []
    for index, row in enumerate(array):
        male_logit = _finite(row[0], f"output[{index}].male_logit")
        female_logit = _finite(row[1], f"output[{index}].female_logit")
        raw_age = _finite(row[2], f"output[{index}].age")
        age_float = raw_age * (metadata.max_age - metadata.min_age) + metadata.avg_age
        age = int(math.floor(_finite(age_float, f"output[{index}].converted_age") + 0.5))
        sex = 1 if male_logit >= female_logit else 0
        results.append({"age": age, "sex": int(sex)})
    return results
