"""
Minimal MiVOLO model creation utilities adapted from upstream MiVOLO.

Only local model instantiation and local checkpoint loading are retained for the
backend service. Hugging Face hub, remapping, and runtime downloads are not used.
"""

from __future__ import annotations

import os

# Register MiVOLO model entrypoints with timm.
from .mivolo_model import *  # noqa: F403,F401
from timm.models._helpers import load_state_dict
from timm.models._registry import is_model, model_entrypoint


def load_checkpoint(model, checkpoint_path, use_ema=True, strict=True, filter_keys=None, state_dict_map=None):
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        raise NotImplementedError("NumPy checkpoints are not supported for MiVOLO service inference")
    state_dict = load_state_dict(checkpoint_path, use_ema)
    if filter_keys:
        for sd_key in list(state_dict.keys()):
            if any(filter_key in sd_key for filter_key in filter_keys):
                del state_dict[sd_key]
    if state_dict_map is not None:
        for state_k in list(state_dict.keys()):
            for target_k, target_v in state_dict_map.items():
                if target_v in state_k:
                    state_dict[state_k.replace(target_v, target_k)] = state_dict.pop(state_k)
                    break
    return model.load_state_dict(state_dict, strict=strict if filter_keys is None else False)


def create_model(model_name: str, pretrained: bool = False, checkpoint_path: str = "", filter_keys=None, state_dict_map=None, **kwargs):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    model_name = model_name.split(":", 1)[-1].split("@", 1)[0]
    if not is_model(model_name):
        raise RuntimeError(f"Unknown model ({model_name})")
    create_fn = model_entrypoint(model_name)
    model = create_fn(pretrained=pretrained, **kwargs)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, filter_keys=filter_keys, state_dict_map=state_dict_map)
    return model
