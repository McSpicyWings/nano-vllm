import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _get_param_or_buffer(model: nn.Module, name: str):
    """Get a nn.Parameter or registered buffer by its dotted name.
    Returns: (obj, kind): kind in {"param", "buffer"}.
    """
    try:
        return model.get_parameter(name), "param"
    except Exception:
        pass
    try:
        # PyTorch 2.0+: supports dotted names.
        return model.get_buffer(name), "buffer"
    except Exception:
        pass
    # Common prefix variants.
    for prefix in ("module.",):
        if name.startswith(prefix):
            return _get_param_or_buffer(model, name[len(prefix):])
    raise KeyError(name)


def _iter_safetensors_files(path: str) -> list[str]:
    return sorted(glob(os.path.join(path, "*.safetensors")))


def _iter_bin_files(path: str) -> list[str]:
    # HF convention: pytorch_model.bin (or sharded pytorch_model-00001-of-0000N.bin)
    return sorted(glob(os.path.join(path, "pytorch_model*.bin")))


def load_model(model: nn.Module, path: str, *, strict: bool = True):
    """Load weights into a model.
    Supports:
      - *.safetensors (preferred)
      - pytorch_model*.bin (fallback)
    Args:
        strict: if False, skips weights that don't exist in the model.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    def _load_one(weight_name: str, loaded_weight: torch.Tensor):
        # Packed weights mapping (e.g. q_proj/k_proj/v_proj -> qkv_proj shards)
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                try:
                    param, kind = _get_param_or_buffer(model, param_name)
                except KeyError:
                    if strict:
                        raise
                    return
                if kind != "param":
                    if strict:
                        raise KeyError(param_name)
                    return
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                return

        # Regular parameter/buffer.
        try:
            obj, kind = _get_param_or_buffer(model, weight_name)
        except KeyError:
            if strict:
                raise
            return

        if kind == "param":
            weight_loader = getattr(obj, "weight_loader", default_weight_loader)
            weight_loader(obj, loaded_weight)
        else:
            obj.data.copy_(loaded_weight)

    st_files = _iter_safetensors_files(path)
    if st_files:
        for file in st_files:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    _load_one(weight_name, f.get_tensor(weight_name))
        return

    bin_files = _iter_bin_files(path)
    if not bin_files:
        raise FileNotFoundError(f"No model weights found under: {path}")

    # Handle (rare) sharded *.bin too.
    for file in bin_files:
        state = torch.load(file, map_location="cpu")
        state_dict = state.get("state_dict", state)
        for weight_name, weight in state_dict.items():
            _load_one(weight_name, weight)