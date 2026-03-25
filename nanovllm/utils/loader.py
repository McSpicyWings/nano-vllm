import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded_any = False
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
        loaded_any = True

    bin_path = os.path.join(path, "pytorch_model.bin")
    if not loaded_any and os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cuda")

        # Special-case Eagle3 draft checkpoints that ship split Q/K/V and gate/up weights.
        if "midlayer.self_attn.q_proj.weight" in state_dict:
            q = state_dict.pop("midlayer.self_attn.q_proj.weight")
            k = state_dict.pop("midlayer.self_attn.k_proj.weight")
            v = state_dict.pop("midlayer.self_attn.v_proj.weight")
            qkv = torch.cat([q, k, v], dim=0)
            model.get_parameter("model.layers.0.self_attn.qkv_proj.weight").data.copy_(qkv)

            o = state_dict.pop("midlayer.self_attn.o_proj.weight")
            model.get_parameter("model.layers.0.self_attn.o_proj.weight").data.copy_(o)

            gate = state_dict.pop("midlayer.mlp.gate_proj.weight")
            up = state_dict.pop("midlayer.mlp.up_proj.weight")
            gate_up = torch.cat([gate, up], dim=0)
            model.get_parameter("model.layers.0.mlp.gate_up_proj.weight").data.copy_(gate_up)

            down = state_dict.pop("midlayer.mlp.down_proj.weight")
            model.get_parameter("model.layers.0.mlp.down_proj.weight").data.copy_(down)

            for name in ("hidden_norm", "input_layernorm", "post_attention_layernorm"):
                weight = state_dict.pop(f"midlayer.{name}.weight")
                model.get_parameter(f"model.layers.0.{name}.weight").data.copy_(weight)

            if "norm.weight" in state_dict:
                model.get_parameter("model.norm.weight").data.copy_(state_dict.pop("norm.weight"))
            if "fc.weight" in state_dict:
                model.get_parameter("fc.weight").data.copy_(state_dict.pop("fc.weight"))
            if "lm_head.weight" in state_dict:
                model.get_parameter("lm_head.weight").data.copy_(state_dict.pop("lm_head.weight"))

            # Optional vocab mappings: d2t maps draft->target ids, t2d maps target->draft ids.
            if hasattr(model, "set_vocab_mapping") and "d2t" in state_dict:
                model.set_vocab_mapping(state_dict.pop("d2t").long())
            # t2d is not currently consumed; keep as attribute for debugging if present.
            if "t2d" in state_dict:
                setattr(model, "t2d", state_dict.pop("t2d"))

            loaded_any = True

        else:
            for weight_name, tensor in state_dict.items():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, tensor, shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except Exception:
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
            loaded_any = True

    if not loaded_any:
        raise FileNotFoundError(f"No model weights found in {path}")
