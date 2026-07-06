"""ESFT freeze/train adaptation for Qwen3_5Moe's *packed* expert tensors.

Why this differs from upstream ESFT
-----------------------------------
Upstream ESFT (DeepSeek-V2-Lite) freezes/trains individual expert *modules* by
converting their Parameters to buffers (``to_buffer``) or back (``to_param``). That
relies on each expert being a separate ``nn.Module`` (``mlp.experts[i].gate_proj``).

Qwen3_5Moe stores all experts of a layer in two packed Parameters:
``experts.gate_up_proj`` (E, 2*I, H) and ``experts.down_proj`` (E, H, I).
``requires_grad`` is per-Parameter, so we cannot freeze a subset of experts by
toggling it. Instead we:

  1. freeze everything, then set the two packed expert Parameters (of layers that
     have any selected expert) to ``requires_grad=True``;
  2. register a *gradient hook* on each packed Parameter that zeroes the gradient
     rows of non-selected experts (dim 0 is the expert axis);
  3. rely on the optimiser putting these packed params in a **weight_decay=0**
     group (see :func:`build_param_groups`). With zero grad AND zero decay, Adam's
     first/second moments for non-selected rows stay 0 and those rows are
     bit-exact frozen. (Upstream uses wd=0.1 on trained experts; we drop it to
     preserve the frozen-expert invariant that packed tensors would otherwise
     break. Documented in NOTES.md.)

The saved "patch" contains only the selected experts' weight slices, analogous to
ESFT saving only trained-expert modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

from .common import find_moe_blocks, MoeBlockRef

# safetensors is optional at import time; only needed for save/load.
try:
    from safetensors.torch import save_file as _st_save
    from safetensors.torch import load_file as _st_load
except Exception:  # pragma: no cover - exercised only when safetensors missing
    _st_save = None
    _st_load = None


@dataclass
class EsftHandles:
    """Bookkeeping returned by :func:`to_esft_qwen`.

    ``masks`` maps ``(layer_idx, "gate_up"|"down")`` -> bool mask over experts.
    ``hook_handles`` are the registered gradient hooks (call ``.remove()`` to undo).
    ``expert_params`` lists the packed Parameters that are being trained, so the
    caller can build a no-decay optimiser group.
    """

    expert_config: dict
    masks: dict = field(default_factory=dict)
    hook_handles: list = field(default_factory=list)
    expert_params: list = field(default_factory=list)
    selected: dict = field(default_factory=dict)  # layer_idx -> sorted list[int]

    def remove(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


def _selected_map(expert_config: dict) -> dict[int, list[int]]:
    return {int(k): sorted(int(e) for e in v) for k, v in expert_config["experts"].items()}


def _make_grad_hook(mask_1d: torch.Tensor):
    """Return a grad hook that zeroes non-selected expert rows (dim 0)."""

    def hook(grad):
        # grad shape: (E, *). Broadcast mask over the trailing dims.
        m = mask_1d.to(device=grad.device, dtype=grad.dtype)
        view = m.view(-1, *([1] * (grad.dim() - 1)))
        return grad * view

    return hook


def to_esft_qwen(
    model: torch.nn.Module,
    expert_config: dict,
    *,
    train_shared_experts: bool | None = None,
    train_non_expert_modules: bool | None = None,
) -> EsftHandles:
    """Configure ``model`` for ESFT training of the selected experts only.

    Freezes all parameters, then unfreezes and gradient-masks the packed expert
    tensors of the layers named in ``expert_config['experts']``. Optionally also
    trains shared experts / all non-expert modules (defaults taken from the config
    unless overridden).
    """
    if train_shared_experts is None:
        train_shared_experts = bool(expert_config.get("shared_experts", False))
    if train_non_expert_modules is None:
        train_non_expert_modules = bool(expert_config.get("non_expert_modules", False))

    selected = _selected_map(expert_config)

    # 1. Freeze everything (or leave non-expert modules trainable if requested).
    model.requires_grad_(train_non_expert_modules)

    handles = EsftHandles(expert_config=expert_config, selected=selected)

    refs = {r.layer_idx: r for r in find_moe_blocks(model)}
    for layer_idx, expert_ids in selected.items():
        if layer_idx not in refs:
            raise KeyError(f"layer {layer_idx} in expert_config not found in model")
        ref: MoeBlockRef = refs[layer_idx]
        experts = ref.experts
        num_experts = int(experts.gate_up_proj.shape[0])
        mask = torch.zeros(num_experts, dtype=torch.bool)
        if expert_ids:
            mask[torch.tensor(expert_ids, dtype=torch.long)] = True

        for pname in ("gate_up_proj", "down_proj"):
            param = getattr(experts, pname)
            param.requires_grad_(True)
            key = "gate_up" if pname == "gate_up_proj" else "down"
            handles.masks[(layer_idx, key)] = mask
            handles.hook_handles.append(param.register_hook(_make_grad_hook(mask)))
            handles.expert_params.append(param)

        # The router, shared_expert_gate stay frozen (ESFT trains only expert FFN).
        if train_shared_experts and ref.shared_expert is not None:
            ref.shared_expert.requires_grad_(True)

    return handles


def build_param_groups(model: torch.nn.Module, handles: EsftHandles, weight_decay: float = 0.0):
    """Optimiser param groups: packed expert params get weight_decay=0 to keep
    non-selected experts bit-exact frozen; everything else trainable uses
    ``weight_decay``.
    """
    expert_ids = {id(p) for p in handles.expert_params}
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in expert_ids:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


# --------------------------------------------------------------------------- #
# Patch save / load: store only the selected experts' weight slices.
# --------------------------------------------------------------------------- #

def _patch_key(layer_idx: int, which: str, expert_id: int) -> str:
    return f"layers.{layer_idx}.experts.{which}.{expert_id}"


def save_expert_patch(model: torch.nn.Module, expert_config: dict, path: str) -> dict:
    """Save only the selected experts' ``gate_up_proj``/``down_proj`` slices.

    Keys: ``layers.{L}.experts.{gate_up|down}.{E}``. The expert config is stored in
    the safetensors metadata so the patch is self-describing.
    """
    if _st_save is None:
        raise RuntimeError("safetensors is required for save_expert_patch")
    selected = _selected_map(expert_config)
    refs = {r.layer_idx: r for r in find_moe_blocks(model)}
    tensors: dict[str, torch.Tensor] = {}
    for layer_idx, expert_ids in selected.items():
        experts = refs[layer_idx].experts
        gup = experts.gate_up_proj.detach().cpu()
        dwn = experts.down_proj.detach().cpu()
        for e in expert_ids:
            tensors[_patch_key(layer_idx, "gate_up", e)] = gup[e].contiguous().clone()
            tensors[_patch_key(layer_idx, "down", e)] = dwn[e].contiguous().clone()
    metadata = {"expert_config": json.dumps(expert_config), "format": "esft-qwen-patch-v1"}
    _st_save(tensors, path, metadata=metadata)
    return {"num_tensors": len(tensors), "path": path}


def load_expert_patch(model: torch.nn.Module, path: str, *, strict: bool = True) -> dict:
    """Write patch slices back into the model's packed expert tensors in place."""
    if _st_load is None:
        raise RuntimeError("safetensors is required for load_expert_patch")
    tensors = _st_load(path)
    refs = {r.layer_idx: r for r in find_moe_blocks(model)}
    n_written = 0
    with torch.no_grad():
        for key, val in tensors.items():
            # key: layers.{L}.experts.{which}.{E}
            _, layer_s, _, which, expert_s = key.split(".")
            layer_idx, expert_id = int(layer_s), int(expert_s)
            if layer_idx not in refs:
                if strict:
                    raise KeyError(f"patch references missing layer {layer_idx}")
                continue
            experts = refs[layer_idx].experts
            target = experts.gate_up_proj if which == "gate_up" else experts.down_proj
            target[expert_id].copy_(val.to(device=target.device, dtype=target.dtype))
            n_written += 1
    return {"num_written": n_written, "path": path}
