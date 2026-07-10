"""Residual-delta ESFT for Qwen3_5Moe packed experts (memory-frugal training path).

Why this replaces the grad-hook masking of ``esft_patch.to_esft_qwen`` for training
-----------------------------------------------------------------------------------
The mask-hook approach sets the *whole* packed tensors (E=256 experts) to
``requires_grad=True`` and zeroes non-selected gradient rows in a hook. Autograd
still **materialises the full gradient of every packed tensor simultaneously**
(~32B params -> 64GB bf16 for the 40-layer k32/p0.2 config), which OOMs before the
hooks run. Optimiser tricks (Adafactor) cannot help: they shrink state, not grads.

Here the packed tensors stay frozen (``requires_grad=False``) and each selected
layer gets two small zero-initialised Parameters:

    delta_gate_up : (n_sel, 2*I, H)     delta_down : (n_sel, H, I)

The experts module's forward is wrapped per instance: it builds

    eff = packed.index_add(0, sel_ids, delta)        # out-of-place

briefly shadows ``self.gate_up_proj``/``self.down_proj`` with the effective
tensors via the instance ``__dict__`` (registered Parameters live in
``_parameters``, so a plain instance attribute wins the lookup), and delegates to
the *class* forward -- i.e. the transformers ``use_experts_implementation``
dispatcher, so the fast ``grouped_mm`` kernels keep being used. Consequences:

  * zero-init  => eff == packed bit-exact at start (x + 0.0 == x);
  * autograd sees ``index_add(frozen, ids, delta)``: the full-size gradient of
    ``eff`` exists only transiently for the ONE layer being recomputed under
    gradient checkpointing (~1.6GB), then collapses into the small delta grad via
    index_select. Steady-state grads are ~2B params (the deltas), not 32B;
  * bit-exact freezing is structural: the packed Parameters are never written
    (index_add is out-of-place, the optimiser only holds delta params);
  * every delta joins the autograd graph every microbatch (index_add always
    executes), so DDP works with ``find_unused_parameters=False``;
  * the whole 35B model + delta grads + AdamW state fit on ONE 96GB GPU, so both
    GPUs run plain DDP (torchrun --nproc_per_node=2) instead of the half-idle
    pipeline that device_map="auto" gave.

The saved patch keeps the ``esft-qwen-patch-v1`` format (*effective* slices
``packed[e] + delta[row]``), so ``esft_patch.load_expert_patch`` and the whole
eval harness work unchanged.
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass, field

import torch
from torch import nn

from .common import find_moe_blocks, MoeBlockRef
from .esft_patch import _selected_map, _patch_key

try:
    from safetensors.torch import save_file as _st_save
    from safetensors.torch import load_file as _st_load
    from safetensors import safe_open as _st_open
except Exception:  # pragma: no cover
    _st_save = None
    _st_load = None
    _st_open = None

DELTA_STATE_NAME = "delta_state.safetensors"


@dataclass
class DeltaHandles:
    """Bookkeeping returned by :func:`to_esft_delta`.

    ``expert_params`` lists the delta Parameters (same field name as EsftHandles so
    ``build_param_groups`` works unchanged). ``experts_modules`` maps layer_idx ->
    the patched ``Qwen3_5MoeExperts`` module.
    """

    expert_config: dict
    selected: dict = field(default_factory=dict)   # layer_idx -> sorted list[int]
    expert_params: list = field(default_factory=list)
    router_params: list = field(default_factory=list)  # unfrozen gate params (router-mobile mode)
    experts_modules: dict = field(default_factory=dict)


def _delta_wrapped_forward(self, hidden_states, top_k_index, top_k_weights):
    """Build effective packed tensors (frozen + delta on selected rows), shadow the
    Parameter attributes for the duration of the call, and delegate to the class
    forward (the transformers experts-implementation dispatcher)."""
    packed_gu = self._parameters["gate_up_proj"]
    packed_dn = self._parameters["down_proj"]
    eff_gu = packed_gu.index_add(0, self.delta_ids, self.delta_gate_up)
    eff_dn = packed_dn.index_add(0, self.delta_ids, self.delta_down)
    object.__setattr__(self, "gate_up_proj", eff_gu)   # nn.Module.__setattr__ would refuse
    object.__setattr__(self, "down_proj", eff_dn)
    try:
        return type(self).forward(self, hidden_states, top_k_index, top_k_weights)
    finally:
        # Drop the shadows so attribute lookup falls back to the frozen Parameters
        # and the effective tensors are freed as soon as autograd allows.
        self.__dict__.pop("gate_up_proj", None)
        self.__dict__.pop("down_proj", None)


def to_esft_delta(model: torch.nn.Module, expert_config: dict) -> DeltaHandles:
    """Freeze the whole model and register trainable residual deltas for the
    selected experts of each layer in ``expert_config``."""
    selected = _selected_map(expert_config)
    model.requires_grad_(False)

    handles = DeltaHandles(expert_config=expert_config, selected=selected)
    refs = {r.layer_idx: r for r in find_moe_blocks(model)}
    for layer_idx, expert_ids in selected.items():
        if layer_idx not in refs:
            raise KeyError(f"layer {layer_idx} in expert_config not found in model")
        ref: MoeBlockRef = refs[layer_idx]
        experts = ref.experts
        if "gate_up_proj" not in experts._parameters or "down_proj" not in experts._parameters:
            raise AttributeError(f"experts module of layer {layer_idx} lacks packed Parameters")
        gup, dwn = experts.gate_up_proj, experts.down_proj
        num_experts = int(gup.shape[0])
        bad = [e for e in expert_ids if not (0 <= e < num_experts)]
        if bad:
            raise ValueError(f"layer {layer_idx}: expert ids out of range: {bad}")

        n_sel = len(expert_ids)
        delta_gu = nn.Parameter(torch.zeros(n_sel, *gup.shape[1:], dtype=gup.dtype, device=gup.device))
        delta_dn = nn.Parameter(torch.zeros(n_sel, *dwn.shape[1:], dtype=dwn.dtype, device=dwn.device))
        experts.register_parameter("delta_gate_up", delta_gu)
        experts.register_parameter("delta_down", delta_dn)
        # row order == sorted expert ids (matches _selected_map / patch save order)
        experts.register_buffer(
            "delta_ids",
            torch.tensor(expert_ids, dtype=torch.long, device=gup.device),
            persistent=False)
        experts._delta_expert_ids = list(expert_ids)
        experts.forward = types.MethodType(_delta_wrapped_forward, experts)

        handles.expert_params.extend([delta_gu, delta_dn])
        handles.experts_modules[layer_idx] = experts

    return handles


# --------------------------------------------------------------------------- #
# Checkpoint (delta-only) save / load
# --------------------------------------------------------------------------- #

def delta_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()
            if n.endswith(("delta_gate_up", "delta_down"))}


def save_delta_state(model: torch.nn.Module, path: str) -> dict:
    if _st_save is None:
        raise RuntimeError("safetensors required")
    sd = delta_state_dict(model)
    _st_save(sd, path, metadata={"format": "esft-qwen-delta-v1"})
    return {"num_tensors": len(sd), "path": path}


def load_delta_state(model: torch.nn.Module, path: str) -> dict:
    if _st_load is None:
        raise RuntimeError("safetensors required")
    sd = _st_load(path)
    own = {n: p for n, p in model.named_parameters()
           if n.endswith(("delta_gate_up", "delta_down"))}
    missing = set(own) - set(sd)
    unexpected = set(sd) - set(own)
    if missing or unexpected:
        raise KeyError(f"delta state mismatch: missing={sorted(missing)[:3]}... "
                       f"unexpected={sorted(unexpected)[:3]}...")
    with torch.no_grad():
        for n, p in own.items():
            p.copy_(sd[n].to(device=p.device, dtype=p.dtype))
    return {"num_written": len(own), "path": path}


# --------------------------------------------------------------------------- #
# Patch save (v1 format: EFFECTIVE slices) -- compatible with load_expert_patch
# --------------------------------------------------------------------------- #

def save_expert_patch_delta(model: torch.nn.Module, handles: DeltaHandles, path: str) -> dict:
    if _st_save is None:
        raise RuntimeError("safetensors required")
    tensors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for layer_idx, expert_ids in handles.selected.items():
            experts = handles.experts_modules[layer_idx]
            for i, e in enumerate(expert_ids):
                eff_gu = (experts.gate_up_proj[e] + experts.delta_gate_up[i]).detach().cpu().contiguous()
                eff_dn = (experts.down_proj[e] + experts.delta_down[i]).detach().cpu().contiguous()
                tensors[_patch_key(layer_idx, "gate_up", e)] = eff_gu
                tensors[_patch_key(layer_idx, "down", e)] = eff_dn
    metadata = {"expert_config": json.dumps(handles.expert_config),
                "format": "esft-qwen-patch-v1", "trained_as": "residual-delta"}
    _st_save(tensors, path, metadata=metadata)
    return {"num_tensors": len(tensors), "path": path}


# --------------------------------------------------------------------------- #
# Frozen-weight verification against the on-disk base checkpoint
# --------------------------------------------------------------------------- #

def _disk_key_index(model_dir: str) -> dict[str, str]:
    """Map state-dict key -> shard filename from model.safetensors.index.json."""
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return json.load(f)["weight_map"]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        if _st_open is None:
            raise RuntimeError("safetensors required")
        with _st_open(single, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"no safetensors index found in {model_dir}")


def _bitwise_equal_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-expert-row bitwise equality for (E, ...) bf16/fp16/fp32 tensors."""
    assert a.shape == b.shape and a.dtype == b.dtype
    int_dtype = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
                 torch.float32: torch.int32}[a.dtype]
    av = a.contiguous().view(int_dtype).reshape(a.shape[0], -1)
    bv = b.contiguous().view(int_dtype).reshape(b.shape[0], -1)
    return (av == bv).all(dim=1)  # (E,) bool


def verify_frozen_vs_disk(model: torch.nn.Module, model_dir: str,
                          handles: DeltaHandles) -> dict:
    """Compare EVERY packed expert tensor (all rows, selected and not) bitwise
    against the base checkpoint on disk, and report delta magnitudes.

    In the delta scheme the packed tensors must be bit-identical to the base for
    ALL experts; training lives in the deltas. Returns a summary dict; raises no
    exception on mismatch (caller inspects ``ok``).
    """
    model_dir = os.path.expanduser(model_dir)
    weight_map = _disk_key_index(model_dir)
    report = {"ok": True, "layers": {}, "n_rows_mismatched": 0,
              "delta": {"n_rows_total": 0, "n_rows_nonzero": 0, "max_abs": 0.0}}
    for layer_idx, expert_ids in sorted(handles.selected.items()):
        experts = handles.experts_modules[layer_idx]
        layer_rep = {}
        for pname in ("gate_up_proj", "down_proj"):
            live = experts._parameters[pname].detach().to("cpu")
            disk_key = None
            for k in weight_map:
                if k.endswith(f"layers.{layer_idx}.mlp.experts.{pname}"):
                    disk_key = k
                    break
            if disk_key is None:
                report["ok"] = False
                layer_rep[pname] = "DISK_KEY_NOT_FOUND"
                continue
            shard = os.path.join(model_dir, weight_map[disk_key])
            with _st_open(shard, framework="pt") as f:
                base = f.get_tensor(disk_key)
            eq = _bitwise_equal_rows(live, base)
            n_bad = int((~eq).sum())
            sel_mask = torch.zeros(eq.shape[0], dtype=torch.bool)
            sel_mask[torch.tensor(expert_ids, dtype=torch.long)] = True
            layer_rep[pname] = {
                "rows_equal": int(eq.sum()), "rows_total": int(eq.numel()),
                "nonselected_mismatch": int((~eq & ~sel_mask).sum()),
                "selected_mismatch": int((~eq & sel_mask).sum()),
            }
            if n_bad:
                report["ok"] = False
                report["n_rows_mismatched"] += n_bad
            del live, base
        for dname in ("delta_gate_up", "delta_down"):
            d = getattr(experts, dname).detach()
            row_nonzero = (d.reshape(d.shape[0], -1) != 0).any(dim=1)
            report["delta"]["n_rows_total"] += int(d.shape[0])
            report["delta"]["n_rows_nonzero"] += int(row_nonzero.sum())
            report["delta"]["max_abs"] = max(report["delta"]["max_abs"],
                                             float(d.abs().max()))
        report["layers"][layer_idx] = layer_rep
    return report
