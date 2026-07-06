"""Model-structure helpers and an exact reproduction of Qwen3_5Moe routing.

Qwen3_5Moe (the architecture behind Qwen3.6-35B-A3B) stores its routed experts as
*packed 3D Parameters* on ``Qwen3_5MoeExperts`` (``gate_up_proj`` of shape
``(num_experts, 2*moe_intermediate, hidden)`` and ``down_proj`` of shape
``(num_experts, hidden, moe_intermediate)``) rather than as a ``ModuleList`` of
per-expert MLPs like DeepSeek-V2-Lite. That single fact drives every adaptation in
this port (see ``esft_patch.py``).

Module path of a routed-MoE block inside ``Qwen3_5MoeForConditionalGeneration``:

    model.model.language_model.layers.{i}.mlp        -> Qwen3_5MoeSparseMoeBlock
        .gate            -> Qwen3_5MoeTopKRouter   (Parameter `weight` (E, H), attr `top_k`)
        .experts         -> Qwen3_5MoeExperts      (packed `gate_up_proj`, `down_proj`)
        .shared_expert   -> Qwen3_5MoeMLP
        .shared_expert_gate -> Linear(H, 1)

For the text-only ``Qwen3_5MoeForCausalLM`` the path is
``model.model.layers.{i}.mlp`` (the vision wrapper is absent). Both are handled by
matching on the module class name rather than a hard-coded attribute path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Qwen3.6-35B-A3B routes top-8 of 256 experts by default (config.num_experts_per_tok).
TOP_K_DEFAULT = 8

# The routed-MoE block class name in transformers' Qwen3_5Moe modeling file.
_MOE_BLOCK_SUFFIX = "SparseMoeBlock"
_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.mlp$")


@dataclass
class MoeBlockRef:
    """A located routed-MoE block plus its layer index and sub-modules."""

    layer_idx: int
    name: str          # fully-qualified module name, e.g. "model.language_model.layers.3.mlp"
    block: torch.nn.Module
    gate: torch.nn.Module
    experts: torch.nn.Module
    shared_expert: torch.nn.Module | None


def find_moe_blocks(model: torch.nn.Module) -> list[MoeBlockRef]:
    """Locate every routed-MoE block by class name, robust to the wrapper depth.

    Returns them sorted by layer index. Works for the multimodal
    ``Qwen3_5MoeForConditionalGeneration`` and the text-only
    ``Qwen3_5MoeForCausalLM`` alike, and for the tiny stand-in used in tests.
    """
    refs: list[MoeBlockRef] = []
    for name, module in model.named_modules():
        if not type(module).__name__.endswith(_MOE_BLOCK_SUFFIX):
            continue
        m = _LAYER_IDX_RE.search(name)
        if m is None:
            # Fall back to the last integer component of the module path.
            ints = re.findall(r"\.(\d+)\.", name + ".")
            if not ints:
                raise ValueError(f"cannot infer layer index from module name {name!r}")
            layer_idx = int(ints[-1])
        else:
            layer_idx = int(m.group(1))
        refs.append(
            MoeBlockRef(
                layer_idx=layer_idx,
                name=name,
                block=module,
                gate=getattr(module, "gate"),
                experts=getattr(module, "experts"),
                shared_expert=getattr(module, "shared_expert", None),
            )
        )
    refs.sort(key=lambda r: r.layer_idx)
    if not refs:
        raise ValueError(
            "no routed-MoE blocks found; expected modules whose class name ends "
            f"with {_MOE_BLOCK_SUFFIX!r}"
        )
    return refs


def infer_moe_dims(model: torch.nn.Module) -> tuple[int, int, int]:
    """Return (num_moe_layers, num_experts, top_k) inferred from the model."""
    refs = find_moe_blocks(model)
    experts0 = refs[0].experts
    num_experts = int(experts0.gate_up_proj.shape[0])
    top_k = int(getattr(refs[0].gate, "top_k"))
    return len(refs), num_experts, top_k


def compute_router_selection(
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact reproduction of ``Qwen3_5MoeTopKRouter.forward`` selection.

    The real router (transformers modeling_qwen3_5_moe.py) does:

        router_logits = F.linear(hidden, weight)          # (N, E)
        router_probs  = softmax(router_logits, float32)   # over ALL experts, first
        top_val, top_idx = topk(router_probs, top_k)      # then top-k
        top_val = top_val / top_val.sum(-1, keepdim=True) # renormalise (ALWAYS on)

    ``norm_topk_prob`` is *not* a config flag in Qwen3_5Moe: the renormalisation
    line is unconditional. The argument exists here only so tests can exercise the
    un-normalised branch; leave it ``True`` to match the model.

    Returns ``(top_idx (N, top_k) int64, top_val (N, top_k) same dtype as logits)``.
    """
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    top_val, top_idx = torch.topk(router_probs, top_k, dim=-1)
    if norm_topk_prob:
        top_val = top_val / top_val.sum(dim=-1, keepdim=True)
    top_val = top_val.to(router_logits.dtype)
    return top_idx, top_val
