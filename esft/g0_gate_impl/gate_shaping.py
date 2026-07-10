#!/usr/bin/env python
"""G0: inference-time gate (router) shaping for Qwen3_5Moe -- training-free.

Question G0 answers
-------------------
Running Qwen3.6-35B-A3B at top-k 32 (instead of the shipped top-8) drags MMLU
down (0.8433 -> 0.8067) because ranks 9..32 carry ~54% of the gate mass through
experts that were never calibrated for that mass. G0 asks: can we claw that
knowledge back *at inference time* by reshaping the router weights, with the model
weights untouched?

Mechanism (why a single code path covers all three variants)
------------------------------------------------------------
The real router (``Qwen3_5MoeTopKRouter.forward``) is::

    router_logits = F.linear(hidden, weight)              # (N, E=256)
    probs         = softmax(router_logits, float32)       # over ALL 256
    val, idx      = topk(probs, top_k)                    # descending
    val           = val / val.sum(-1, keepdim=True)       # renorm (always on)
    val           = val.to(bf16)                          # -> experts()

``SparseMoeBlock`` then multiplies each expert's output by its ``val`` weight, so
an expert whose ``val`` is zeroed contributes nothing -- an effective drop with NO
change to the ``(N, top_k)`` tensor shape. Every G0 variant is therefore just a
*reweighting of ``val``* over the SAME top-k indices:

- **temp**    : ``probs = softmax(logits / tau)`` before topk. Temperature is a
                monotonic map, so the top-k *indices are identical*; only the
                relative weights sharpen (tau<1 -> mass piles onto rank 1). This is
                the "concentrate onto the calibrated head" lever.
- **masscut** : keep the shortest rank prefix whose cumulative (renormalised) mass
                first reaches ``p``; zero the rest; renorm. Effective-k becomes
                *dynamic per token* -- the mechanism metric we log.
- **rankdamp**: multiply ranks 9..32 (0-indexed positions >= 8) by ``alpha``,
                keep ranks 1..8 intact, renorm. A blunt "trust the head, discount
                the uncalibrated tail" lever.

All arithmetic runs in fp32 (matching the model's softmax dtype) and is cast to
the logits dtype only at the very end, exactly like the stock forward. With
``tau=1`` and no cut/damp the recompute is bit-identical to the stock router, so
the OFF path (no hook registered) is provably a no-op.

Wiring
------
``maybe_install_from_env(refs)`` reads three env vars and, only if ``G0_SHAPE`` is
set, registers a forward hook on every gate module:

    G0_SHAPE = temp | masscut | rankdamp     (unset -> no hook, stock behaviour)
    G0_PARAM = tau (temp) | p (masscut) | alpha (rankdamp)
    G0_DEBUG = 1  -> print gate-mass diagnostics for the first G0_DEBUG_CALLS
                     forward calls per GPU (default 8), so a silent no-op is caught.

The hook returns a *reshaped* ``(router_logits, val, idx)`` tuple; ``router_logits``
is passed through untouched (only ``val`` is reweighted), so any downstream
consumer that reads raw logits (aux-loss logging etc.) still sees the true logits.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# Ranks 1..8 (0-indexed 0..7) are the "calibrated head" the shipped top-8 model
# was trained on; rankdamp discounts everything from position HEAD_K onward.
HEAD_K = 8

VALID_VARIANTS = ("temp", "masscut", "rankdamp")


# --------------------------------------------------------------------------- #
# diagnostics: accumulate per-GPU so "the flag did nothing" is impossible to miss
# --------------------------------------------------------------------------- #

class _DebugAccumulator:
    """Collects gate-mass stats across the first ``max_calls`` forward calls."""

    def __init__(self, variant, param, gpu_id, max_calls):
        self.variant = variant
        self.param = param
        self.gpu_id = gpu_id
        self.max_calls = max_calls
        self.calls = 0
        # running means over tokens seen
        self.tok = 0
        self.eff_k_sum = 0.0            # effective active experts (val>0)
        self.head_mass_base_sum = 0.0   # rank1..8 mass BEFORE shaping
        self.head_mass_shaped_sum = 0.0 # rank1..8 mass AFTER shaping
        self.rank1_base_sum = 0.0
        self.rank1_shaped_sum = 0.0

    def update(self, base_val, shaped_val):
        # base_val / shaped_val: (N, top_k) fp32, each row sums to 1.
        n = base_val.shape[0]
        self.tok += n
        eff_k = (shaped_val > 0).sum(dim=-1).float().sum().item()
        self.eff_k_sum += eff_k
        hk = min(HEAD_K, base_val.shape[-1])
        self.head_mass_base_sum += base_val[:, :hk].sum().item()
        self.head_mass_shaped_sum += shaped_val[:, :hk].sum().item()
        self.rank1_base_sum += base_val[:, 0].sum().item()
        self.rank1_shaped_sum += shaped_val[:, 0].sum().item()
        self.calls += 1
        if self.calls == self.max_calls:
            self.report(final=True)

    def report(self, final=False):
        t = max(self.tok, 1)
        tag = "FINAL" if final else "..."
        print(
            f"[gpu{self.gpu_id} G0-debug {tag}] variant={self.variant} "
            f"param={self.param} calls={self.calls} tokens={self.tok} | "
            f"eff_active_experts={self.eff_k_sum / t:.3f} "
            f"(top_k slots={int(round(self.eff_k_sum / t)) if self.tok else '?'}) | "
            f"head(1-{HEAD_K})_mass base={self.head_mass_base_sum / t:.4f} "
            f"-> shaped={self.head_mass_shaped_sum / t:.4f} | "
            f"rank1_wt base={self.rank1_base_sum / t:.4f} "
            f"-> shaped={self.rank1_shaped_sum / t:.4f}",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# core shaping
# --------------------------------------------------------------------------- #

def shape_router_scores(router_logits, top_k, variant, param):
    """Return ``(base_val, shaped_val, idx)`` all in fp32.

    ``base_val`` is the stock renormalised top-k weight (for diagnostics / the
    OFF-equivalence check); ``shaped_val`` is after the variant's reweighting.
    ``idx`` is the stock top-k index tensor. The recompute mirrors the stock
    forward's fp32 softmax exactly, so ``variant``-with-identity params reproduces
    the stock weights bit-for-bit.
    """
    logits_f = router_logits.float()

    # temperature is applied BEFORE softmax; monotonic, so idx is unchanged vs tau=1
    tau = param if variant == "temp" else 1.0
    probs = F.softmax(logits_f / tau, dim=-1)
    val, idx = torch.topk(probs, top_k, dim=-1)          # descending
    base_val = val / val.sum(dim=-1, keepdim=True)       # stock renorm, fp32

    if variant == "temp":
        shaped = base_val                                # sharpening already in probs
    elif variant == "masscut":
        # keep the shortest prefix whose cumulative mass first reaches p.
        csum = base_val.cumsum(dim=-1)
        # rank i survives iff the mass BEFORE it is still < p (so the rank that
        # first crosses p is included; rank 0 always survives since csum-val=0).
        keep = (csum - base_val) < float(param)
        shaped = base_val * keep
        shaped = shaped / shaped.sum(dim=-1, keepdim=True)
    elif variant == "rankdamp":
        damp = torch.ones_like(base_val)
        damp[:, HEAD_K:] = float(param)
        shaped = base_val * damp
        shaped = shaped / shaped.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown G0 variant {variant!r}; choose {VALID_VARIANTS}")

    return base_val, shaped, idx


def _make_hook(variant, param, dbg):
    def hook(module, inputs, output):
        router_logits, _router_scores, _router_idx = output
        top_k = int(module.top_k)
        base_val, shaped_val, idx = shape_router_scores(
            router_logits, top_k, variant, param)
        if dbg is not None and dbg.calls < dbg.max_calls:
            dbg.update(base_val.detach(), shaped_val.detach())
        return router_logits, shaped_val.to(router_logits.dtype), idx

    return hook


# --------------------------------------------------------------------------- #
# installation
# --------------------------------------------------------------------------- #

def _read_env():
    variant = os.environ.get("G0_SHAPE", "").strip().lower()
    if not variant:
        return None
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"G0_SHAPE={variant!r} invalid; choose one of {VALID_VARIANTS} or unset")
    if "G0_PARAM" not in os.environ:
        raise ValueError(f"G0_SHAPE={variant} set but G0_PARAM missing")
    param = float(os.environ["G0_PARAM"])
    debug = os.environ.get("G0_DEBUG", "").strip() in ("1", "true", "yes")
    max_calls = int(os.environ.get("G0_DEBUG_CALLS", "8"))
    return variant, param, debug, max_calls


def maybe_install_from_env(refs, gpu_id=0):
    """Install the gate-shaping hook on every gate in ``refs`` iff G0_SHAPE is set.

    ``refs`` is a list of ``MoeBlockRef`` (from ``esft_qwen.common.find_moe_blocks``)
    or any objects exposing a ``.gate`` module with a ``top_k`` attribute. Returns
    the list of hook handles (empty when shaping is OFF). Idempotent per process is
    the caller's concern; call once right after the model is loaded.
    """
    parsed = _read_env()
    if parsed is None:
        print(f"[gpu{gpu_id}] G0 gate-shaping OFF (G0_SHAPE unset) -- stock router",
              flush=True)
        return []
    variant, param, debug, max_calls = parsed

    dbg = _DebugAccumulator(variant, param, gpu_id, max_calls) if debug else None
    handles = []
    for r in refs:
        gate = r.gate if hasattr(r, "gate") else r
        handles.append(gate.register_forward_hook(_make_hook(variant, param, dbg)))
    print(
        f"[gpu{gpu_id}] G0 gate-shaping ON: variant={variant} param={param} "
        f"hooks={len(handles)} debug={bool(debug)}",
        flush=True,
    )
    return handles
