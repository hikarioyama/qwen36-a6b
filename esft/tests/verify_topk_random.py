#!/usr/bin/env python
"""ACCEPTANCE GATE (CPU, tiny stand-in): prove --router-topk-random actually varies
gate.top_k per training forward and snaps back to k=32 for eval/save.

No full model / no GPU: builds a tiny nn.Module whose MoE blocks are named
``layers.{i}.mlp`` and whose class name ends in ``SparseMoeBlock`` (what
find_moe_blocks keys on), each carrying a gate with a ``top_k`` attr and a forward
that RECORDS the top_k it observed. We then drive the exact production hook
(train_esft.install_topk_random_hook) and assert:

  (a) across many training forwards, observed k covers the sampled set and only ever
      takes values IN the set;
  (b) within any single forward, ALL gates saw the SAME k;
  (c) in eval mode (model.eval()), every forward observes k == max(set) (=32);
  (d) the RNG is deterministic for a fixed seed (two installs -> identical k sequence),
      which is what keeps DDP ranks in sync.

Run: ~/vllm-env/bin/python tests/verify_topk_random.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from esft_qwen.common import find_moe_blocks
from train_esft import install_topk_random_hook


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 8  # config default; hook must override it


class TinySparseMoeBlock(nn.Module):
    """Class name ends in 'SparseMoeBlock' -> find_moe_blocks matches it."""

    def __init__(self, seen):
        super().__init__()
        self.gate = _Gate()
        self.experts = nn.Module()  # find_moe_blocks requires an `experts` attr
        self._seen = seen  # shared list: (block_idx, top_k) observed this forward

    def forward(self, x):
        self._seen.append(self.gate.top_k)  # read at call time, like the real router
        return x


class _Layer(nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.mlp = TinySparseMoeBlock(seen)


class TinyMoEModel(nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        self.seen = []  # top_k values observed across all gates in the current forward
        self.layers = nn.ModuleList(_Layer(self.seen) for _ in range(n_layers))

    def forward(self, x):
        self.seen.clear()
        for lyr in self.layers:
            x = lyr.mlp(x)
        return x


def main():
    torch.manual_seed(0)
    topk_set = [8, 16, 24, 32]
    eval_k = max(topk_set)
    n_layers = 4

    model = TinyMoEModel(n_layers)
    gates = [ref.gate for ref in find_moe_blocks(model)]
    assert len(gates) == n_layers, f"expected {n_layers} gates, got {len(gates)}"

    handle, ek = install_topk_random_hook(model, gates, topk_set, seed=123)
    assert ek == eval_k, f"eval_k should be {eval_k}, got {ek}"
    # before any forward, hook set every gate to eval_k (infer_moe_dims safety)
    assert all(g.top_k == eval_k for g in gates), "pre-forward gates must default to k=32"

    x = torch.zeros(1)

    # (a)+(b) training forwards
    model.train()
    per_forward_k = []
    for _ in range(400):
        model(x)
        seen = model.seen
        assert len(seen) == n_layers, "every gate must be visited once per forward"
        assert len(set(seen)) == 1, f"(b) all gates must share one k per forward, got {set(seen)}"
        per_forward_k.append(seen[0])
    observed = set(per_forward_k)
    assert observed <= set(topk_set), f"(a) observed k outside set: {observed - set(topk_set)}"
    assert observed == set(topk_set), f"(a) some set values never sampled in 400 forwards: {set(topk_set) - observed}"
    print(f"[a,b] train: observed k = {sorted(observed)}; "
          f"counts = { {k: per_forward_k.count(k) for k in sorted(observed)} }")

    # (c) eval mode -> always k=32, RNG must NOT advance
    model.eval()
    for _ in range(20):
        model(x)
        assert set(model.seen) == {eval_k}, f"(c) eval must be k={eval_k}, got {set(model.seen)}"
    assert all(g.top_k == eval_k for g in gates), "(c) gates left at k=32 after eval"
    print(f"[c] eval: all forwards observed k={eval_k}, gates reset to k={eval_k}")

    # (d) determinism: fresh install, same seed -> identical training k sequence
    handle.remove()
    model2 = TinyMoEModel(n_layers)
    gates2 = [ref.gate for ref in find_moe_blocks(model2)]
    install_topk_random_hook(model2, gates2, topk_set, seed=123)
    model2.train()
    seq2 = []
    for _ in range(400):
        model2(x)
        seq2.append(model2.seen[0])
    assert seq2 == per_forward_k, "(d) same seed must reproduce identical k sequence (DDP sync)"
    print(f"[d] determinism: identical {len(seq2)}-forward k sequence for seed=123")

    print("PASS: router-topk-random varies k per forward, shares k across layers, "
          "snaps to k=32 for eval, and is DDP-deterministic")


if __name__ == "__main__":
    main()
