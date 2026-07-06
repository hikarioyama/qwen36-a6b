#!/usr/bin/env python
"""Phase-0 (GPU-gated): collect ESFT router statistics for Qwen3.6-35B-A3B.

Loads the model, registers forward hooks on every routed-MoE router
(``Qwen3_5MoeTopKRouter``), runs the packed profiling blocks through it, and
accumulates per-(layer, expert) ESFT gate/token scores. Because we hook the *real*
router module, the routing (softmax-over-all -> top-k -> renormalise) is reproduced
exactly, including the always-on top-k renormalisation.

``--top-k N`` overrides ``num_experts_per_tok`` (via config) so you can profile the
routing distribution at a different K than the model default (8).

Output: an ``.npz`` with token_scores/gate_scores (num_moe_layers, num_experts),
the raw counts, the layer-index map, and metadata.

Run (GPU phase, when VRAM is free):
    <venv>/bin/python collect_router_stats.py \
        --profiling data/profiling/math.jsonl \
        --out stats/math_top8.npz --top-k 8
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load_blocks(path):
    blocks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                blocks.append(json.loads(line)["input_ids"])
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiling", required=True, help="profiling jsonl (input_ids blocks)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B",
                    help="model path or HF id (use a local snapshot to avoid download)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="override num_experts_per_tok for profiling (default: model config)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-blocks", type=int, default=0, help="0 = all blocks")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    # Import the shared library (this file lives at the project root).
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from esft_qwen.common import find_moe_blocks
    from esft_qwen.scoring import ScoreAccumulator

    dtype = getattr(torch, args.dtype)

    config = AutoConfig.from_pretrained(args.model)
    # Locate the text sub-config (multimodal wrapper) or the config itself.
    text_cfg = getattr(config, "text_config", config)
    if args.top_k is not None:
        text_cfg.num_experts_per_tok = int(args.top_k)
        print(f"override num_experts_per_tok -> {args.top_k}")

    print(f"loading model {args.model} (dtype={args.dtype}, device_map={args.device_map}) ...")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, config=config, dtype=dtype, device_map=args.device_map)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, dtype=dtype, device_map=args.device_map)
    model.eval()

    refs = find_moe_blocks(model)
    num_experts = int(refs[0].experts.gate_up_proj.shape[0])
    top_k = int(getattr(refs[0].gate, "top_k"))
    layer_indices = [r.layer_idx for r in refs]
    print(f"moe layers={len(refs)} experts={num_experts} top_k={top_k} "
          f"layer_idx=[{layer_indices[0]}..{layer_indices[-1]}]")

    acc = ScoreAccumulator(num_layers=len(refs), num_experts=num_experts, top_k=top_k)
    gate_to_pos = {id(r.gate): pos for pos, r in enumerate(refs)}

    def make_hook(gate_module):
        pos = gate_to_pos[id(gate_module)]

        def hook(_m, _in, out):
            # Qwen3_5MoeTopKRouter returns (router_logits, router_scores, router_indices).
            _, scores, indices = out
            acc.update(pos, indices.detach().to("cpu").numpy(),
                       scores.detach().float().to("cpu").numpy())
        return hook

    handles = [r.gate.register_forward_hook(make_hook(r.gate)) for r in refs]

    blocks = load_blocks(args.profiling)
    if args.max_blocks:
        blocks = blocks[: args.max_blocks]
    print(f"profiling {len(blocks)} blocks x {len(blocks[0]) if blocks else 0} tokens")

    first_param_device = next(model.parameters()).device
    with torch.no_grad():
        for bi, ids in enumerate(blocks):
            input_ids = torch.tensor([ids], dtype=torch.long, device=first_param_device)
            attn = torch.ones_like(input_ids)
            model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            print(f"  block {bi + 1}/{len(blocks)} done (tokens so far ~{acc.n_tokens})", flush=True)

    for h in handles:
        h.remove()

    fin = acc.finalize()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        token_scores=fin["token_scores"],
        gate_scores=fin["gate_scores"],
        token_raw=fin["token_raw"],
        gate_raw=fin["gate_raw"],
        layer_indices=np.array(layer_indices, dtype=np.int64),
        n_tokens=np.array(fin["n_tokens"]),
        top_k=np.array(top_k),
        num_experts=np.array(num_experts),
    )
    print(f"saved {args.out}  (n_tokens={fin['n_tokens']}, top_k={top_k})")
    # Sanity: rows should sum to ~1.
    print(f"token_scores row-sum mean={fin['token_scores'].sum(1).mean():.4f} "
          f"gate_scores row-sum mean={fin['gate_scores'].sum(1).mean():.4f}")


if __name__ == "__main__":
    main()
