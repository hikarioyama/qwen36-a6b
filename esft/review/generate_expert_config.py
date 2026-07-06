#!/usr/bin/env python
"""Phase-0 (CPU): turn router statistics into an ESFT expert config, and compare
expert selection across domains.

Two modes:

  * generate:  --stats math.npz --score token --top-p 0.2 --out configs/math.json
        Emits ESFT-format {"experts": {layer: [ids]}, "shared_experts", ...}.
        Presets: ESFT-Token uses (score=token, p=0.2); ESFT-Gate uses (score=gate, p=0.1).

  * compare:   --compare math.npz coding.npz japanese.npz --score token --top-p 0.2
        Builds the config for each and reports the pairwise Jaccard overlap of the
        selected expert sets (per-layer + mean), to see how domain-specialised the
        selections are.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _load(path):
    d = np.load(path)
    return d


def _score_matrix(npz, score):
    key = "token_scores" if score == "token" else "gate_scores"
    return npz[key], npz["layer_indices"].tolist()


def do_generate(args):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from esft_qwen.scoring import build_expert_config

    npz = _load(args.stats)
    mat, layer_indices = _score_matrix(npz, args.score)
    cfg = build_expert_config(
        mat, layer_indices, args.top_p,
        train_shared_experts=args.train_shared_experts,
        train_non_expert_modules=args.train_non_expert_modules,
    )
    # Add provenance metadata (ignored by to_esft_qwen, useful for humans).
    cfg["_meta"] = {
        "stats": os.path.abspath(args.stats),
        "score_function": args.score,
        "top_p": args.top_p,
        "top_k": int(npz["top_k"]),
        "n_tokens": int(npz["n_tokens"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=1)

    counts = [len(v) for v in cfg["experts"].values()]
    print(f"wrote {args.out}")
    print(f"  score={args.score} top_p={args.top_p} layers={len(counts)} "
          f"experts/layer: min={min(counts)} mean={np.mean(counts):.1f} max={max(counts)} "
          f"total_selected={sum(counts)}")


def do_compare(args):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from esft_qwen.scoring import build_expert_config, domain_overlap_matrix

    configs = {}
    for path in args.compare:
        name = os.path.splitext(os.path.basename(path))[0]
        npz = _load(path)
        mat, layer_indices = _score_matrix(npz, args.score)
        configs[name] = build_expert_config(mat, layer_indices, args.top_p)

    ov = domain_overlap_matrix(configs)
    domains = ov["domains"]
    mean = ov["mean"]

    print(f"\n=== domain expert-selection overlap (Jaccard) | score={args.score} p={args.top_p} ===")
    print("mean over layers:")
    hdr = "            " + "  ".join(f"{d[:8]:>8}" for d in domains)
    print(hdr)
    for i, d in enumerate(domains):
        row = "  ".join(f"{mean[i, j]:8.3f}" for j in range(len(domains)))
        print(f"{d[:10]:>10}  {row}")

    # Per-domain selected-count summary.
    print("\nselected experts/layer (mean):")
    for name, cfg in configs.items():
        counts = [len(v) for v in cfg["experts"].values()]
        print(f"  {name:>10}: mean={np.mean(counts):.1f} total={sum(counts)}")

    if args.out:
        out = {
            "score_function": args.score, "top_p": args.top_p,
            "domains": domains,
            "mean_jaccard": mean.tolist(),
            "layers": ov["layers"],
            "per_layer_jaccard": {str(l): ov["per_layer"][l].tolist() for l in ov["layers"]},
            "configs": {k: v for k, v in configs.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote overlap report -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", help="single stats .npz (generate mode)")
    ap.add_argument("--compare", nargs="+", help="multiple stats .npz (compare mode)")
    ap.add_argument("--score", choices=["token", "gate"], default="token",
                    help="ESFT-Token (token, p=0.2) or ESFT-Gate (gate, p=0.1)")
    ap.add_argument("--top-p", type=float, default=0.2)
    ap.add_argument("--out", help="output path (config json or overlap report)")
    ap.add_argument("--train-shared-experts", action="store_true")
    ap.add_argument("--train-non-expert-modules", action="store_true")
    args = ap.parse_args()

    if args.compare:
        do_compare(args)
    elif args.stats:
        if not args.out:
            ap.error("--out required in generate mode")
        do_generate(args)
    else:
        ap.error("provide --stats (generate) or --compare (compare)")


if __name__ == "__main__":
    main()
