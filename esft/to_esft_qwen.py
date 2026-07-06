#!/usr/bin/env python
"""ESFT freeze/patch entry point for Qwen3.6-35B-A3B.

The implementation lives in ``esft_qwen/esft_patch.py`` (single source of truth,
unit-tested by ``tests/test_smoke.py``). This module re-exports the public API and
adds a small CLI to apply a config to a model and inspect the trainable footprint.

API (import from here or from ``esft_qwen``):
    to_esft_qwen(model, expert_config, ...) -> EsftHandles
        Freeze all params, then unfreeze + gradient-mask only the selected experts'
        packed tensors. Qwen3_5Moe packs experts into 3D Parameters, so per-expert
        freezing is done by masking gradient rows (dim 0 = expert axis), not by the
        upstream ESFT buffer/param trick.
    build_param_groups(model, handles, weight_decay) -> optimiser param groups
        Puts packed expert params in a weight_decay=0 group so non-selected experts
        stay bit-exact frozen through AdamW steps.
    save_expert_patch(model, expert_config, path) / load_expert_patch(model, path)
        Persist / restore only the selected experts' weight slices.

CLI (GPU-gated for the real model; runs on any Qwen3_5Moe checkpoint):
    <venv>/bin/python to_esft_qwen.py --model <path> --config configs/math.json \
        [--save-patch out/math_init.safetensors] [--dtype bfloat16]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esft_qwen.esft_patch import (  # re-export
    to_esft_qwen,
    build_param_groups,
    save_expert_patch,
    load_expert_patch,
    EsftHandles,
)

__all__ = [
    "to_esft_qwen",
    "build_param_groups",
    "save_expert_patch",
    "load_expert_patch",
    "EsftHandles",
]


def _strip_meta(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model path or HF id")
    ap.add_argument("--config", required=True, help="expert config json")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="cpu",
                    help="'cpu' to inspect without GPU; 'auto' for real training placement")
    ap.add_argument("--save-patch", default=None, help="save the (untrained) selected-expert patch here")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    expert_config = _strip_meta(json.load(open(args.config)))

    config = AutoConfig.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} (device_map={args.device_map}) ...")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, config=config, dtype=dtype, device_map=args.device_map)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, dtype=dtype, device_map=args.device_map)

    handles = to_esft_qwen(model, expert_config)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_layers = len(expert_config["experts"])
    n_selected = sum(len(v) for v in expert_config["experts"].values())
    print(f"ESFT applied: {n_layers} MoE layers, {n_selected} experts selected total")
    print(f"trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.3f}%)")
    print(f"grad-masked packed expert tensors: {len(handles.expert_params)}")

    if args.save_patch:
        info = save_expert_patch(model, expert_config, args.save_patch)
        print(f"saved patch: {info}")


if __name__ == "__main__":
    main()
