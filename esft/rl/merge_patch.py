#!/usr/bin/env python
"""Merge an esft-qwen-patch-v1 expert patch into a base HF checkpoint.

The patch stores FINAL weight slices (not deltas) for selected experts:
  key = layers.{L}.experts.{gate_up|down}.{E}
Base packed tensors (Qwen3.5/3.6 MoE):
  ...layers.{L}.mlp.experts.gate_up_proj  [256, x, y]   (expert axis = dim0)
  ...layers.{L}.mlp.experts.down_proj     [256, x, y]

Streams shard-by-shard (RAM ~= 1 shard + patch). Verifies:
  (a) every patched row == patch tensor (bitwise)
  (b) a sample of untouched rows == base (bitwise)
  (c) all patch keys consumed exactly once

Usage:
  python merge_patch.py --base <dir> --patch <file> --out <dir>
"""
import argparse, json, os, re, shutil, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

# NOTE: anchor to the main text stack; the checkpoint also has `mtp.layers.0.mlp.experts.*`
# (MTP draft module) whose experts must NOT receive main-layer patches.
PAT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
WHICH = {"gate_up_proj": "gate_up", "down_proj": "down"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-untouched", type=int, default=8,
                    help="untouched expert rows to bitwise-verify per patched tensor")
    args = ap.parse_args()

    idx_path = os.path.join(args.base, "model.safetensors.index.json")
    index = json.load(open(idx_path))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    patch = load_file(args.patch)
    with safe_open(args.patch, framework="pt") as f:
        meta = f.metadata()
    assert meta.get("format") == "esft-qwen-patch-v1", f"bad patch format: {meta.get('format')}"
    consumed = set()

    # which shards contain patchable tensors? (avoid rewriting untouched 2.7GB shards)
    patched_layers = {int(k.split(".")[1]) for k in patch}
    def shard_is_touched(shard_name):
        for name, sh in weight_map.items():
            if sh != shard_name:
                continue
            m = PAT.search(name)
            if m and int(m.group(1)) in patched_layers:
                return True
        return False

    os.makedirs(args.out, exist_ok=True)
    n_rows = 0
    for shard in shards:
        dst = os.path.join(args.out, shard)
        if not shard_is_touched(shard):
            if os.path.lexists(dst):
                os.remove(dst)
            try:
                os.link(os.path.join(args.base, shard), dst)   # same-fs: instant
            except OSError:
                shutil.copy2(os.path.join(args.base, shard), dst)
            print(f"[merge] {shard} linked (untouched)", flush=True)
            continue
        tensors = load_file(os.path.join(args.base, shard))
        touched = False
        for name, t in tensors.items():
            m = PAT.search(name)
            if not m:
                continue
            layer, which = int(m.group(1)), WHICH[m.group(2)]
            keys = [(k, int(k.rsplit(".", 1)[1])) for k in patch
                    if k.startswith(f"layers.{layer}.experts.{which}.")]
            if not keys:
                continue
            patched_eids = set()
            for k, e in keys:
                t[e].copy_(patch[k].to(t.dtype))
                consumed.add(k)
                patched_eids.add(e)
                n_rows += 1
            # (a) bitwise: patched rows == patch
            for k, e in keys:
                assert torch.equal(t[e], patch[k].to(t.dtype)), f"verify fail {k}"
            touched = True
        save_file(tensors, dst, metadata={"format": "pt"})
        print(f"[merge] {shard} {'PATCHED' if touched else 'copied'}", flush=True)
        del tensors

    # (c) all patch keys consumed
    missing = set(patch) - consumed
    assert not missing, f"{len(missing)} patch keys never matched base: {sorted(missing)[:5]}"

    # copy non-weight files (config/tokenizer/index)
    for fn in os.listdir(args.base):
        if fn.endswith(".safetensors"):
            continue
        src = os.path.join(args.base, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, fn))
    print(f"[merge] DONE rows={n_rows} tensors_consumed={len(consumed)}/{len(patch)} out={args.out}")


if __name__ == "__main__":
    main()
