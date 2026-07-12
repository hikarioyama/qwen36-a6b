#!/usr/bin/env python3
"""router 診断 baseline: pre-topk softmax の rank 質量 + entropy + 使用分布。

k32 借金 (-3.17pt) の機構観測用。pre-topk softmax は設定 k に依存しないため
モデルごとに 1 走で足りる (k8/k32 の差は決定論的再正規化)。
記録 (per MoE layer):
  - rank 1-8 / 9-32 / 33-256 の平均質量 (降順ソート後の pre-topk 確率)
  - pre-topk 分布の平均 entropy (nats)
  - top-k32 での expert 使用回数ヒストグラム (load balance / collapse 検出)
usage: router_rank_baseline.py --model <dir> --profiling <jsonl> --out <npz>
"""
import argparse, json, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # deploy copy lives one level down


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
    ap.add_argument("--model", required=True)
    ap.add_argument("--profiling", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from esft_qwen.common import find_moe_blocks

    config = AutoConfig.from_pretrained(args.model)
    text_cfg = getattr(config, "text_config", config)
    text_cfg.num_experts_per_tok = 32  # usage@k32 を見る (logits には無関係)

    print(f"loading {args.model} (bf16, gpu{args.gpu}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, device_map={"": args.gpu})
    model.eval()

    refs = find_moe_blocks(model)
    L = len(refs)
    E = int(refs[0].experts.gate_up_proj.shape[0])
    print(f"moe layers={L} experts={E}", flush=True)

    mass18 = np.zeros(L); mass932 = np.zeros(L); ent = np.zeros(L)
    usage = np.zeros((L, E), dtype=np.int64); ntok = np.zeros(L, dtype=np.int64)
    pos_of = {id(r.gate): i for i, r in enumerate(refs)}

    def make_hook(gate):
        pos = pos_of[id(gate)]
        def hook(_m, _in, out):
            logits, _scores, indices = out
            with torch.no_grad():
                p = torch.softmax(logits.float(), dim=-1)          # (T, E)
                sp = p.sort(dim=-1, descending=True).values
                mass18[pos] += sp[:, :8].sum().item()
                mass932[pos] += sp[:, 8:32].sum().item()
                ent[pos] += (-(p * p.clamp_min(1e-9).log()).sum(-1)).sum().item()
                idx = indices.reshape(-1).to("cpu").numpy()
                np.add.at(usage[pos], idx, 1)
                ntok[pos] += p.shape[0]
        return hook

    handles = [r.gate.register_forward_hook(make_hook(r.gate)) for r in refs]
    blocks = load_blocks(args.profiling)
    dev = next(model.parameters()).device
    with torch.no_grad():
        for bi, ids in enumerate(blocks):
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
            if (bi + 1) % 16 == 0:
                print(f"  block {bi+1}/{len(blocks)}", flush=True)
    for h in handles:
        h.remove()

    n = ntok.astype(np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             mass_rank1_8=mass18 / n, mass_rank9_32=mass932 / n,
             entropy=ent / n, usage_k32=usage, n_tokens=ntok,
             layer_indices=np.array([r.layer_idx for r in refs]))
    print(f"saved {args.out}")
    print(f"summary: rank1-8 {(mass18/n).mean():.4f}  rank9-32 {(mass932/n).mean():.4f}  "
          f"entropy {(ent/n).mean():.3f} nats  tokens {int(ntok[0])}", flush=True)
    print("ROUTER_BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
