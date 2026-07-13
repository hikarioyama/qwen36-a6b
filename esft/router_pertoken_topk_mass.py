#!/usr/bin/env python3
"""動的 K の前提検証: per-token の top-k 質量分布ダンプ。

router_rank_baseline.py は層ごとの平均しか残さないため、「トークンごとに
必要なエキスパート人数が違うか」(= 動的 K が静的 α ダイヤルに勝つ条件) を
判定できない。本スクリプトは同一 profiling blocks で per-token per-layer の
  - m8:    pre-topk softmax の top-8 質量 (float16)
  - m32:   同 top-32 質量 (float16)
  - keff95: top-32 質量の 95% に到達する最小 k (uint8, 1..32)
を丸ごと保存する。router は tail05 で凍結中なので base 1 走で現行 run にも有効。
usage: router_pertoken_topk_mass.py --model <dir> --profiling <jsonl> --out <npz> [--gpu 0]
"""
import argparse, json, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))


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

    blocks = load_blocks(args.profiling)
    total_tok = sum(len(b) for b in blocks)

    config = AutoConfig.from_pretrained(args.model)
    print(f"loading {args.model} (bf16, gpu{args.gpu}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, device_map={"": args.gpu})
    model.eval()

    refs = find_moe_blocks(model)
    L = len(refs)
    print(f"moe layers={L} total_tok={total_tok}", flush=True)

    m8 = np.zeros((L, total_tok), dtype=np.float16)
    m32 = np.zeros((L, total_tok), dtype=np.float16)
    keff = np.zeros((L, total_tok), dtype=np.uint8)
    cursor = np.zeros(L, dtype=np.int64)
    pos_of = {id(r.gate): i for i, r in enumerate(refs)}

    def make_hook(gate):
        pos = pos_of[id(gate)]
        def hook(_m, _in, out):
            logits, _scores, _indices = out
            with torch.no_grad():
                p = torch.softmax(logits.float(), dim=-1)              # (T, E)
                sp = p.sort(dim=-1, descending=True).values[:, :32]    # (T, 32)
                cs = sp.cumsum(-1)
                t8 = cs[:, 7]
                t32 = cs[:, 31]
                thr = 0.95 * t32
                k95 = (cs < thr.unsqueeze(-1)).sum(-1) + 1             # 1..32
                n = p.shape[0]
                c = cursor[pos]
                m8[pos, c:c + n] = t8.to("cpu").numpy().astype(np.float16)
                m32[pos, c:c + n] = t32.to("cpu").numpy().astype(np.float16)
                keff[pos, c:c + n] = k95.to("cpu").numpy().astype(np.uint8)
                cursor[pos] = c + n
        return hook

    handles = [r.gate.register_forward_hook(make_hook(r.gate)) for r in refs]
    dev = next(model.parameters()).device
    import torch as _t
    with _t.no_grad():
        for bi, ids in enumerate(blocks):
            x = _t.tensor([ids], dtype=_t.long, device=dev)
            model(x)
            if (bi + 1) % 16 == 0:
                print(f"block {bi + 1}/{len(blocks)}", flush=True)
    for h in handles:
        h.remove()

    assert int(cursor[0]) == total_tok, f"cursor mismatch {cursor[0]} != {total_tok}"
    np.savez_compressed(args.out, m8=m8, m32=m32, keff95=keff,
                        n_tokens=np.array([total_tok]))
    print(f"PERTOKEN_DUMP_DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
