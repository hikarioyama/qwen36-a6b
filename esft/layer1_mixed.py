#!/usr/bin/env python
"""Layer-1 eval for mixed_v1 checkpoints: per-domain held-out NLL on the trainer's
exact val split (byte-identical across nodes, verified 2026-07-07: first10 val idx
[187, 1493, 2130, ...] match aux-host).

One invocation = one model arm on one GPU:
  # base@k32 reference (delta-free):
  python layer1_mixed.py --gpu 0 --out reports/l1mix_base.json
  # checkpoint arm:
  python layer1_mixed.py --gpu 1 --delta /mnt/data/mixed_eval/ckpt300/delta_state.safetensors \
      --out reports/l1mix_ckpt300.json

Compare two arm jsons (CPU, paired bootstrap per domain):
  python layer1_mixed.py --compare reports/l1mix_base.json reports/l1mix_ckpt300.json
"""
import argparse, json, math, os, sys

SEED = 5934875
CACHE = "/mnt/data/mixed_eval/mixed_v1.jsonl.seq7168.seed5934875.ccr0.0.max0.pt"
MANIFEST = CACHE + ".manifest.json"
SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/"
    "995ad96eacd98c81ed38be0c5b274b04031597b0")
CONFIG = "/mnt/data/mixed_eval/mixed_v1_token_k32_p0.18.json"
TOPK = 32

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def val_blocks():
    import torch
    obj = torch.load(CACHE, map_location="cpu")
    input_ids, labels = obj if isinstance(obj, (tuple, list)) else (obj["input_ids"], obj["labels"])
    N = input_ids.shape[0]
    torch.manual_seed(SEED)
    n_val = min(max(1, int(N * 0.02)), 64)
    _, va = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(input_ids, labels), [N - n_val, n_val])
    idx = sorted(va.indices)
    doms = json.load(open(MANIFEST))["block_domains"]
    return [(i, doms[i], input_ids[i], labels[i]) for i in idx]


def run_arm(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM
    from esft_qwen.common import find_moe_blocks
    dev = f"cuda:{args.gpu}"
    blocks = val_blocks()
    print(f"[l1mix] val blocks={len(blocks)} domains="
          f"{ {d: sum(1 for _, dd, *_ in blocks if dd == d) for d in set(b[1] for b in blocks)} }",
          flush=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            SNAP, dtype=torch.bfloat16, device_map={"": args.gpu})
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            SNAP, dtype=torch.bfloat16, device_map={"": args.gpu})
    model.eval(); model.config.use_cache = False
    for r in find_moe_blocks(model):
        r.gate.top_k = TOPK
    if args.delta:
        from esft_qwen.delta_patch import to_esft_delta, load_delta_state
        cfg = json.load(open(args.config))
        to_esft_delta(model, cfg)
        info = load_delta_state(model, args.delta)
        print(f"[l1mix] delta loaded: {info}", flush=True)

    out = []
    with torch.no_grad():
        for k, (i, dom, x, y) in enumerate(blocks):
            xb = x.unsqueeze(0).to(dev); yb = y.unsqueeze(0).to(dev)
            logits = model(input_ids=xb).logits
            sl = logits[:, :-1, :].float(); sy = yb[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.size(-1)), sy.reshape(-1),
                ignore_index=-100, reduction="sum")
            nt = int((sy != -100).sum().item())
            out.append({"block": i, "domain": dom, "sum_nll": ce.item(), "n_tok": nt})
            del logits, sl
            if (k + 1) % 16 == 0:
                print(f"[l1mix] {k+1}/{len(blocks)}", flush=True)
    agg = {}
    for d in sorted(set(r["domain"] for r in out)):
        rs = [r for r in out if r["domain"] == d]
        s = sum(r["sum_nll"] for r in rs); t = sum(r["n_tok"] for r in rs)
        agg[d] = {"blocks": len(rs), "tok": t, "nll": s / t, "ppl": math.exp(s / t)}
    s = sum(r["sum_nll"] for r in out); t = sum(r["n_tok"] for r in out)
    agg["_overall"] = {"blocks": len(out), "tok": t, "nll": s / t, "ppl": math.exp(s / t)}
    res = {"delta": args.delta or None, "topk": TOPK, "blocks": out, "agg": agg}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    for d, a in agg.items():
        print(f"[l1mix] {d}: blocks={a['blocks']} nll={a['nll']:.4f} ppl={a['ppl']:.3f}")
    print(f"[l1mix] wrote {args.out}")


def compare(pa, pb, iters=10000):
    import random
    A = json.load(open(pa)); B = json.load(open(pb))
    ba = {r["block"]: r for r in A["blocks"]}; bb = {r["block"]: r for r in B["blocks"]}
    ids = sorted(set(ba) & set(bb))
    doms = sorted(set(ba[i]["domain"] for i in ids)) + ["_overall"]
    rng = random.Random(SEED)
    print(f"paired compare A={pa} B={pb} n={len(ids)}")
    for d in doms:
        sel = [i for i in ids if d == "_overall" or ba[i]["domain"] == d]
        if not sel:
            continue
        sa = sum(ba[i]["sum_nll"] for i in sel); ta = sum(ba[i]["n_tok"] for i in sel)
        sb = sum(bb[i]["sum_nll"] for i in sel); tb = sum(bb[i]["n_tok"] for i in sel)
        point = sa / ta - sb / tb
        diffs = []
        for _ in range(iters):
            pick = [sel[rng.randrange(len(sel))] for _ in sel]
            xa = sum(ba[i]["sum_nll"] for i in pick); na = sum(ba[i]["n_tok"] for i in pick)
            xb = sum(bb[i]["sum_nll"] for i in pick); nb = sum(bb[i]["n_tok"] for i in pick)
            diffs.append(xa / na - xb / nb)
        diffs.sort()
        lo, hi = diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]
        print(f"  {d:16s} n={len(sel):3d} A_nll={sa/ta:.4f} B_nll={sb/tb:.4f} "
              f"A-B={point:+.4f} CI[{lo:+.4f},{hi:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--delta", default=None)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    assert args.out, "--out required for an arm run"
    run_arm(args)


if __name__ == "__main__":
    main()
