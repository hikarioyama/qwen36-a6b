#!/usr/bin/env python
"""Top-k sweep: quality (GSM8K acc) + speed (tok/s) vs num_experts_per_tok.

Model: Qwen3.6-35B-A3B (BF16). K in {8,12,16,24,32}.

top-k override
--------------
K is set by mutating ``gate.top_k`` on every routed-MoE block. That attribute is
read at call time inside ``Qwen3_5MoeTopKRouter.forward``
(``torch.topk(router_probs, self.top_k)``); ``Qwen3_5MoeExperts.forward`` then
keys off the shape of the returned indices. So the routing math is byte-identical
to the shipped model -- softmax over all 256 experts (fp32) -> topk(K) ->
UNCONDITIONAL renorm of the top-K weights -- exactly the path
``collect_router_stats.py`` exercises via ``config.num_experts_per_tok``, at the
same code but a cheaper lever (no per-K reload). Proven by tests/verify_topk_override.py.

Data parallelism
----------------
Two full BF16 copies, one per GPU (~70GB each, fits in 96GB). The 300 problems are
split even/odd across the two GPUs and generated concurrently, so both GPUs run hot
(target NVTOP util >70%). Per-K partial results are flushed to JSON after each GPU
reports, so a crash mid-sweep still leaves everything finished so far on disk.

Eval protocol
-------------
GSM8K test, first N=300 (deterministic; seed only governs an optional shuffle,
default off => reproducible first-N). 0-shot chat prompt + "Let's think step by
step.", greedy (do_sample=False), max_new_tokens=512. Answer = number after the
last ####, else the last number in the post-</think> segment; exact numeric match
vs gold (#### in the reference answer).

Run:  ~/esft-work/venv/bin/python ~/esft/topk_sweep.py
Opts: --n 300 --batch-size 16 --max-new 512 --ks 8,12,16,24,32
"""
from __future__ import annotations

import os
import sys
import re
import json
import time
import argparse
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.expanduser("~/esft-work/models/Qwen3.6-35B-A3B")
OUT = os.path.expanduser("~/esft/reports/topk_sweep.json")
EOS_IDS = [248046, 248044]        # <|im_end|>, <|endoftext|>


# ----------------------------- data / scoring -----------------------------

def _last_number(text):
    seg = text.split("</think>")[-1] if "</think>" in text else text
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", seg)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", seg)
    return nums[-1].replace(",", "") if nums else None


def _gold(answer):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
    return m.group(1).replace(",", "") if m else None


def _num_eq(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a.strip() == b.strip()


def build_problems(n, seed, shuffle):
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main", split="test")
    if shuffle:
        d = d.shuffle(seed=seed)
    d = d.select(range(min(n, len(d))))
    return [(row["question"], _gold(row["answer"])) for row in d]


# ----------------------------- worker -----------------------------

def worker(gpu_id, problems, ks, batch_size, max_new, q):
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM)
    sys.path.insert(0, PROJECT_ROOT)
    from esft_qwen.common import find_moe_blocks

    torch.cuda.set_device(gpu_id)
    dev = f"cuda:{gpu_id}"

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map={"": gpu_id})
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map={"": gpu_id})
    model.eval()

    refs = find_moe_blocks(model)
    print(f"[gpu{gpu_id}] loaded  moe_layers={len(refs)}  problems={len(problems)}", flush=True)

    rendered = [
        tok.apply_chat_template(
            [{"role": "user", "content": qq + "\nLet's think step by step."}],
            add_generation_prompt=True, tokenize=False)
        for qq, _ in problems
    ]
    golds = [g for _, g in problems]

    for k in ks:
        for r in refs:
            r.gate.top_k = int(k)
        assert all(int(r.gate.top_k) == k for r in refs)

        correct = 0
        gen_tokens = 0
        n = 0
        t0 = time.time()
        for i in range(0, len(rendered), batch_size):
            chunk = rendered[i:i + batch_size]
            gchunk = golds[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=False,
                    eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id)
            gen = out[:, in_len:]
            for row, gold in zip(gen, gchunk):
                gt = row.tolist()
                hit = [gt.index(e) for e in EOS_IDS if e in gt]
                ntok = (min(hit) + 1) if hit else len(gt)
                gen_tokens += ntok
                pred = _last_number(tok.decode(row, skip_special_tokens=True))
                correct += int(_num_eq(pred, gold))
                n += 1
            print(f"[gpu{gpu_id} k={k}] {n}/{len(rendered)} acc={correct/max(n,1):.3f} "
                  f"tok={gen_tokens}", flush=True)
        dt = time.time() - t0
        q.put({"gpu": gpu_id, "k": k, "correct": correct, "n": n,
               "gen_tokens": gen_tokens, "gen_time": dt})
    q.put({"gpu": gpu_id, "done": True})


# ----------------------------- aggregation / flush -----------------------------

def aggregate(partial):
    """partial: {k: {gpu_id: msg}} -> {'k=..': {...}} for k with BOTH gpus present."""
    out = {}
    for k in sorted(partial):
        parts = partial[k]
        if len(parts) < 2:
            continue
        c = sum(p["correct"] for p in parts.values())
        n = sum(p["n"] for p in parts.values())
        g = sum(p["gen_tokens"] for p in parts.values())
        tsum = sum(p["gen_time"] for p in parts.values())
        tmax = max(p["gen_time"] for p in parts.values())
        out[f"k={k}"] = {
            "acc": round(c / n, 4) if n else None,
            "correct": c,
            "n": n,
            "gen_tokens": g,
            "tok_s": round(g / tsum, 2) if tsum else None,          # per-GPU-equiv decode rate
            "tok_s_parallel": round(g / tmax, 2) if tmax else None,  # 2-GPU wall-clock throughput
            "gen_time_sum_s": round(tsum, 1),
        }
    return out


def flush(partial, meta):
    data = {"_meta": meta, **aggregate(partial)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUT)


def print_table(partial):
    agg = aggregate(partial)
    print("\n" + "=" * 62)
    print(f"{'K':>4} | {'acc':>7} | {'n':>4} | {'tok/s':>9} | {'tok/s(2gpu)':>12}")
    print("-" * 62)
    for key in sorted(agg, key=lambda s: int(s.split("=")[1])):
        r = agg[key]
        print(f"{key.split('=')[1]:>4} | {r['acc']:>7} | {r['n']:>4} | "
              f"{r['tok_s']:>9} | {r['tok_s_parallel']:>12}")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--ks", default="8,12,16,24,32")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle test set with --seed (default: deterministic first-N)")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    gpus = [int(x) for x in args.gpus.split(",")]
    assert len(gpus) == 2, "this script is wired for exactly 2 GPUs"

    problems = build_problems(args.n, args.seed, args.shuffle)
    print(f"loaded {len(problems)} problems  ks={ks}  batch={args.batch_size}  "
          f"max_new={args.max_new}  gpus={gpus}", flush=True)

    # even/odd split across the two GPUs
    subsets = {gpus[0]: problems[0::2], gpus[1]: problems[1::2]}
    meta = {
        "model": MODEL, "n": len(problems), "ks": ks,
        "batch_size": args.batch_size, "max_new": args.max_new,
        "prompt": "0-shot chat + 'Let's think step by step.', greedy",
        "split": "gsm8k/test first-N" + (" shuffled" if args.shuffle else ""),
        "seed": args.seed,
        "gpus": gpus, "split_sizes": {str(g): len(s) for g, s in subsets.items()},
        "note": "reasoning model; 512 tokens may truncate <think> -- see verify report",
    }

    q = mp.Queue()
    procs = [mp.Process(target=worker,
                        args=(g, subsets[g], ks, args.batch_size, args.max_new, q))
             for g in gpus]
    for p in procs:
        p.start()

    partial = {}
    done = set()
    while len(done) < len(procs):
        msg = q.get()
        if msg.get("done"):
            done.add(msg["gpu"])
            continue
        partial.setdefault(msg["k"], {})[msg["gpu"]] = msg
        flush(partial, meta)
        agg = aggregate(partial)
        key = f"k={msg['k']}"
        if key in agg:
            print(f">>> K={msg['k']} COMPLETE  acc={agg[key]['acc']}  "
                  f"tok/s={agg[key]['tok_s']}  (flushed to {OUT})", flush=True)

    for p in procs:
        p.join()

    flush(partial, meta)
    print_table(partial)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
