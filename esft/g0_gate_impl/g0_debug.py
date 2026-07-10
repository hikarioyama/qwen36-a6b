#!/usr/bin/env python
"""G0 single-batch debug / anti-silent-no-op gate (1 GPU, ~2 min).

Loads the 35B at top-k 32, runs ONE real MMLU batch through it, captures the
router's own ``(logits, scores, indices)`` at a mid-stack gate via a temporary
hook, then:

  1. proves the shaping recompute is exact -- ``shape_router_scores(logits, k,
     'temp', tau=1.0)`` must reproduce the model's own top-k weights bit-for-bit
     (this is the fp32-path / OFF-equivalence proof: if this fails, the OFF arm is
     NOT bit-identical and the whole sweep is void);
  2. prints, for every sweep variant, the gate-mass distribution it produces on
     the SAME real logits -- effective active experts, rank1..8 head mass, rank1
     weight -- so a "the flag did nothing" bug is impossible to miss.

Run on aux-host (needs ONE free GPU):
    G0_GPU=0 ~/esft-work/venv/bin/python g0_debug.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/esft"))
import gate_shaping as gs  # noqa: E402
from esft_qwen.common import find_moe_blocks  # noqa: E402

MODEL_35B = os.path.expanduser("~/esft-work/models/Qwen3.6-35B-A3B")
TOPK = 32
PROBE_LAYER_ORDINAL = -1   # which gate to probe; -1 == a deep block
SWEEP = [("temp", 0.5), ("temp", 0.7),
         ("masscut", 0.80), ("masscut", 0.90),
         ("rankdamp", 0.25), ("rankdamp", 0.5)]

MMLU_PROMPTS = [
    "The capital of France is\n\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid\n\n"
    "Answer with the letter (A, B, C, or D) of the correct option.",
    "Water is chemically\n\nA. H2O\nB. CO2\nC. NaCl\nD. O2\n\n"
    "Answer with the letter (A, B, C, or D) of the correct option.",
    "The derivative of x^2 is\n\nA. 2x\nB. x\nC. x^3/3\nD. 2\n\n"
    "Answer with the letter (A, B, C, or D) of the correct option.",
    "A base has pH\n\nA. below 7\nB. exactly 7\nC. above 7\nD. undefined\n\n"
    "Answer with the letter (A, B, C, or D) of the correct option.",
]


def main():
    gpu = int(os.environ.get("G0_GPU", "0"))
    dev = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)

    from transformers import (AutoTokenizer, AutoModelForImageTextToText,
                              AutoModelForCausalLM)
    tok = AutoTokenizer.from_pretrained(MODEL_35B)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_35B, dtype=torch.bfloat16, device_map={"": gpu})
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_35B, dtype=torch.bfloat16, device_map={"": gpu})
    model.eval()

    refs = find_moe_blocks(model)
    for r in refs:
        r.gate.top_k = TOPK
    probe = refs[PROBE_LAYER_ORDINAL]
    print(f"probing gate at layer {probe.layer_idx} (top_k={probe.gate.top_k}), "
          f"{len(refs)} MoE layers total", flush=True)

    captured = {}

    def capture(module, inputs, output):
        rl, rs, ri = output
        captured["logits"] = rl.detach().float().cpu()
        captured["scores"] = rs.detach().float().cpu()   # model's own top-k weights
        captured["idx"] = ri.detach().cpu()

    h = probe.gate.register_forward_hook(capture)

    prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
               for p in MMLU_PROMPTS]
    enc = tok(prompts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    with torch.no_grad():
        model(**enc)
    h.remove()

    logits = captured["logits"]
    model_scores = captured["scores"]     # (N, 32) fp32, renormalised
    n_tok = logits.shape[0]
    print(f"\ncaptured {n_tok} token router rows, {logits.shape[1]} experts\n")

    # ---- (1) exact recompute / OFF-equivalence proof ----------------------
    base_val, shaped_id, idx = gs.shape_router_scores(logits, TOPK, "temp", 1.0)
    max_abs = (base_val - model_scores).abs().max().item()
    # bit-identity at bf16 (the dtype experts() actually consumes)
    bit_ok = torch.equal(base_val.to(torch.bfloat16), model_scores.to(torch.bfloat16))
    print(f"[OFF-equivalence] tau=1.0 recompute vs model's own scores: "
          f"max|Δ|(fp32)={max_abs:.2e}  bf16_bit_identical={bit_ok}")
    if not bit_ok:
        print("  !! WARNING: OFF arm would NOT be bit-identical to stock -- "
              "investigate before trusting the sweep.")

    # ---- (2) per-variant distribution table -------------------------------
    def stats(val):
        eff_k = (val > 0).sum(dim=-1).float().mean().item()
        head = val[:, :gs.HEAD_K].sum(dim=-1).mean().item()
        r1 = val[:, 0].mean().item()
        tail = val[:, gs.HEAD_K:].sum(dim=-1).mean().item()
        return eff_k, head, r1, tail

    b_eff, b_head, b_r1, b_tail = stats(base_val)
    print(f"\n{'variant':>16} | {'eff_k':>6} | {'head(1-8)':>9} | "
          f"{'tail(9-32)':>10} | {'rank1_wt':>8}")
    print("-" * 66)
    print(f"{'k32 stock (OFF)':>16} | {b_eff:>6.2f} | {b_head:>9.4f} | "
          f"{b_tail:>10.4f} | {b_r1:>8.4f}")
    for variant, param in SWEEP:
        _bv, sv, _idx = gs.shape_router_scores(logits, TOPK, variant, param)
        e, hd, r1, tl = stats(sv)
        print(f"{variant + ' ' + str(param):>16} | {e:>6.2f} | {hd:>9.4f} | "
              f"{tl:>10.4f} | {r1:>8.4f}")
    print("\nRead: head mass should RISE and tail mass FALL vs stock for every "
          "variant; masscut's eff_k should drop below 32 (dynamic-k). If a row "
          "equals the stock row, that variant is a silent no-op.")


if __name__ == "__main__":
    main()
