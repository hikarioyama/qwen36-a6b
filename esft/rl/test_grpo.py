"""CPU-only unit tests for the GRPO update loop (grpo_train.py). No GPU, no real
weights: a tiny real ``Qwen3_5MoeForCausalLM`` (2 layers, 8 experts, hidden 32,
vocab 64) exercises the delta mechanism end to end, and a mock chat-template
tokenizer exercises the prefill re-stitching.

Checks (mirrors the task spec):
  (a) group_advantages matches hand computation, std==0 group skipped;
  (b) delta toggle changes completion logp (ON != OFF) for a nonzero delta;
  (c) toy overfit: loss drops over a few dozen steps (grad reaches the deltas);
  (d) router + non-selected experts get no gradient (grad is None/zero);
  (e) prefill re-stitch: full_assistant tokens == prefill+completion tokens, and
      n_prefix lands exactly on the prefill/completion boundary.

Run:  CUDA_VISIBLE_DEVICES="" <venv>/bin/python rl/test_grpo.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # rl/
sys.path.insert(0, os.path.dirname(HERE))     # esft/ (for esft_qwen)

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from esft_qwen.delta_patch import to_esft_delta
from grpo_train import (
    group_advantages,
    delta_disabled,
    completion_logps,
    completion_loss,
    build_token_ids,
)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# Tiny real model + expert config
# --------------------------------------------------------------------------- #

HIDDEN = 32
NUM_EXPERTS = 8
TOP_K = 2
MOE_INTER = 16
VOCAB = 64
N_LAYERS = 2

EXPERT_CFG = {"experts": {"0": [0, 3], "1": [5]},
              "shared_experts": False, "non_expert_modules": False}


def build_causal_lm(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=MOE_INTER,
        moe_intermediate_size=MOE_INTER,
        shared_expert_intermediate_size=MOE_INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        max_position_embeddings=256,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    model = Qwen3_5MoeForCausalLM(cfg).to(torch.float32)
    for p in model.parameters():
        if p.dim() >= 1:
            nn.init.normal_(p, mean=0.0, std=0.05)
    model.config.use_cache = False
    return model


def randomize_deltas(handles, scale=0.5, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in handles.expert_params:
            p.copy_(torch.randn(p.shape, generator=g) * scale)


# --------------------------------------------------------------------------- #
# Mock tokenizer with a deterministic, invertible chat template
# --------------------------------------------------------------------------- #

class MockTokenizer:
    """Char-level tokenizer + a simple chat template. The template concatenates
    ``<|role|>content`` per message; an open assistant turn (continue_final_message)
    is rendered by simply appending its content with no trailing close tag, so
    ``prefill`` then ``prefill+gen`` share an exact token prefix -- the property
    build_token_ids relies on. Byte==token because each char maps to one id."""

    def __init__(self):
        chars = ("<|>/system user assistant"
                 "abcdefghijklmnopqrstuvwxyz0123456789 \n.#=+-")
        self.vocab = {c: i for i, c in enumerate(dict.fromkeys(chars))}
        self._next = len(self.vocab)

    def _id(self, c):
        if c not in self.vocab:
            self.vocab[c] = self._next
            self._next += 1
        return self.vocab[c]

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, continue_final_message=False):
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>{m['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = "".join(parts)
        return text  # tokenize=False path only

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(c) for c in text]}


# --------------------------------------------------------------------------- #
# (a) group advantage
# --------------------------------------------------------------------------- #

def test_a_group_advantage():
    r = [0.0, 1.0, 2.0, 3.0]
    adv, skip = group_advantages(r)
    t = torch.tensor(r)
    exp = (t - t.mean()) / (t.std(unbiased=False) + 1e-6)
    check("a.advantage matches hand calc", torch.allclose(adv, exp, atol=1e-5),
          f"{adv.tolist()}")
    check("a.advantage mean ~ 0", abs(float(adv.mean())) < 1e-5)
    _, skip2 = group_advantages([0.5, 0.5, 0.5])
    check("a.std==0 group skipped", skip2 is True)
    _, skip3 = group_advantages([-1.0, -1.0])
    check("a.all-fail group skipped", skip3 is True)


# --------------------------------------------------------------------------- #
# (b) delta toggle changes logp
# --------------------------------------------------------------------------- #

def test_b_delta_toggle_changes_logp():
    model = build_causal_lm(seed=0)
    handles = to_esft_delta(model, EXPERT_CFG)
    randomize_deltas(handles, scale=0.5, seed=2)
    ids = [3, 7, 12, 5, 20, 9, 4, 15]
    n_prefix = 3
    dev = torch.device("cpu")
    with torch.no_grad():
        on = completion_logps(model, ids, n_prefix, dev).sum()
        with delta_disabled(handles):
            off = completion_logps(model, ids, n_prefix, dev).sum()
    check("b.delta ON != OFF logp", not torch.allclose(on, off, atol=1e-4),
          f"on={float(on):.4f} off={float(off):.4f}")

    # zero deltas -> ON == OFF (delta scheme is a no-op at zero init)
    with torch.no_grad():
        for p in handles.expert_params:
            p.zero_()
        on0 = completion_logps(model, ids, n_prefix, dev).sum()
        with delta_disabled(handles):
            off0 = completion_logps(model, ids, n_prefix, dev).sum()
    check("b.zero delta -> ON == OFF", torch.allclose(on0, off0, atol=1e-5),
          f"on0={float(on0):.5f} off0={float(off0):.5f}")


# --------------------------------------------------------------------------- #
# (c) toy overfit: loss drops
# --------------------------------------------------------------------------- #

def test_c_overfit():
    model = build_causal_lm(seed=0)
    handles = to_esft_delta(model, EXPERT_CFG)
    dev = torch.device("cpu")
    ids = [3, 7, 12, 5, 20, 9, 4, 15, 8, 2]
    n_prefix = 4
    adv = torch.tensor(1.0)  # push logp of these completion tokens UP
    opt = torch.optim.SGD(handles.expert_params, lr=0.5)

    first = last = None
    for step in range(40):
        opt.zero_grad()
        out = completion_loss(model, handles, ids, n_prefix, adv, beta=0.0,
                              device=dev, compute_kl=False)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = float(out["loss"])
        last = float(out["loss"])
    check("c.loss decreased over 40 steps", last < first - 1e-3,
          f"first={first:.4f} last={last:.4f}")


# --------------------------------------------------------------------------- #
# (d) router + non-selected experts get no gradient
# --------------------------------------------------------------------------- #

def test_d_grad_freezing():
    model = build_causal_lm(seed=0)
    handles = to_esft_delta(model, EXPERT_CFG)
    dev = torch.device("cpu")
    ids = [3, 7, 12, 5, 20, 9, 4, 15]
    n_prefix = 3
    out = completion_loss(model, handles, ids, n_prefix, torch.tensor(1.0),
                          beta=0.02, device=dev, compute_kl=True)
    out["loss"].backward()

    # every non-delta parameter must have no gradient (frozen structurally)
    bad = []
    for name, p in model.named_parameters():
        is_delta = name.endswith(("delta_gate_up", "delta_down"))
        if is_delta:
            continue
        if p.requires_grad:
            bad.append(f"{name}:requires_grad")
        if p.grad is not None and torch.count_nonzero(p.grad) > 0:
            bad.append(f"{name}:nonzero_grad")
    check("d.router/non-selected experts: no grad", not bad, str(bad[:3]))

    # deltas DID receive gradient
    got = [p.grad is not None and torch.count_nonzero(p.grad) > 0
           for p in handles.expert_params]
    check("d.deltas received gradient", all(got), f"{sum(got)}/{len(got)}")

    # packed expert tensors are frozen (no requires_grad -> never trained)
    refs = handles.experts_modules
    check("d.packed expert tensors frozen",
          all(not e.gate_up_proj.requires_grad and not e.down_proj.requires_grad
              for e in refs.values()))


# --------------------------------------------------------------------------- #
# (e) prefill re-stitch byte/token exact
# --------------------------------------------------------------------------- #

def test_e_prefill_restitch():
    tok = MockTokenizer()
    prompt = [{"role": "system", "content": "sys"},
              {"role": "user", "content": "fix the bug"}]
    prefill = "<think>\n"
    gen = "reason</think>\n<solution>edit</solution>"
    full = prefill + gen

    input_ids, n_prefix = build_token_ids(tok, prompt, full, prefill)

    # full render == prompt-open + full_assistant chars, so ids == char ids of that
    expected_full_text = tok.apply_chat_template(
        prompt + [{"role": "assistant", "content": full}], continue_final_message=True)
    expected_ids = tok(expected_full_text)["input_ids"]
    check("e.full ids == chat-template(prompt+full)", input_ids == expected_ids)

    # the completion region decodes back to exactly gen
    prefix_text = tok.apply_chat_template(
        prompt + [{"role": "assistant", "content": prefill}], continue_final_message=True)
    prefix_ids = tok(prefix_text)["input_ids"]
    check("e.n_prefix at prefill boundary", n_prefix == len(prefix_ids),
          f"n_prefix={n_prefix} len(prefix)={len(prefix_ids)}")

    gen_ids = input_ids[n_prefix:]
    inv = {v: k for k, v in tok.vocab.items()}
    decoded = "".join(inv[i] for i in gen_ids)
    check("e.completion tokens decode to gen", decoded == gen, repr(decoded[:40]))

    # prefill is a strict token prefix of full
    check("e.prefill is a token prefix of full", input_ids[:n_prefix] == prefix_ids)


def main():
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    test_a_group_advantage()
    test_b_delta_toggle_changes_logp()
    test_c_overfit()
    test_d_grad_freezing()
    test_e_prefill_restitch()
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'='*60}\nGRPO TESTS: {n_pass}/{len(results)} passed")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    main()
