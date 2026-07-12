# Qwen3.6-35B-A6B — expanding MoE top-k from 8 to 32

A research log of a single question: **if you activate more experts per token in a
pretrained MoE, can you turn the extra compute into extra capability?**

The base model, `Qwen3.6-35B-A3B`, routes each token to the **top-8 of 256**
experts (~3B active parameters). Setting the inference top-k to **32** activates
~6B ("A6B"). This repository documents the campaign to make that k=32 model
*beat* the base — the trainer, the eval harness, the data pipeline, and, most
importantly, the honest measurements (every number carries `n` and its
comparison condition).

---

## TL;DR — the α dial

Naively raising inference `k` from 8 to 32 **costs accuracy** before any training:

> base@k32 vs base@k8: **−3.17 pt** MMLU (489/600 vs 508/600, same items, paired,
> exact `p=0.013`).

The cause is not broken experts and not worse routing. It is purely **softmax
renormalization**: adding 24 more experts to the normalized mixture dilutes the
weight the original top-8 receive. The nesting is exact — *the top-8 of a k=32
selection are the same 8 experts as k=8*.

The fix is a runtime operation with **no weight change**, the **α dial**
(`--router-tail-scale`): for each token, rank the selected gate scores, multiply
ranks 9–32 by `α`, renormalize.

- `α = 1` is plain k=32.
- **`α = 0` is mathematically identical to k=8** (top-8 renormalization).
- Intermediate `α` interpolates between k8-like and k32 mixing.

Sweeping α recovers the entire debt with **zero training** (MMLU `n=600`, same
items, paired):

| α (base model) | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| MMLU % | 84.67 | 84.33 | **84.50** | 82.83 | 81.50 |
| Δ vs α=1 (paired) | — | +2.83 | **+3.00 [+1.35,+4.65]** | +1.33 | — |

The curve is flat on `[0, 0.5]`, then collapses. The `−3.17 pt` "k32 tax" is fully
repaid at any `α ≤ 0.5`. Verification: `α = 0` and `base@k8` produce identical
predictions on 98/100 items — bf16 noise level, matching the 98.7% cross-machine
agreement of the *same* weights.

Full detail: [`reports/ALPHA_DIAL_20260712.md`](reports/ALPHA_DIAL_20260712.md).

---

## Why the dial changes the pipeline

Because calibration is now a **dial retuned by re-sweeping α** (no gradients),
the design splits cleanly:

- **Calibration → the α dial.** Retune it for free at every interval boundary,
  and ship it as an inference-server setting.
- **Capability → full-FFN training, router frozen.** Training capacity is spent
  on the experts, not on relearning a mixing ratio the dial already fixes. This
  retired the earlier joint router-training path and its anchor-KL machinery.
- **Progress meter → the optimal α over time.** As k=32 capacity is unlocked, the
  best α should rise and eventually an α should appear that **beats the k8 floor
  (84.67)**. Boundary sweeps are the progress meter, not the product.
- **Honest floor.** The real floor is **base@k8**, not base@k32 — beating a model
  you first made *worse* is not a win. Every verdict reports vs base@k8 *and* vs
  base@k32.

A mechanism finding drove this turn: under joint router training the pre-top-k
routing entropy **flattens monotonically** (5.145 → 5.328 over 1000 steps; the
mass carried by the top-32 fell 37.6% → 29.1%). MMLU recovery there came from FFN
adaptation, not router repair — evidence that recalibrating the router by
gradient is the wrong lever, and the dial is the right one.

---

## Repository layout

| Path | What |
|---|---|
| [`DEVLOG.md`](DEVLOG.md) | Chronological decisions, measurements (with `n`/CI), dead ends and lessons |
| [`esft/PLAN.md`](esft/PLAN.md) | Master plan |
| [`esft/eval_harness.py`](esft/eval_harness.py) | Choice-logprob / generation eval, paired McNemar verdicts, the `--router-tail-scale` (α) hook |
| [`esft/deploy/train_fullffn_dcp.py`](esft/deploy/train_fullffn_dcp.py) | Full-FFN / router-only DCP trainer; `--router-tail-scale`, `--tokenize-mode offsets`, frozen-router mode |
| [`esft/router_rank_baseline.py`](esft/router_rank_baseline.py) | Router health probe: rank-mass, entropy, expert-usage histogram |
| [`esft/judge/`](esft/judge) | Rubric-driven second-stage quality selection over machine-verified data |
| [`esft/selfgen_toolcall_intent_v1.py`](esft/selfgen_toolcall_intent_v1.py) | Intent-level tool-call self-generation (T1–T4 difficulty tiers, distractors) |
| [`esft/corpus_decontam_v1.py`](esft/corpus_decontam_v1.py) | External-corpus intake + n-gram eval-decontamination gate |
| [`esft/rl/`](esft/rl) | GRPO / SWE-RL reward and prompt tooling (`RL_DESIGN.md`) |
| [`reports/`](reports) | Standalone study reports (α dial, data-quality strategy, pilots) |
| [`esft/reports/eval/`](esft/reports/eval) | Eval result JSON, per-item, for paired re-analysis |

`esft/deploy/` mirrors the launcher/trainer copies deployed to the training
host; the two copies are kept byte-identical.

## Reproducing the CPU checks

The unit tests are CPU-only and load no model weights (some skip if the pinned
tokenizer snapshot is absent):

```bash
cd esft
python3 tests/test_fullffn_joint_trainer.py      # α-dial + router-only, 12 tests
python3 tests/test_corpus_to_trainer_v1.py       # converter + offsets tokenization
python3 tests/test_selfgen_toolcall_intent_v1.py # intent-tier selfgen
python3 tests/test_corpus_judge_v1.py            # judge ledger contract
```

---

## Honest status

**Measured (same-condition, paired, `n` stated):**

- k=32 tax vs base@k8: −3.17 pt MMLU (`n=600`, `p=0.013`).
- α dial repays it: α≤0.5 is statistically level with base@k8; α=0.5 is +3.00 pt
  over α=1 (`n=600` paired).
- α=0 ≡ base@k8: 98/100 identical predictions (bf16 noise).
- Joint router training flattens routing entropy monotonically over 1000 steps.

**Hypothesized / not yet measured:**

- Whether FFN training can lift the *optimal-α* accuracy **above** the 84.67 k8
  floor — the single dial only recovers the floor; exceeding it is the FFN's job.
- α=0.5 harmlessness on the other four eval axes (only MMLU is swept so far).
- Layer-wise α and α annealing (untested upside).

**Not in this repo:** training data, model weights, and rollouts are excluded by
[`.gitignore`](.gitignore) for size and provenance reasons. What ships is the
**pipeline code, harnesses, and the measured record** — data lineage and
processing are documented in `DEVLOG.md` and the reports. See
[`reports/DATA_QUALITY_STRATEGY_20260711.md`](reports/DATA_QUALITY_STRATEGY_20260711.md)
for the data strategy and its eval-integrity gates.

---

## Naming

- Repository / host identifier: `qwen36-a6b`
- Base model: `Qwen3.6-35B-A3B`; expanded target: `Qwen3.6-35B-A6B (k=32)`

Identifiers from the abandoned 285B candidate are not reused for this project.
References to `deepseek-ai/ESFT`, DeepSeek-V2-Lite, DeepSeekMath and other
upstream work retain their actual names. Machine-specific paths, runtime
manifests, and host configuration are kept local and untracked; where launcher
scripts must name a path, it is overridable via environment variable.

The development log names three machines by **role**, not network identity:
`gpu-host` (the multi-GPU training reactor), `aux-host` (a secondary GPU host
for control arms and CPU data work), and `local` (the fixed evaluation machine).
The `esft/deploy/` directory holds the launcher/trainer copies staged onto
`gpu-host`.
