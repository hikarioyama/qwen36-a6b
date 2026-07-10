# JMMLU paired pilot — 2026-07-10

## Status

**COMPLETE.** This remains a Japanese-axis harness pilot, not a model adoption
or protocol-freeze decision.  The requested B2-1000 artifact was used only
after its SHA-256 and tensor count were verified; no B2-500 substitute was
used.

## Frozen-for-this-attempt pilot conditions

| Field | Value |
|---|---|
| Benchmark | `nlp-waseda/JMMLU`, four-choice deterministic choice-logprob scoring |
| Subset | `n=300`, shuffle seed `0`, first 300 after shuffle |
| Subset identity | ordered `item_key` SHA-256 `28256a017617f07820d2271065da2f5cbf9aabbd5a96384598e4b78278afaeac` |
| Base arm | verified true stock revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, top-k 8 |
| Base tensor fingerprint | `3a1ca2a61e9a86af44c5114d72a9033504d3a20e27c3c6838f4162b87e3aa315` |
| B2 arm required | B2-1000 expert patch, top-k 32, SHA-256 `c1b3f041051e9c184e5a3ea14126f921e3a2619b29454e3e73b96f79f45199d3` |
| Paired condition | same item keys, same seed, same local machine/harness/batch size; serial base then B2 |
| GPUs | local physical GPU 0 and 1 only (`CUDA_VISIBLE_DEVICES=0,1`); GPU 2 excluded |
| Predeclared harness margin | 0.02, inherited only to satisfy the campaign manifest; this pilot has **no adoption or protocol-freeze decision** |

The recorded outcome fields are `B2 - base` paired accuracy difference, paired
95% Wald CI, exact McNemar p-value, discordant counts, and truncation counts.
JMMLU choice-logprob does not generate text; actual truncation count was zero
in both arms.

## Harness integration

`esft/eval_harness.py` already registered `Jmmlu` as a `Mmlu` subclass. It reads
the canonical `JMMLU.zip` test CSVs, applies the established deterministic
`datasets` shuffle/select sequence, and inherits the exact A/B/C/D choice-logprob
prompt and scorer without a generative answer-extraction path.

This bring-up adds `[protocols.jmmlu]` to `esft/codex_harness.toml` and admits
`jmmlu` to `codex_harness.py campaign`.  The protocol is `n=300`, seed 0,
shuffle enabled, batch size 16, no-think, choice-logprob.  A structural dry-run
emitted the two serial commands and the CPU test suite passed 21/21.  A dedicated
local-only B2-1000 config then passed full preflight: GPU 0/1 availability,
true-stock identity, B2-1000 SHA/tensor identity, no active evaluator, and the
code sandbox.  GPU 2 was never selected.

## Execution and paired result

Run directory: `esft/reports/eval/codex_runs/20260710_jmmlu_b2_1000_pilot_v1/`.
The manifest is `complete`; base ran first and B2 second on physical GPUs 0/1
only (`CUDA_VISIBLE_DEVICES=0,1`).  Run time was 2026-07-10 13:31:50–13:32:38
UTC.  The campaign rechecked true-stock identity before and after each arm and
the B2 patch identity before and after its arm.

| Arm (same n=300, seed 0, shuffled, choice-logprob) | Correct | Accuracy | Truncated |
|---|---:|---:|---:|
| true stock base@k8 | 225 | 0.7500 | 0 |
| B2-1000 expert patch@k32 | 220 | 0.7333 | 0 |

- Paired difference, B2 minus base: **−0.0167** (−1.67pt); paired 95% Wald CI
  **[−0.0451, +0.0117]**.
- Discordant outcomes: base-only 12, B2-only 7.  Exact McNemar `p=0.3593`
  (`significant=false`).  The n=300 result therefore does **not** resolve a
  JMMLU difference.
- The predeclared 0.02 non-inferiority margin is **INCONCLUSIVE**: the
  conditional Clopper–Pearson/Bonferroni lower 95% bound is −0.0731, below the
  −0.02 boundary.  This pilot makes no adoption decision.
- Both item files contain the same 300 unique keys in the same order.  Their
  newline-terminated ordered-key SHA-256 is the frozen
  `28256a017617f07820d2271065da2f5cbf9aabbd5a96384598e4b78278afaeac`.
  (The no-final-newline representation is
  `aee1bf7fff0b1bd5db18748f5ae45ffb8eb0a42df9158367440aa613ed4c40b3`.)

## Dataset and licence

The local cache was fetched from `nlp-waseda/JMMLU` revision
`3637b25e444ccfdcde4d23a783cbe8e674faa01b` on 2026-07-10.  Its `JMMLU.zip`
SHA-256 is
`3ba7d912943ede44fb7ec06aa1df067ac6bee157e65c5588736b9b983f0e684d`.

The dataset card currently labels the dataset **CC BY-NC-ND 4.0** and additionally
states subject-specific research/evaluation restrictions for some Japanese
history, geography, civics, idiom material.  It is therefore evaluation-only in
this campaign and is not included in training data or redistributed.  The
repository history contains older, inconsistent licence text; the pinned
revision and its dataset card are the applicable provenance for this attempt.

Source: <https://huggingface.co/datasets/nlp-waseda/JMMLU/blob/3637b25e444ccfdcde4d23a783cbe8e674faa01b/README.md>.

## Blocker resolution

The requested B2-1000 patch was subsequently placed at
`~/models/esft/b2_1000/expert_patch.safetensors`.  Its SHA-256 was
verified as `c1b3f041051e9c184e5a3ea14126f921e3a2619b29454e3e73b96f79f45199d3`
and its safetensors tensor count as 1,666 before launch and again by campaign
preflight.  No remote node, GPU 2, or pre-existing checkpoint/evaluation asset
was used or overwritten.
