# BFCL tool-call paired pilot — 2026-07-10

## Status

**RESUMED AND COMPLETED — pilot only; not a protocol freeze or adoption
decision.**  The former GPU/source/wheel blockers were resolved locally and the
same-condition serial measurement below completed on 2026-07-10.  The original
blocked bring-up is retained as history so its `n=0` state is not confused with
the completed run.

## Completed resumed pilot (2026-07-10)

| Field | Frozen condition |
|---|---|
| Run | `esft/reports/eval/codex_runs/20260710_bfcl_b2_1000_pilot_v1/` (manifest SHA-256 `066e7b79…f5baa04`) |
| Benchmark | BFCL v4 deterministic non-live AST subset; all live, execution, memory, multi-turn, and `web_search`/external-API categories excluded |
| Source | pinned Gorilla `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`; local `bfcl-eval` 2026.3.23 wheel in `external/bfcl-venv` |
| Subset | `n=300`, all eligible entries sorted by `(category,id)`, global `random.Random(0).shuffle()`, then first 300; frozen before either arm |
| Allocation | simple Python 114; Java 26; JavaScript 14; parallel 50; multiple 44; parallel-multiple 52 |
| Arms | true-stock revision `995ad…97b0` (expert fingerprint `3a1ca2a6…e3aa315`) base@k8, then B2-1000 patch@k32; B2 SHA-256 `c1b3f041…f45199d3`, 1,666 tensors rechecked |
| Runtime | local GPU 0/1 only, serial arms, BF16, no-think native Qwen template, greedy, batch 4, `max_new=512`, offline; GPU 2/gpu-host/aux-host unused |
| Scoring | native `<tool_call><function=…><parameter=…>` is normalized by the thin adapter, then the pinned upstream `ast_parse` and `ast_checker` score every item; the wheel's unused optional `soundfile` import was stubbed only to import that checker (recorded in item metadata) |

### Result (same-condition paired)

| Arm | Correct | Accuracy | Truncated |
|---|---:|---:|---:|
| true-stock base@k8 | 237 / 300 | 0.7900 | 1 |
| B2-1000@k32 | 249 / 300 | 0.8300 | 0 |

**Paired AST correctness:** Δ(B2 − base) = **+0.0400**; paired Wald CI95
**[+0.0111, +0.0689]**; base-only/B2-only = **4/16**; exact two-sided McNemar
**p=0.01182**.  The preregistered 0.02 non-inferiority screen is PASS
(conditional Clopper–Pearson/Bonferroni lower95 `+0.0024`).  Truncation Δ =
`−0.0033`, CI95 `[-0.0099,+0.0032]`, McNemar `p=1.0` (unresolved).

This is positive evidence for the tool-call axis in this **n=300 pilot**, but
does not promote B2: the general-capability gates remain unresolved and this
benchmark protocol is not yet frozen.

### Scorer limitation discovered (do not hide)

The pinned upstream AST checker raises `KeyError('string')` for every selected
`simple_java` and `simple_javascript` item: those BFCL function descriptions
use generic JSON-schema `"string"`, while the same pinned checker indexes the
Java/JS-only type maps.  Both arms therefore score 0/26 Java and 0/14 JavaScript
under the **unmodified official checker**.  Python/parallel/multiple categories
account for the observed paired gain; this defect means the global number is a
bring-up result, not a cross-language BFCL claim.  Do not patch the upstream
checker or silently drop these categories after seeing outcomes.  A later
protocol decision must choose a pinned upstream fix/new checkout or predeclare
a Python-only deterministic subset and rerun both arms.

The n=10 template smoke immediately before this run passed the same end-to-end
path (base 9/10, B2 10/10, no truncations; paired Δ +0.10, McNemar p=1.0), and
was used only to validate syntax, not capability.

## Requested pilot contract (not yet executed)

| Field | Requested condition |
|---|---|
| Benchmark | BFCL v4, non-live deterministic AST categories only |
| Intended categories | `simple_python`, `simple_java`, `simple_javascript`, `parallel`, `multiple`, `parallel_multiple` (subject to the pinned upstream checkout's category manifest) |
| Explicit exclusions | `web_search` (SerpAPI/external API); every live or execution category that requires an external service |
| Subset | `n=200–400`, deterministic shuffle seed `0`; exact n/category allocation remains unfrozen |
| Arms | true-stock base@k8 versus B2-1000 expert patch@k32 |
| Patch identity | `~/models/esft/b2_1000/expert_patch.safetensors`, SHA-256 `c1b3f041051e9c184e5a3ea14126f921e3a2619b29454e3e73b96f79f45199d3`, 1,666 tensors (to be rechecked by campaign preflight) |
| Pairing | same ordered item keys, seed, chat-template/rendering path, generation cap, batch size, and local machine; serial base then B2 |
| Statistics | per-item binary AST correctness; B2 minus base paired difference, 95% CI, exact McNemar, and per-arm truncation counts |
| GPU scope | local physical GPU 0/1 only; GPU 2, gpu-host, and aux-host excluded |

The official BFCL README documents the `bfcl generate`/`bfcl evaluate`
workflow and explicitly states that `web_search` requires SerpAPI.  The intended
route is therefore: render each selected item through the Qwen3.5-MoE tool-call
chat template, retain the raw completion and a stable content-derived item key,
adapt the completion only through the upstream model-handler parser, then use
the upstream AST checker to record one Boolean per item.  Only after the
official parser accepts the Qwen output in a ten-item smoke may those Boolean
items be passed to the existing `eval_harness.py` paired verdict.

Sources: [BFCL README](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md), [upstream AST checker](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py).

## Bring-up checks actually performed

| Check | Observed result |
|---|---|
| `nvidia-smi` before launch | Failed: could not communicate with the NVIDIA driver. GPU 0/1 availability and GPU 2 exclusion cannot be verified. |
| `free -h` before launch | 125 GiB total, 109 GiB available; swap was 4.0/4.0 GiB used. This does not substitute for a working GPU check. |
| B2-1000 patch identity | Rechecked locally: SHA-256 `c1b3f041051e9c184e5a3ea14126f921e3a2619b29454e3e73b96f79f45199d3`, 1,666 tensors. |
| Local BFCL/Gorilla checkout | None found under `~`; no BFCL source or dataset is available locally. |
| `bfcl_eval` import | Not installed in the configured Python environment. |
| Repository retrieval | `git ls-remote https://github.com/ShishirPatil/gorilla.git HEAD` failed because `github.com` could not be resolved. |
| Qwen template ↔ BFCL parser smoke | Not run: the official repository/parser and usable GPU runtime are both absent. |
| Smoke / paired generation | Not run; no arm output, item-level correctness, CI, McNemar result, or truncation count exists. |

These rows describe the **earlier blocked attempt only**.  They were superseded
by the completed resumed-pilot checks above; no historical artifact was
overwritten.

## Historical result of the blocked attempt

At that earlier point, **no numerical result** existed: `n=0` (not an evaluated
subset), so no accuracy, paired difference, CI95, McNemar p-value, or
truncation count was reportable.  This historical statement is superseded by
the completed `n=300` result at the top of this document; its artifacts are
new run-directory files and did not overwrite the blocked-attempt record.

## Historical unblock sequence (completed)

1. Restore the local NVIDIA driver such that `nvidia-smi` enumerates physical
   GPUs 0, 1, and 2; verify GPUs 0/1 are idle and run `free -h` again.
2. Make a pinned Gorilla/BFCL checkout or the `bfcl-eval` package plus its
   dataset assets available locally.  Record the commit/package version and
   SHA-256 of the selected evaluation data; do not silently use a moving `main`.
3. Read that checkout's category manifest and parser/model-handler source,
   freeze an exact non-live category list and deterministic n/category allocation
   before generating either arm.  Keep all external-API categories excluded.
4. Add a thin adapter that calls the **pinned upstream** parser and AST checker
   (rather than a look-alike scorer), writes raw completion plus one `correct`
   Boolean and stable item key, and carries complete paired protocol metadata.
5. Run CPU parser/AST fixtures and a Qwen chat-template smoke (`n=10`) on GPU
   0/1.  Inspect raw outputs to prove that the model's tool-call syntax is what
   the upstream parser consumes.
6. After the existing campaign preflight passes, run base@k8 then B2@k32
   serially on physical 0/1 only and apply the established paired verdict.

The pilot protocol remains **unfrozen**.  Its measured result is harness
bring-up evidence only and must not be used for adoption without a separate
user-approved protocol decision.
