# M-IFEval 日本語 seed 分散パイロット — 2026-07-11

## Status

**Strict scorer preflight now passes and the serial GPU campaign is running.**
This remains a protocol bring-up record, not a model-adoption or
protocol-freeze decision, until all paired outputs and analysis are complete.

The scorer is now isolated as required in
`external/mifeval-venv/bin/python` (Python 3.12). On the 2026-07-11 retry,
`absl-py`, `immutabledict`, `nltk`, and `spacy` all imported successfully, as
did the already-present Japanese dependencies. However, the unmodified upstream
`evaluation_main.py` imports its complete multilingual registry at startup, not
only Japanese rules. That registry requires two unavailable SpaCy model
distributions: **`es_core_news_sm`** and **`xx_sent_ud_sm`**. The strict scorer
therefore stops at `spacy.load("es_core_news_sm")` before any GPU allocation.
The harness deliberately refuses to replace, trim, or emulate the upstream
registry/rules, so neither GPU 0 nor GPU 1 was allocated and GPU 2, gpu-host, and
aux-host were untouched.

### 2026-07-11 strict-scorer retry — complete dependency audit

The audit exercised every import root used by the strict registry and checked
the resource loads reached by the English, Spanish, and French modules. It did
not stop after the first failure.

| Check | Result |
|---|---|
| Import roots | PASS: `absl-py` 2.5.0, `immutabledict` 4.3.1, `nltk` 3.10.0, `spacy` 3.8.14, `langdetect` 1.0.9, `janome` 0.5.0, `ja_sentence_segmenter` 0.0.2, `emoji` 2.15.0 |
| NLTK resources actually loaded by upstream | PASS: `nltk:tokenizers/punkt/english.pickle` and `nltk:tokenizers/punkt/french.pickle`; `punkt_tab` present |
| Missing strict-registry dependencies (complete list) | **BLOCKED: `es_core_news_sm`, `xx_sent_ud_sm`** |

These are SpaCy model distribution package names, not Python-library import
roots. No other missing package or resource was found in the audit. This is an
environment-only blocker; it does not alter the frozen dataset, scorer source,
model identities, sampling protocol, or analysis plan.

## Intended pilot protocol (not frozen)

| Field | Value |
|---|---|
| Dataset | local `external/M-IFEval/data/ja_input_data.jsonl`, all **n=172** upstream Japanese items (keys 1--172) in source order |
| Scoring | unmodified upstream `test_instruction_following_strict`; item binary pass = all instruction-level rule checks pass |
| Arms | verified true-stock base@k8; B2-1000 patch@k32 (`c1b3f041…f45199d3`, 1,666 tensors) |
| Samples | sampling seeds `{0,1,2,3,4}`; base seed stream then B2 seed stream, serial only |
| Generation | Qwen no-think chat prompt; `do_sample=true`, temperature 0.7, top-p 0.95, max-new 2048, batch 4 |
| Hardware | local physical GPU 0/1 only; `CUDA_VISIBLE_DEVICES=0,1`; GPU 2 excluded |
| Provenance | true-stock revision/fingerprint and patch identity are checked through the existing local campaign preflight before any generation |

## Indicators and planned paired analysis

- **Pass rate:** mean of the 172 × 5 item-level binary strict outcomes. The
  report also retains the five per-seed pass rates and their sample standard
  deviation; this is the requested pass-rate dispersion.
- **Seed agreement:** for each item, the fraction of its `C(5,2)=10` unordered
  seed pairs that have the same binary outcome, then averaged over 172 items.
  A value of 1 means all five seeds agree, including unanimous failure.
- **Paired deltas:** B2 minus base for both per-item five-seed mean pass and
  per-item agreement. CI95 is a 10,000-replicate percentile bootstrap that
  resamples items while keeping all five outcomes for an item together.
- **Truncation:** total capped generations per arm/seed is recorded separately;
  no generated benchmark result will be reported without it.

No acceptance margin, significance rule, or adoption criterion is set here.
The protocol therefore remains **unfrozen**, and this pilot cannot support a
model-selection decision even after the dependency blocker is removed.

## Results

| Arm | Generated items | Strict pass rate | Seed pass-rate SD | Seed agreement | Truncated |
|---|---:|---:|---:|---:|---:|
| true-stock base@k8 | 0 / 860 | not measured | not measured | not measured | not measured |
| B2-1000@k32 | 0 / 860 | not measured | not measured | not measured | not measured |

Paired pass-rate delta, paired agreement delta, and their CI95 are **not
available**: `n=0` generated paired items. No GPU job, run directory, or model
artifact was created or overwritten.

## Bring-up artifact and validation

`esft/mifeval_pilot.py` implements frozen-source loading, exclusive-create
artifacts, serial base/B2 × five-seed execution, upstream strict scoring,
truncation accounting, and the clustered paired bootstrap. Model inference stays
in the existing runtime, while preflight and each strict-score process execute
only through `external/mifeval-venv/bin/python`; no scorer is reimplemented.
The new CPU tests pass (3/3), and the existing evaluation-harness tests pass
(21/21). The strict-scorer preflight correctly fails before GPU allocation with
the missing-model error above.

### 2026-07-11 final resume attempt — scorer PASS, serial campaign running

The user installed `es_core_news_sm` and `xx_sent_ud_sm` into the dedicated
scorer environment and confirmed `spacy.load` for both. The actual harness
gate then passed with the unmodified upstream registry: `n=172`, registry size
118, scorer Python 3.12.13, and the source hashes recorded by
`mifeval_pilot.py`. The pilot's CPU structural tests passed (3/3) and the
existing evaluation-harness tests passed (21/21).

The local campaign preflight subsequently passed on GPU 0/1 (RTX PRO 6000
Blackwell, 97,887 MiB each; 5 MiB used and 0% utilization at preflight), after
which the new serial run `20260711_mifeval_ja_b2_1000_seed5_v1` began with
base@k8 seed 0. The transient standalone `nvidia-smi` failure before this
successful harness preflight is not treated as a campaign result. The durable
watcher is armed on the exact campaign parent process and manifest. While the
campaign runs, pass rates, seed agreement, pass-rate dispersion, truncation,
and paired deltas/CI95 remain unmeasured; do not infer them from partial arms.
Do not replace, trim, or partially reimplement the upstream rules.

## 実測結果 (v5, 2026-07-11 未明, protocol 未凍結)

v4 は launcher 終了時に子プロセスが落ちて停滞 → v5 として detached 再起動し完走 (manifest status=complete)。n=172 × 5 seeds × 2 arms、strict rule scorer、GPU 0/1 直列。

| 指標 | base@k8 | B2-1000@k32 | paired Δ (B2−base) / CI95 |
|---|---:|---:|---|
| strict pass rate (5-seed 平均) | 0.5628 | 0.4256 | **−0.1372 [−0.2291, −0.0465]** (item-clustered bootstrap 10k) |
| seed 間一致率 (agreement) | 0.8326 | 0.8291 | −0.0035 [−0.0547, +0.0477] (未確定) |
| pass rate の seed SD | 0.0096 | 0.0352 | — (B2 は seed 感度が約3.7倍) |
| truncated 合計 | 16 | 22 | — |

解釈:
- **B2-1000 は日本語 verifiable 指示追従で有意に悪化**。B2 の負の証跡としては今夜最大の効果量。
- seed 間一致率 (自己一貫性の proxy) は差なし。ただし B2 は seed ごとの pass rate ブレが大きく、生成安定性は低下方向。
- protocol 未凍結のパイロットであり採用判定には使わない (が、B2 系列除外の推奨をさらに補強する)。
