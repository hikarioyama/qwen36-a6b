# Corpus intake and decontamination v1 — 2026-07-11

## Scope and measurement status

This report is an intake probe, not a full corpus pass.  The source corpora were
read only and were not modified.  The measured structural sample is the first
32 records of every data file: **n=4,544 records / 142 files**.  It used
`esft/corpus_decontam_v1.py scan --batch-size 32 --limit-per-file 32`.

The Toucan row counts below are **measured exactly from Parquet metadata**
(n=1,646,546 total), without materializing all rows.  ToolMind JSONL exact row
counts and full-corpus mean lengths are **not measured**: obtaining either
requires a complete streaming pass, which this task explicitly defers to
Fable.  `368,611` is only the user-provided/public nominal count, not an
independent measurement.

“Token” here is explicitly a character/4 approximation over canonical JSON
serialization, not a tokenizer measurement.

## Toucan-1.5m (local README: Apache-2.0)

| Teacher subset | Files | Actual rows (Parquet metadata) | Length measurement | Conversation / tool representation (measured sample) |
|---|---:|---:|---:|---|
| Kimi-K2 | 40 | 518,516 | n=1,280; 40,938.5 chars, 10,234.6 approx tokens | `messages` is a JSON string encoding a role-message array; observed `system/user/assistant/function`, legacy `function_call`, and `name`/`arguments` objects. |
| OSS | 47 | 457,130 | n=1,504; 52,491.6 chars, 13,122.9 approx tokens | Same 9-column schema and legacy function-call representation as Kimi-K2. |
| Qwen3 | 44 | 551,613 | n=1,408; 42,055.5 chars, 10,513.9 approx tokens | Same 9-column schema and legacy function-call representation as Kimi-K2. |
| SFT | 3 | 119,287 | n=96; 11,532.8 chars, 2,883.2 approx tokens | `messages` and `tools` are JSON strings; observed roles `user/assistant/tool_call/tool_response`, plus nested function objects. |

The first three subsets have one identical Arrow schema across all their files:
`uuid, subset_name, messages, question, available_tools, target_tools,
question_quality_assessment, response_quality_assessment, metadata` (all
string columns).  SFT has one identical schema across its three files:
`uuid, subset_name, question, target_tools, tools, messages` (all strings).
Thus the apparent objects/arrays are JSON encoded inside string columns; an
ingestor must parse those strings rather than assume native Parquet lists.

## ToolMind (wrapper README: Apache-2.0; upstream component terms not verified offline)

All eight JSONL files have the measured top-level shape
`{"conversations": [...], "tools": [...]}`.  `conversations` is a native
message array; observed tool calls use `assistant.tool_calls[]` with
`function: {name, arguments}`, followed where applicable by `role: tool`.
The following values are **sample estimates only (n=32/file)**.

| File | Avg chars / approx tokens | Observed `tool_calls` occurrences | Origin / license manifest |
|---|---:|---:|---|
| `graph_syn_datasets/graphsyn.jsonl` | 15,264.2 / 3,816.1 | 76 | ToolMind synthetic graph data; `UNVERIFIED_upstream` (hypothesized only). |
| `open_datasets/APIGen-MT-5k-query.jsonl` | 23,563.0 / 5,890.7 | 115 | **B_suspected_noncommercial_or_review_required**; user-directed suspicion, not independently verified. Exclude from commercial selection pending upstream review. |
| `open_datasets/BUTTONInstruct-query.jsonl` | 8,709.3 / 2,177.3 | 83 | ToolMind README identifies BUTTONInstruct; `UNVERIFIED_upstream` (hypothesized only). |
| `open_datasets/ToolACE-query.jsonl` | 5,066.1 / 1,266.5 | 32 | ToolMind README identifies ToolACE; `UNVERIFIED_upstream` (hypothesized only). |
| `open_datasets/When2Call-query.jsonl` | 3,758.1 / 939.5 | 18 | ToolMind README identifies When2Call; `UNVERIFIED_upstream` (hypothesized only). |
| `open_datasets/glaive-function-calling-v2-query.jsonl` | 1,751.1 / 437.8 | 44 | ToolMind README identifies glaive-function-calling-v2; `UNVERIFIED_upstream` (hypothesized only). |
| `open_datasets/tau-train-query.jsonl` | 17,587.3 / 4,396.8 | 176 | **B_suspected_noncommercial_or_review_required**; user-directed suspicion, not independently verified. Exclude from commercial selection pending upstream review. |
| `open_datasets/xlam-function-calling-60k-query.jsonl` | 4,130.2 / 1,032.6 | 23 | ToolMind README identifies xlam-function-calling-60k; `UNVERIFIED_upstream` (hypothesized only). |

The two B labels are a compliance gate, not a statement of their definitive
licenses.  No network/DNS lookup was attempted, so every other upstream label
above is deliberately `hypothesized`/unverified rather than inferred.

## `corpus_decontam_v1.py` contract

`scan` emits JSON (per file: schema, rows, sample coverage, length, message
signals, and license label) and Markdown.  With `--limit-per-file 0`, it streams
all records and reports actual JSONL counts and full means.

`decontam` streams either JSONL or Parquet.  It writes the clean corpus under a
new output root, preserves input format, and emits `manifest.json` plus
`removals.jsonl`.  A removal log records only source path, row index, SHA-256
of canonical row content, and matching evaluation-set names; it does not copy
the removed content.

The v1 predicate is an exact normalized **8-gram** overlap across all text
leaves (including JSON encoded strings): NFKC + casefold, word tokens for Latin
text and character tokens for CJK text.  The in-memory index holds BLAKE2b-128
digests of eval grams; corpus rows remain batch-streamed (default 512 rows), so
peak corpus memory is far below 16 GB.  This is a high-recall lexical screen,
not a semantic decontamination proof.

Evaluation origins are determined only from the existing local evaluation
scripts and forced offline:

| Eval set | Local origin used | On missing local data |
|---|---|---|
| MMLU / GSM8K / HumanEval | `esft/eval_harness.py` `Mmlu/Gsm8k/HumanEval.load`, local `datasets` cache | `SKIPPED` in manifest |
| JMMLU | `esft/eval_harness.py` `Jmmlu.load`, local `JMMLU.zip` cache | `SKIPPED` in manifest |
| BFCL | `esft/bfcl_pilot.py` local Gorilla `bfcl_eval/data/BFCL_v4_*.json` | `SKIPPED` in manifest |
| M-IFEval | `esft/mifeval_pilot.py` `external/M-IFEval/data/ja_input_data.jsonl` | `SKIPPED` in manifest |

Local-loader preflight on this host found all six sources available (this is a
loader measurement, not a corpus run): MMLU n=14,042, GSM8K n=1,319,
HumanEval n=164, JMMLU n=7,536, BFCL n=4,696, and M-IFEval n=172.  The Arrow
cache is read directly because `datasets.load_dataset` attempts lock/cache
writes that are incompatible with a read-only cache; this changes no benchmark
content or source selection.  A later host still records `SKIPPED` if its local
asset is missing.

The command refuses to create a “clean” output when **all** evaluation sources
are skipped, unless the explicitly unsafe `--allow-no-eval` is supplied.

## CPU smoke test (measured)

`python3 esft/tests/test_corpus_decontam_v1.py` passed on 2026-07-11:

- n=2 JSONL + n=2 Parquet synthetic records;
- `scan` completed and recognized message arrays/tool calls;
- `decontam` with an offline MMLU fixture removed the two exact 8-gram matches
  and preserved one JSONL plus one Parquet record;
- CJK normalization has a dedicated eight-character-gram unit test.

No GPU command or `nvidia-smi` was run.

## Fable full-run procedure (not launched here)

Choose a new, empty run root; never target either source directory.  The
commands below are CPU/IO only and must be started by Fable in its detached job
wrapper.  Keep the resulting JSON, Markdown, manifest, and removal log as the
run record.

```bash
cd ~/projects/qwen36-a6b
RUN_ROOT=/mnt/vault/corpora/derived/qwen36-a6b-intake-20260711-v1
mkdir -p "$RUN_ROOT"

python3 esft/corpus_decontam_v1.py scan \
  --input /mnt/vault/corpora/toucan-1.5m \
  --input /mnt/vault/corpora/toolmind \
  --batch-size 512 --limit-per-file 0 \
  --output-json "$RUN_ROOT/scan.json" \
  --output-md "$RUN_ROOT/scan.md"

python3 esft/corpus_decontam_v1.py decontam \
  --input /mnt/vault/corpora/toucan-1.5m \
  --input /mnt/vault/corpora/toolmind \
  --batch-size 512 \
  --output-dir "$RUN_ROOT/clean"
```

Before accepting the result, inspect `clean/manifest.json`: every intended eval
set should be `AVAILABLE`; any `SKIPPED` set is a documented coverage gap, not
evidence of cleanliness.  The commands do not initiate network access and do
not touch GPUs.
