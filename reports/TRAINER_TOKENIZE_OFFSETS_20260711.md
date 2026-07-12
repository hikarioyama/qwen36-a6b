# Trainer tools + offset-mapping tokenization (2026-07-11)

## Decision

Implemented locally only in `esft/deploy/train_fullffn_dcp.py`; no gpu-host
deployment, network, SSH, GPU, or model-weight load was performed.  Fable is to
deploy only after its current interval boundary is complete.

`--tokenize-mode` is now an explicit opt-in:

- `incremental` is the default and retains the old per-prefix rendering,
  tokenisation, and label construction call shape.  It is the reproducibility
  path for existing and in-progress runs.
- `offsets` renders the complete conversation once with
  `apply_chat_template(messages, tools=record.get("tools"), tokenize=False)`,
  tokenises it once with the fast tokenizer's `return_offsets_mapping=True`, and
  labels only tokens wholly contained in an assistant turn.  A token crossing a
  turn boundary (or with a zero-width offset) is `-100`.

The assistant spans are identified by scanning the rendered
`<|im_start|>ROLE\n ... <|im_end|>\n` wrappers from left to right.  This maps
repeated assistant text/tool calls by occurrence, rather than by ambiguous
content substring search.  The span count is checked against the input's
assistant-turn count.

## Think-token policy

**Decision:** supervise the complete assistant channel, including the assistant
wrapper and any `<think>...</think>` tags and contents.  This is a deliberate
continuity choice: these bytes are emitted inside the assistant turn by the
model template, and the legacy incremental implementation also assigns its
assistant-channel additions to labels.  System, user, and tool turns are never
supervised in offsets mode.  Boundary-crossing tokens are masked instead of
being assigned to an adjacent role.

## Native tools records

`esft/corpus_to_trainer_v1.py` gained `--tools-mode {preamble,native}`.

- `preamble` remains the default and preserves the former output format: schema
  text is placed in the first user message for legacy incremental rendering.
- `native` places the normal source system message back in its system turn and
  emits the schema in the record's `tools` field.  It does not inject a user
  preamble.  This mode is intended for trainer `--tokenize-mode offsets`.

The tokenisation worker now receives the complete record, so it forwards native
`tools`; `pack_examples` does the same.  Cache payload names include both
`.tok{incremental|offsets}` and `.tools{0|1}`.  A source-stat-validated index
sidecar records the tools-presence bit after preparation, so distributed ranks
can find the correct payload without rereading the corpus.

## CPU measurements

All measurements below are CPU-only with the pinned local tokenizer snapshot
`995ad96eacd98c81ed38be0c5b274b04031597b0`, its chat template, and no model
weights.  These are tokenisation checks, not model-quality measurements.

**Measured:** one v3-style plain conversation (`n=1` conversation, 4 message
turns: user/assistant/user/assistant; same tokenizer/template condition) rendered
to 32 tokens.

| Path | input IDs equal one-shot final render | labeled tokens | labels differing from incremental |
|---|---:|---:|---:|
| incremental | no | 10 | — |
| offsets | yes | 16 | 15 token positions |

The difference is expected: the fast tokenizer combines newlines at assistant
boundaries, so incremental prefix token IDs are not a partition of the final
render.  The table is a single fixture (`n=1`), not a corpus-level rate.

**Measured tests:** `test_corpus_to_trainer_v1.py`, `n=11` tests PASS; it covers
one native-tools conversation with 4 message turns, a plain v3 fixture, worker
forwarding, packing, the cache-key separation, and an inline pre-change
incremental oracle.  `test_fullffn_joint_trainer.py`, `n=4` tests PASS.
`py_compile` and `git diff --check` PASS.  No paired verdict or generated-output
truncation applies to this CPU tokenisation work.

## Post-deployment verification for Fable

Do not mix this flag-gated route into the in-flight interval.  At the next
approved boundary:

1. Deploy the three local files together, then run the two CPU test modules in
   offline mode.  Convert a small native-tools fixture with
   `--tools-mode native` and prepare it with `--tokenize-mode offsets`; verify
   cache names contain `tokoffsets.tools1`, final input IDs match a one-shot
   render, and the rendered leading system block contains `# Tools` and the
   schema.
2. Before any multi-GPU run, execute the required Grok GPU preflight, arm a new
   never-reused job-event watcher with the exact process and completion
   artifacts, and confirm it is running.
3. Re-run the existing fresh/resume Full-FFN grad gate under otherwise identical
   conditions, changing only the approved dataset representation and
   `--tokenize-mode offsets`.  Require router/expert coverage, frozen-gradient
   assertions, router LR, and fresh-versus-resume digest equality to pass before
   a probe.  This is a correctness gate, not an evaluation result.
4. Run a short same-condition step probe (not a broad sweep): verify first-step
   loss/finite gradients, offsets cache reuse across ranks, checkpoint creation,
   and resume once.  Compare only like-for-like runs; do not make capability,
   paired, or truncation claims from this probe.

The post-deployment GPU checks above are proposed, not run here.
