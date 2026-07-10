# 35B-A6B Campaign: Codex Operating Contract

## Scope

- This repository is an ML systems and evaluation project for expanding
  `Qwen3.6-35B-A3B` from native top-k 8 to top-k 32.
- Canonical machine/repository identifier is `qwen36-a6b`; the expanded target is
  `Qwen3.6-35B-A6B (k=32)`. Never apply identifiers from the abandoned 285B
  candidate to this project or its artifacts.
- Preserve correct upstream and literature names such as `deepseek-ai/ESFT`,
  DeepSeek-V2-Lite, and DeepSeekMath. Keep machine-specific runtime manifests and
  absolute paths out of version control.
- This file overrides unrelated workstation-level project descriptions. The work
  here is not electrical harness engineering and not Moonlight client work.
- Within this repository, ignore inherited Moonlight role/task details, Claude
  lead status prefixes, and the global GPT-5.5 runtime assumption. Keep inherited
  language, reply-format, safety, and user-preference instructions.
- The objective is a statistically honest A6B configuration that keeps the k32
  capability gains without knowledge, math, or coding regressions.

## Source Of Truth

Read only as much as the task needs, in this order:

1. The user's current request.
2. `reports/status_20260710.html` for the latest handoff and current branch point.
3. `DEVLOG.md` for measured history and failed approaches.
4. `esft/PLAN.md` and `esft/GOALS.md` for older design intent; newer measured
   evidence wins when these documents disagree.
5. Per-item JSON under `esft/reports/eval/` for numerical claims.

Do not import conclusions from Claude Code session prose when the repository,
artifacts, or a fresh measurement can establish them directly.

## Codex Work Model

- Project runtime is defined by `.codex/config.toml`: `gpt-5.6-sol` at `ultra`
  effort. This selects Sol's automatic task delegation for the campaign's mixed
  code, statistics, and systems work.
- The root agent owns the plan, user communication, repository mutations, and all
  GPU launch/stop decisions.
- Use parallel subagents for bounded independent work such as artifact audit,
  statistical review, implementation review, or test design.
- Subagents must not launch GPU jobs unless the root agent explicitly delegates a
  named GPU set. This prevents two agents from evaluating on the same devices.
- Give editing agents disjoint file ownership. The root agent reconciles and tests
  shared interfaces before launch.
- Treat long GPU runs as experiments, not background shell commands: preflight,
  run in a unified exec session, monitor output, and preserve the manifest.
- Before training, a multi-GPU campaign, or any job expected to occupy GPUs for
  15 minutes or more, invoke the global `$grok-gpu-preflight` skill. Freeze the
  job, complete Grok Fast + high web research, verify decision-changing claims,
  and disposition every actionable finding before launch. Short isolated smoke
  tests that cannot expand into a full run are exempt.
- Prefer one decisive experiment over a broad speculative sweep. State measured,
  inferred, and proposed claims separately.

## Handoff Snapshot (2026-07-10)

- G1 B2 (`v3 + KL beta=0.5`, 500 steps) reached MMLU@k32 `0.8417`, statistically
  unresolved versus true-stock base@k8 `0.8467` (paired `p` approximately `0.74`).
- The next required gate is a fresh same-condition base@k8 versus B2@k32 paired
  evaluation on GSM8K and HumanEval.
- Do not start full-FFN training. It remains frozen unless the B2 regression gate
  fails and the user approves that branch.
- Static rank damping is not a product default: it recovers MMLU but significantly
  harms GSM8K.

This is a dated handoff snapshot, not live state. Re-read the status report and
current artifacts before acting, and update the project record after a branch gate.

## Repository Safety

- The worktree contains important uncommitted and untracked experiment assets.
  Never reset, clean, restore, or rewrite unrelated files.
- Do not call a model "stock" from its directory name or README. Verify the
  configured revision and tensor fingerprint first.
- Remote nodes are external state. Inspect current processes and artifacts before
  assuming the status recorded in a report is still current.
- Cross-machine agent coordination is user-as-bridge unless the user explicitly
  authorizes a direct remote action.

## Entry Points

- Before changing or running anything under `esft/`, read `esft/AGENTS.md`; nested
  instructions are not automatically loaded when a Codex session starts at repo root.
- Campaign preflight: `python3 esft/codex_harness.py preflight`
- Dry-run a paired gate: `python3 esft/codex_harness.py campaign gsm8k --dry-run`
- A real campaign also requires `--noninferiority-margin <fraction>`; choose and
  record it before seeing the new arm results.
- CPU harness tests: `python3 esft/tests/test_eval_harness.py`
- Existing ESFT smoke test: `CUDA_VISIBLE_DEVICES="" python3 esft/tests/test_smoke.py`

## Long Job Events

- The `jobEvents` MCP server is the durable completion channel for long local and
  remote jobs. At session entry, call `list_job_events` before assuming
  that the dated handoff is still current.
- While a turn is intentionally monitoring running work, use
  `wait_for_job_event`; do not implement model-side polling loops.
- Inspect artifacts and record the resulting decision before calling
  `ack_job_event`. An ACK means the event was handled, not merely observed.
- MCP resource notifications do not start an idle model turn. Unhandled events
  remain queued and must be collected at the next turn.
- Watcher definitions live in the untracked `tools/job-events-mcp/jobs.json`; the
  user service is `qwen36-a6b-job-events-watcher.service`. Before a new long GPU
  launch, add its exact process identity and positive completion artifacts, then verify
  `job_watch_status` reports `running`. Use a never-reused job ID, add artifact
  start-time/structure checks for remote work, and restart the user service after
  changing `jobs.json`.

Before reporting a numerical result, include the sample count, protocol identity,
paired verdict where applicable, and truncation count for generated benchmarks.
