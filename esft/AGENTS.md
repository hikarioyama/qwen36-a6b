# ESFT Experiment Rules

## Canonical Local Assets

- Evaluation implementation: `eval_harness.py` in this directory.
- Codex campaign runner: `codex_harness.py` with `codex_harness.toml`.
- True stock snapshot revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Forbidden base directory: `/mnt/data/models/Qwen3.6-35B-A3B-agentic-k32`.
  It contains a merged agentic patch despite its stock-looking metadata.
- G1 B2 patch: `/mnt/data/probe_ckpt_eval/g1B2_patch.safetensors`.

Run `python3 codex_harness.py preflight` before any real evaluation. Do not bypass
an identity or protocol failure by changing the expected hash without explaining
and independently verifying the new artifact.

## Evaluation Contracts

- Use the same machine, harness revision, Python environment, dataset ordering,
  seed, prompt mode, token cap, and batch size for paired arms.
- MMLU/JMMLU must use `--choice-logprob`; generative MMLU numbers are invalid.
- Fresh MMLU gates use `--shuffle --seed 0` so all subjects are represented. The
  historical first-600 G1 values remain a reproduction anchor, not a representative
  general-knowledge estimate.
- MMLU and GSM8K use `n=600`. HumanEval uses all 164 tasks.
- HumanEval uses `max_new=4096`. Always report truncation and compare the paired
  `truncated` field as well as correctness.
- HumanEval/MBPP candidate execution requires a passing bubblewrap self-test. A
  missing or failed OS sandbox is an infrastructure failure, never a wrong answer.
- Bubblewrap is the host-security boundary. Candidate code and benchmark asserts
  still share one isolated Python interpreter, so correctness scoring assumes
  ordinary non-adversarial benchmark completions; it is not a proof against a
  completion deliberately forged to impersonate the test runner.
- Use exact paired McNemar through `eval_harness.py --paired-verdict`; do not infer
  significance from overlap of independent confidence intervals.
- McNemar `p >= 0.05` means the difference is unresolved, not that non-inferiority
  is proven. State a regression margin before a go/no-go claim and check the paired
  confidence bound against it; `n=600` may only support a large-regression screen.
- New results must have stable item keys and matching protocol metadata. Legacy
  reports without those fields are references, not substitutes for a fresh gate.
- Never overwrite a result tag silently. Use a new run directory or pass an
  explicit overwrite flag only when the discarded artifact is understood.

## GPU Ownership

- GPU 0 and 1 form the local evaluation pair. One root-agent-run campaign owns
  both for its full base/B2 sequence.
- Do not run base and B2 simultaneously. Serial execution avoids memory and
  thermal cross-arm differences and preserves a simple failure boundary.
- A subagent may inspect logs or results but must not mutate the harness or launch
  another process while a campaign is active.

## Result Discipline

Every accepted run records:

- exact command and timestamps;
- git HEAD and dirty paths;
- Python and relevant package versions;
- evaluation harness SHA-256;
- stock revision plus tensor fingerprint;
- patch SHA-256 and tensor count;
- GPU inventory and protocol parameters;
- correctness, truncation, and paired verdicts.

HumanEval at `max_new=4096` is a fixed-budget evaluation, not an untruncated
capability estimate. Report correctness and non-EOS cap rate as separate outcomes.

If a result violates established physics or reverses a strong anchor, audit model
identity, harness hash, evaluation path, and generation truncation before proposing
a new mechanism.
