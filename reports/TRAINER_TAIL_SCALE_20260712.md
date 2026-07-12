# Trainer router-tail-scale — 2026-07-12

## Purpose and scope

`esft/deploy/train_fullffn_dcp.py` now accepts
`--router-tail-scale FLOAT`.  An explicit value installs a forward hook on every
routed-MoE gate.  It applies during both optimization forwards and the Trainer's
in-training evaluation forwards.  The hook transforms only the selected gate
`scores`; it returns the original `logits` and `indices` unchanged.

For every token independently, the hook:

1. ranks the selected scores in descending order;
2. multiplies rank 9 onward (zero-based positions `8:`) by `alpha`;
3. renormalizes with float32 arithmetic and casts back to the original score
   dtype.

`alpha=0` is therefore top-8 renormalization.  `alpha=1` is an identity and,
like the evaluator, does not register a hook.  `None` is the default: it does
not even discover MoE blocks or register a hook, and writes no new metadata, so
the legacy forward path and output layout stay byte-identical.

## Agreement with `eval_harness.py`

The score transformation is equivalent to the reference implementation in
`esft/eval_harness.py` (`load_subject_model`): `scores.float()`, descending
`argsort`, an all-ones mask with ranks `8:` set to `alpha`, multiplication,
`sum(...).clamp_min(1e-9)` renormalization, then cast back to the prior dtype.
As in evaluation, it is a post-gate operation; changing already-normalized
selected scores and renormalizing is equivalent to applying the same multiplier
before their selected-score normalization.

The intentional scope difference is lifecycle only: evaluation installs its
hook while loading a subject model, whereas the trainer installs it before
Trainer/FSDP wrapping so the same pure function is visited by train, eval, and
activation-checkpoint recomputation.  The trainer uses `order[..., 8:]` rather
than `order[:, 8:]`; real gate outputs are two-dimensional `[tokens, top_k]`, so
this is numerically identical there and also keeps the CPU stand-in batch-safe.

No gate parameter is read or modified by the hook.  This makes it compatible
with `FULL_SHARD`: it only consumes the gate's emitted tuple while that gate is
already executing.  There is no RNG or mutable state, so
`--deterministic-fullffn` does not require a special path and checkpointed
recomputation receives the same transformation.

## Reproducibility metadata

An explicit alpha is logged at startup as `[router-tail-scale]` with rank,
float32, renormalization, and train+in-training-eval scope.  It is also recorded
in `router_tail_scale.json` for delta checkpoints and final exports.  Full-FFN /
router-only DCP checkpoints carry the same structured metadata in
`checkpoint_complete.json`; resume rejects a changed or omitted alpha.  The
regular saved training arguments include the CLI value as well.

## CPU verification

`python3 esft/tests/test_fullffn_joint_trainer.py` passed **12/12** CPU tests
(5 new synthetic tail-scale cases; no GPU or model load).  The added cases cover:

- alpha `0` versus top-8 renormalization;
- an alpha `0.5` numeric example;
- two tokens with opposite rank order, proving ranks are token-local;
- default `None`, proving no forward hook is registered and the score tensor
  retains identical bytes.
- non-reentrant activation-checkpoint recomputation, proving the alpha `0.5`
  scores are byte-identical on the original and recomputed forwards while
  `torch.use_deterministic_algorithms(True)` is active (the PyTorch mode used by
  `--deterministic-fullffn`).

These are implementation tests, not a benchmark result (four requested
numerical/default cases plus one recomputation case; the rank-local case uses a
batch of `n=2` synthetic tokens).  No generation occurs,
so truncation is not applicable.

The motivating measured comparison remains separate: MMLU choice-logprob used
the same 600 items and paired protocol (`n=600`; no generation/truncation), with
base@k32 versus base@k8 `-3.17pt` (489/600 versus 508/600; exact paired
`p=0.013`).  This change did not launch an evaluation and reports no new model
metric.
