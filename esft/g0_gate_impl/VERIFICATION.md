# G0 gate-shaping — verification & decision rule

Training-free test: does reshaping the **router weights at inference time** claw
back the k32 knowledge regression (MMLU 0.8433 @ k8 → 0.8067 @ k32) without
touching a single model weight?

## What "shaping" does (mechanism, so it can't be a silent no-op)

Stock `Qwen3_5MoeTopKRouter.forward`:
`softmax(logits, fp32)` over all 256 → `topk(k)` (descending) → renorm → bf16 →
`experts()` multiplies each expert output by its weight. **Zeroing a weight ==
dropping that expert, with no tensor-shape change.** So every variant is a
*reweighting of the top-k weights over the SAME indices*:

- **temp** `softmax(logits/τ)` before topk. τ is monotonic → **indices identical**,
  only weights sharpen onto rank 1 (the calibrated head).
- **masscut** keep the shortest rank prefix whose cumulative mass first reaches
  `p`; zero the rest; renorm → **effective-k becomes dynamic per token**.
- **rankdamp** multiply ranks 9..32 (positions ≥ 8) by `α`, keep ranks 1..8,
  renorm.

All arithmetic is fp32 (matching the model's softmax dtype), cast to bf16 only at
the end — identical to stock. `τ=1` with no cut/damp reproduces stock exactly.

## Gate 1 — CPU math check (no GPU, already GREEN locally)

`gate_shaping.py` was exercised on realistic bf16 logits `(N=5, E=256, k=32)`:

| variant | eff_k | head(1-8) | tail(9-32) | rank1 |
|---|---|---|---|---|
| **k32 stock (OFF)** | 32.0 | 0.419 | 0.581 | 0.102 |
| temp 0.5 | 32.0 | 0.638 | 0.362 | 0.262 |
| temp 0.7 | 32.0 | 0.510 | 0.490 | 0.159 |
| masscut 0.80 | **22.8** | 0.514 | 0.486 | 0.126 |
| masscut 0.90 | **27.4** | 0.460 | 0.540 | 0.112 |
| rankdamp 0.25 | 32.0 | 0.741 | 0.259 | 0.181 |
| rankdamp 0.50 | 32.0 | 0.590 | 0.410 | 0.144 |

Asserts that passed: **OFF-equivalence** — `temp τ=1.0` recompute is bf16
**bit-identical** to the stock router (idx equal, fp32 max|Δ| = 0.0); every arm
renormalises to 1.0; head mass rises / tail falls for all six; masscut eff_k drops
below 32. Rerun anytime:

```
cd g0_gate_impl && python3 - <<'PY'
# (the smoke block from the impl session) — or just run g0_debug.py on GPU
PY
```

## Gate 2 — real-logits debug on GPU (`g0_debug.py`, ~2 min, 1 GPU)

Loads the 35B at k32, runs ONE real MMLU batch, captures the router's own
`(logits, scores, indices)` at a deep gate, then:

1. **OFF-equivalence proof on real logits**: `shape_router_scores(logits, k,
   'temp', 1.0)` must be `bf16_bit_identical=True` vs the model's own scores. If
   this prints a WARNING, the OFF arm is NOT bit-clean → **stop, do not trust the
   sweep**.
2. Prints the eff_k / head / tail / rank1 table above on *real* logits. Expected:
   head mass **rises** and tail **falls** for every variant vs the stock row;
   masscut eff_k drops below 32. **Any row equal to the stock row = that variant
   is a silent no-op** → investigate before sweeping.

```
ssh aux-host 'cd ~/esft && G0_GPU=0 ~/esft-work/venv/bin/python g0_debug.py'
```

## Gate 3 — in-sweep liveness

Every shaped arm runs with `G0_DEBUG=1 G0_DEBUG_CALLS=8`, so each GPU logs, for
its first 8 router calls:

```
[gpu0 G0-debug FINAL] variant=masscut param=0.80 ... eff_active_experts=NN.NN | head(1-8)_mass base=0.XX -> shaped=0.YY | rank1_wt base=... -> shaped=...
```

`base != shaped` there is the runtime proof the hook fired. `eff_active_experts`
is the per-config mechanism metric (dynamic-k for masscut).

## OFF-path guarantee (controls)

`run_g0_sweep.sh` runs the two stock controls with
`env -u G0_SHAPE -u G0_PARAM -u G0_DEBUG` so **no hook is registered** — the k32
and k8 control arms are byte-identical to the unpatched harness (the inserted code
early-returns before touching the model). This is also why `g0_k32_base` is the
correct paired baseline: same harness, same items, hook simply absent.

## Decision rule

Each shaped arm is compared PAIRED (exact McNemar on discordant items) against the
`g0_k32_base` control on the identical 600 MMLU items (auto-run at the end of the
sweep).

- **G0 SURVIVES** if some shaped arm has `delta = acc_shaped − acc_k32 ≥ +0.010`
  **and** `significant=True` (McNemar p<0.05). Training-free reshaping recovered
  ≥1pt of the k32 regression → worth pushing toward the full k8→k32 gap.
- **G0 DIES** if no arm clears +1.0pt significant. The k32 knowledge loss is not a
  pure gate-mass-allocation artefact; it needs the trained fix (ESFT / full-FFN),
  and inference-time reshaping is a dead lever — recorded, move on.

Context anchor: the full regression is 0.8433 (k8) − 0.8067 (k32) = **−3.66pt**.
`g0_k8_base` is in the sweep as the ceiling; a shaped arm approaching k8 accuracy
while keeping k32's 32-expert capacity would be the strong win.
