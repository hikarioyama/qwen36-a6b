# ESFT for Qwen3.6-35B-A3B — Phase 0

Expert-Specialized Fine-Tuning (ESFT, deepseek-ai/ESFT, MIT) ported to Qwen3.6-35B-A3B
(`Qwen3_5Moe`: 40 layers, 256 routed experts × top-8, packed 3D expert tensors).
Phase 0 = everything that does **not** need the GPU. See `NOTES.md` for the mechanism
and the key design decisions.

## Layout

```
esft_qwen/                 shared library (single source of truth, unit-tested)
  common.py                find_moe_blocks, exact routing reproduction
  scoring.py               ESFT gate/token scoring, top-p selection, Jaccard overlap
  esft_patch.py            to_esft_qwen (packed-expert grad masking), patch save/load
prepare_profiling_data.py  [CPU, DONE] build profiling + train data
collect_router_stats.py    [GPU] hook routers, accumulate expert stats -> npz
generate_expert_config.py  [CPU] stats -> ESFT config; --compare for domain overlap
to_esft_qwen.py            [CPU/GPU] apply config to a model + inspect / save patch
train_esft.py              [GPU] HF Trainer ESFT training, saves tiny expert patch
tests/test_smoke.py        [CPU] 22 assertions on the real Qwen3_5Moe MoE classes
data/profiling/{domain}.jsonl   32×4096-token blocks (built)
data/train/{domain}.jsonl       ESFT-format {"messages": [...]} (built)
vendor/ESFT/                deepseek-ai/ESFT reference clone
```

Interpreter: set `$VENV` to a Python environment with transformers 5.7.0. Always run with
`CUDA_VISIBLE_DEVICES=""` for the CPU steps.

## Status

| Step | State | Notes |
|------|-------|-------|
| env + arch investigation | done | see NOTES.md |
| prepare_profiling_data | done | 32×4096 profiling + full train jsonl for 3 domains |
| smoke test | done | 22/22 green (routing, freeze bit-invariance, patch roundtrip) |
| real-model structure check | done | meta-device: 40 MoE layers / 256 experts / top8 confirmed |
| collect_router_stats | code done, **GPU-gated** | needs real weights forward |
| generate_expert_config | done | validated on synthetic stats |
| to_esft_qwen | code done | functions unit-tested; CLI GPU-gated on real weights |
| train_esft | code done, **GPU-gated** | HF Trainer, wd=0 expert group |

## Reproduce the CPU work

```bash
: "${VENV:?Set VENV to the Python interpreter}"
cd ~/projects/qwen36-a6b/esft

# 1. Data (already built; re-run to regenerate)
CUDA_VISIBLE_DEVICES="" $VENV prepare_profiling_data.py                 # profiling + full train
CUDA_VISIBLE_DEVICES="" $VENV prepare_profiling_data.py --skip-train    # profiling only

# 2. Smoke test (quality gate)
CUDA_VISIBLE_DEVICES="" $VENV tests/test_smoke.py
```

## GPU-phase runbook (after VRAM frees up)

Use a local snapshot path for `--model` to avoid any download. `MODEL` below can be the
BF16 checkpoint once its weights are local, or any Qwen3.6-35B-A3B weights.

```bash
: "${VENV:?Set VENV to the Python interpreter}"
MODEL=Qwen/Qwen3.6-35B-A3B          # or a local snapshot dir with weights

# 3. Collect router stats per domain (top-8 default; --top-k N to override)
for d in math coding japanese; do
  $VENV collect_router_stats.py --model "$MODEL" \
    --profiling data/profiling/$d.jsonl --out stats/${d}_top8.npz --top-k 8
done

# 4. Generate ESFT configs (ESFT-Token p=0.2 first choice; ESFT-Gate p=0.1 alt)
for d in math coding japanese; do
  $VENV generate_expert_config.py --stats stats/${d}_top8.npz \
    --score token --top-p 0.2 --out configs/${d}_token_p0.2.json
done

# 4b. Domain overlap (how specialised are the selected experts?)
$VENV generate_expert_config.py --compare stats/math_top8.npz stats/coding_top8.npz \
  stats/japanese_top8.npz --score token --top-p 0.2 --out reports/overlap.json

# 5. (optional) inspect trainable footprint / save an untrained patch
$VENV to_esft_qwen.py --model "$MODEL" --config configs/math_token_p0.2.json --device-map auto

# 6. Train (per domain). Effective batch ≈ 32 = per_device × grad_accum × n_gpus.
$VENV train_esft.py --model "$MODEL" \
  --expert-config configs/math_token_p0.2.json \
  --train-data data/train/math.jsonl \
  --output-dir runs/math_esft --grad-accum 32
```

Hyperparameters (ESFT `configs/base.yaml`): LR 1e-5, seq 4096, ≤500 steps, eval/save
every 100, load-best-at-end. `train_esft.py` puts the packed expert tensors in a
weight_decay=0 optimiser group (see NOTES.md → weight-decay drift).

## Open items (GPU-only verification)

- Real routing distribution + selected-expert overlap on actual weights (synthetic
  numbers only so far for the config/compare path).
- `--top-k` override effect on the routing distribution.
- Training convergence and the tiny-patch load path on the real model.
- Prompt-masking fidelity in `train_esft.render_and_tokenize` (per-turn masking with the
  Qwen chat template) should be eyeballed on a few real samples before a long run.
