#!/bin/bash
# Repeat one fresh checkpoint-5 -> step-6 branch without eval/checkpoint-6 I/O.
set -euo pipefail

ESFT_DIR=${ESFT_DIR:-/mnt/docker-raid/models/esft}
VENV=${VENV:-python3}
TRAINER=${TRAINER:-train_fullffn_dcp.py}
MODEL=${MODEL:-/mnt/docker-raid/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0}
SOURCE=${SOURCE:-codex_runs/fullffn_probe_dcp_v3_20260710/checkpoint-5}
REFERENCE_LOG=${REFERENCE_LOG:-codex_runs/fullffn_probe_dcp_v3_20260710_resume_from_5/resume_5_to_6.log}
OUT=${OUT:-codex_runs/fullffn_resume_repro_v1_20260710}
LOG=${LOG:-${OUT}/resume_5_to_6.log}
COMPARE=${COMPARE:-1}
DETERMINISTIC=${DETERMINISTIC:-0}

cd "$ESFT_DIR"
mkdir -p "$OUT"

EXTRA_ARGS=()
ENV_ARGS=(FULLFFN_PROBE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
if [[ "$DETERMINISTIC" == "1" ]]; then
  EXTRA_ARGS+=(--deterministic-fullffn)
  ENV_ARGS+=(CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring NCCL_PROTO=Simple)
fi

env "${ENV_ARGS[@]}" "$VENV" -m torch.distributed.run --nproc_per_node=8 \
  "$TRAINER" --model "$MODEL" --method full-ffn \
  --expert-config configs/fullffn_probe.json \
  --train-data v3.jsonl --replay-data mixed_v2.jsonl --replay-ratio 0.30 \
  --output-dir "$OUT" --seq-length 7168 --router-top-k 32 --fused-ce \
  --random-concat-ratio 0 --optimizer adafactor --weight-decay 0.0 \
  --grad-accum 4 --per-device-batch-size 1 --max-steps 6 \
  --eval-steps 100 --save-steps 100 --logging-steps 1 \
  --resume-from-checkpoint "$SOURCE" --skip-final-checkpoint --skip-final-hf-export \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG"

if [[ "$COMPARE" == "0" ]]; then
  echo "RESUME REPRO RECORD COMPLETE"
  exit 0
fi

"$VENV" - "$REFERENCE_LOG" "$LOG" <<'PY'
import re, sys

reference, repeated = sys.argv[1:]
patterns = {
    "load_model": re.compile(r"\[rank(\d+)\] \[fullffn-model-load\].*model_match=(True|False)"),
    "load_optimizer": re.compile(
        r"\[rank(\d+)\] \[fullffn-optimizer-load\].*model_match=(True|False) "
        r"optimizer_tensors_match=(True|False) optimizer_scalars_match=(True|False)"
    ),
    "rng": re.compile(
        r"\[rank(\d+)\] \[fullffn-phase-a\] stage=RNG_BEFORE_FORWARD .*?"
        r"rng_sha256=([0-9a-f]{64}) model_training=(True|False)"
    ),
    "batch_loss": re.compile(
        r"\[rank(\d+)\] \[fullffn-step-input\] step=5 micro=(\d+) "
        r"batch_sha256=([0-9a-f]{64}) loss_hex=([^\s\[]+)"
    ),
    "gradient": re.compile(
        r"\[rank(\d+)\] \[fullffn-phase-a\] stage=GRAD6_AFTER_CLIP .*?"
        r"'digest': '([0-9a-f]{64})'.*?'tensor_count': (\d+), 'none_count': (\d+)"
    ),
    "post_optimizer": re.compile(
        r"\[rank(\d+)\] \[fullffn-phase-a\] stage=STEP6_POST_OPT .*?"
        r"'model': '([0-9a-f]{64})'.*?'optimizer_tensors': '([0-9a-f]{64})'.*?"
        r"'optimizer_scalars': '([0-9a-f]{64})'.*?'optimizer_state_entries': (\d+)"
    ),
}

def capture(path, pattern, batch=False):
    values = {}
    for match in pattern.findall(open(path).read()):
        rank, *rest = match
        if batch:
            key = (int(rank), int(rest[0]))
            values[key] = tuple(rest[1:])
        else:
            values[int(rank)] = tuple(rest)
    return values

for label, pattern in patterns.items():
    left = capture(reference, pattern, batch=(label == "batch_loss"))
    right = capture(repeated, pattern, batch=(label == "batch_loss"))
    expected = 32 if label == "batch_loss" else 8
    assert len(left) == len(right) == expected, (label, len(left), len(right))
    assert left == right, (label, left, right)
    print(f"ASSERT {label}: MATCH ({expected})")

print("RESUME REPRO RESULT: GREEN")
PY
