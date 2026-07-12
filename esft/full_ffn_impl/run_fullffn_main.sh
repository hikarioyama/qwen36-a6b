#!/bin/bash
# Full-FFN staged campaign. Default target is the first 200-step decision point.
# Extend with TARGET_STEPS=300 RESUME=.../checkpoint-200 after evaluation.
set -euo pipefail

ESFT_DIR=${ESFT_DIR:-/mnt/docker-raid/models/esft}
VENV=${VENV:-python3}
TRAINER=${TRAINER:-train_fullffn_dcp.py}
MODEL=${MODEL:-/mnt/docker-raid/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0}
TRAIN_DATA=${TRAIN_DATA:-v3.jsonl}
REPLAY_DATA=${REPLAY_DATA:-mixed_v2.jsonl}
OUT=${OUT:-codex_runs/fullffn_v1_20260710}
TARGET_STEPS=${TARGET_STEPS:-200}
RESUME=${RESUME:-}
LOG=${LOG:-${OUT}/train_to_${TARGET_STEPS}.log}

cd "$ESFT_DIR"
mkdir -p "$(dirname "$LOG")"

ARGS=(
  "$TRAINER"
  --model "$MODEL"
  --method full-ffn
  --expert-config configs/fullffn_probe.json
  --train-data "$TRAIN_DATA"
  --replay-data "$REPLAY_DATA"
  --replay-ratio 0.30
  --output-dir "$OUT"
  --router-top-k 32
  --seq-length 7168
  --fused-ce
  --random-concat-ratio 0
  --seed 5934875
  --optimizer adafactor
  --learning-rate 1e-5
  --weight-decay 0.0
  --grad-accum 4
  --per-device-batch-size 1
  --max-steps "$TARGET_STEPS"
  --eval-steps 100
  --save-steps 200
  --logging-steps 10
)

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume-from-checkpoint "$RESUME")
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VENV" -m torch.distributed.run --nproc_per_node=8 "${ARGS[@]}" \
  2>&1 | tee "$LOG"

echo "[$(date --iso-8601=seconds)] FULLFFN TARGET $TARGET_STEPS EXIT rc=0"
