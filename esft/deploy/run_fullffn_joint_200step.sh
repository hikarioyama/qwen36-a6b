#!/bin/bash
# Full-FFN + router joint 200-step probe (grad-gate v3 PASS 後の本 probe)。
# 構成は run_fullffn_joint_gradgate.sh の fresh 段と同一条件、max-steps 200。
# FULLFFN_PROBE は外す (per-step digest/coverage 検査のオーバーヘッド回避、ゲートは通過済み)。
# 最終成果物: resumable DCP checkpoint-200 + HF full model export (eval 用、~70GB)。
set -euo pipefail
ESFT_DIR=/mnt/docker-raid/models/esft
VENV=${VENV:-python3}
MODEL=/mnt/docker-raid/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
OUT=codex_runs/fullffn_joint_200step_20260711
cd "$ESFT_DIR"
mkdir -p "$OUT"
JOINT_ARGS=(--train-router --router-lr-mult 0.08 --router-anchor-weight 0.15 --allow-router-joint-fullffn --deterministic-fullffn)
ENVS=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring NCCL_PROTO=Simple)
echo "=== PHASE 200step start $(date -u +%s)"
env "${ENVS[@]}" "$VENV" -m torch.distributed.run --nproc_per_node=8 \
  train_fullffn_dcp.py --model "$MODEL" --method full-ffn \
  --expert-config configs/fullffn_probe.json \
  --train-data v3.jsonl --replay-data mixed_v2.jsonl --replay-ratio 0.30 \
  --output-dir "$OUT" --seq-length 7168 --router-top-k 32 --fused-ce \
  --random-concat-ratio 0 --optimizer adafactor --weight-decay 0.0 \
  --grad-accum 4 --per-device-batch-size 1 --max-steps 200 \
  --eval-steps 50 --save-steps 100 --logging-steps 1 \
  "${JOINT_ARGS[@]}" \
  2>&1 | tee "$OUT/train200.log"
echo "JOINT_200STEP_DONE"
