#!/bin/bash
# k32 長期戦・第一弾: 200 ステップ試走の checkpoint-200 から 1000 step まで継続 (+800 step)。
# 「変化量が足りないだけ」仮説の最安検証 — コーパス/構成は試走と同一、訓練量だけ伸ばす。
# checkpoint は 300 step ごと (300/600/900) + 最終 1000。保存ごとにローカル vault へ回収する。
set -euo pipefail
ESFT_DIR=/mnt/docker-raid/models/esft
VENV=~/esft-venv/bin/python
MODEL=/mnt/docker-raid/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
PREV=codex_runs/fullffn_joint_200step_20260711
OUT=codex_runs/fullffn_joint_1000step_20260711
cd "$ESFT_DIR"
test -f "$PREV/checkpoint-200/checkpoint_complete.json" || { echo "RESUME_BLOCKED: no checkpoint-200"; exit 1; }
mkdir -p "$OUT"
JOINT_ARGS=(--train-router --router-lr-mult 0.08 --router-anchor-weight 0.15 --allow-router-joint-fullffn --deterministic-fullffn)
ENVS=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring NCCL_PROTO=Simple)
echo "=== PHASE resume200to1000 start $(date -u +%s)"
env "${ENVS[@]}" "$VENV" -m torch.distributed.run --nproc_per_node=8 \
  train_fullffn_dcp.py --model "$MODEL" --method full-ffn \
  --expert-config configs/fullffn_probe.json \
  --train-data v3.jsonl --replay-data mixed_v2.jsonl --replay-ratio 0.30 \
  --output-dir "$OUT" --seq-length 7168 --router-top-k 32 --fused-ce \
  --random-concat-ratio 0 --optimizer adafactor --weight-decay 0.0 \
  --grad-accum 4 --per-device-batch-size 1 --max-steps 1000 \
  --eval-steps 100 --save-steps 300 --logging-steps 1 \
  --resume-from-checkpoint "$PREV/checkpoint-200" \
  "${JOINT_ARGS[@]}" \
  2>&1 | tee "$OUT/train1000.log"
echo "JOINT_1000STEP_DONE"
