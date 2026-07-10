#!/bin/bash
# joint200 kill 判定 eval: base@k8 (fresh reference) / joint200@k32 / joint200@k8 の
# MMLU n=600 を同一 run 内 3 腕直列で流し、paired verdict を出す。
# 判定 (kill 表の読み替え): joint@k8 vs base@k8 劣化 ≤0.8pt かつ joint@k32 が改善方向 → 本走 GO。
set -euo pipefail
ESFT=/mnt/docker-raid/models/esft
VENV=~/esft-venv/bin/python
EXPORT=$ESFT/codex_runs/fullffn_joint_200step_20260711
TS=20260711
cd "$ESFT"
# export の完全性チェック (HF save_pretrained の成果物)
test -f "$EXPORT/config.json" || { echo "EVAL_BLOCKED: no config.json in $EXPORT"; exit 1; }
ls "$EXPORT"/*.safetensors >/dev/null 2>&1 || { echo "EVAL_BLOCKED: no safetensors in $EXPORT"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
echo "=== ARM base_k8 start $(date -u +%s)"
$VENV eval_harness.py --model base --benchmark mmlu --topk 8 --n 600 --tag joint200_${TS}_base_k8_mmlu
echo "=== ARM joint_k32 start $(date -u +%s)"
$VENV eval_harness.py --model base --model-path "$EXPORT" --benchmark mmlu --topk 32 --n 600 --tag joint200_${TS}_joint_k32_mmlu
echo "=== ARM joint_k8 start $(date -u +%s)"
$VENV eval_harness.py --model base --model-path "$EXPORT" --benchmark mmlu --topk 8 --n 600 --tag joint200_${TS}_joint_k8_mmlu
echo "=== PAIRED VERDICTS"
R="$HOME/esft/reports/eval"
$VENV eval_harness.py --paired-verdict "$R/joint200_${TS}_base_k8_mmlu_items.json" "$R/joint200_${TS}_joint_k32_mmlu_items.json" || true
$VENV eval_harness.py --paired-verdict "$R/joint200_${TS}_base_k8_mmlu_items.json" "$R/joint200_${TS}_joint_k8_mmlu_items.json" || true
echo "JOINT200_EVAL_DONE"
