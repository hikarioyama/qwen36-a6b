#!/bin/bash
# 区間 3: v4 コーパス (転写除去済み v3 + Toucan 厳選 + general 増強 + selfgen 層別 = 322,262 行)。
# 区間 2 の最終 HF export から fresh 起動 (optimizer 新規)。構成は区間 2 と同一 —
# 変数はコーパスのみ。これで区間傾き比較 (v3 区間 vs v4 区間) が same-condition になる。
# G1: 現行形式で凍結 (offsets 改修は smoke+12h マージン未達で今区間見送り)。
set -euo pipefail
# Host-specific paths; override via env. Defaults document the recorded run.
ESFT_DIR=${ESFT_DIR:-/mnt/docker-raid/models/esft}
VENV=${VENV:-python3}
MODEL=$ESFT_DIR/codex_runs/fullffn_joint_1000step_20260711   # 区間 2 の最終 HF export
OUT=codex_runs/fullffn_joint_v4corpus_20260712
cd "$ESFT_DIR"
test -f "$MODEL/config.json" || { echo "LAUNCH_BLOCKED: 区間 2 export が無い"; exit 1; }
ls "$MODEL"/*.safetensors >/dev/null 2>&1 || { echo "LAUNCH_BLOCKED: safetensors が無い"; exit 1; }
test -f data/v4_20260711.jsonl || { echo "LAUNCH_BLOCKED: v4 コーパスが無い"; exit 1; }
# G2-③ ゲート: レンダリング検証がエラーゼロで完了していること
$VENV - << 'EOF'
import json
s = json.load(open('data/v4_render_check.json'))
assert s['errors'] == 0, f"render errors: {s['errors']} — 発射中止"
print(f"render check OK: {s['ok']} rows, p90={s['tokens_p90']} tokens, over7168={s['over_seq_7168']}")
EOF
mkdir -p "$OUT"
# 新コーパスは tokenization cache が無いと multi-process 起動を trainer が拒否する
# (2026-07-12 に区間 3 初回発射で踏んだ)。無ければ単独プロセスで prepare を先に回す。
CACHE=data/cache/v4_20260711.jsonl.seq7168.seed5934875.ccr0.0.max0.pt
if [ ! -f "$CACHE" ]; then
  echo "=== PREPARE cache missing -> --prepare-data-only"
  "$VENV" train_fullffn_dcp.py --model "$MODEL" --method full-ffn \
    --expert-config configs/fullffn_probe.json \
    --train-data data/v4_20260711.jsonl --replay-data mixed_v2.jsonl --replay-ratio 0.30 \
    --output-dir "$OUT" --seq-length 7168 --router-top-k 32 \
    --random-concat-ratio 0 --grad-accum 4 --per-device-batch-size 1 --max-steps 1000 \
    --prepare-data-only 2>&1 | tee "$OUT/prepare_data.log"
fi
JOINT_ARGS=(--train-router --router-lr-mult 0.08 --router-anchor-weight 0.15 --allow-router-joint-fullffn --deterministic-fullffn)
ENVS=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring NCCL_PROTO=Simple)
echo "=== PHASE v4corpus_interval3 start $(date -u +%s)"
env "${ENVS[@]}" "$VENV" -m torch.distributed.run --nproc_per_node=8 \
  train_fullffn_dcp.py --model "$MODEL" --method full-ffn \
  --expert-config configs/fullffn_probe.json \
  --train-data data/v4_20260711.jsonl --replay-data mixed_v2.jsonl --replay-ratio 0.30 \
  --output-dir "$OUT" --seq-length 7168 --router-top-k 32 --fused-ce \
  --random-concat-ratio 0 --optimizer adafactor --weight-decay 0.0 \
  --grad-accum 4 --per-device-batch-size 1 --max-steps 1000 \
  --eval-steps 100 --save-steps 300 --logging-steps 1 \
  "${JOINT_ARGS[@]}" \
  2>&1 | tee "$OUT/train_v4.log"
echo "JOINT_V4_INTERVAL3_DONE"
