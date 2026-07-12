#!/bin/bash
# tail-scale 区間 (2026-07-12 設計転換後の第 1 区間):
#   base から fresh / router 完全凍結 / --router-tail-scale 0.5 / v4 燃料 / 1000 step。
# 根拠: α 掃引実測 (base@k32+α0.5 = 84.50%, base@k8 と統計的同等, n=600 paired)。
# 狙い: 借金ゼロの較正点から FFN を鍛え、境界の α 掃引で「84.67 超えの α」が現れるかを見る。
# joint 系 (--train-router / anchor) は不使用 — 較正はダイヤルが担う。
set -euo pipefail
# Host-specific paths; override via env. Defaults document the recorded run.
ESFT_DIR=${ESFT_DIR:-/mnt/docker-raid/models/esft}
VENV=${VENV:-python3}
MODEL=${MODEL:-$ESFT_DIR/base/Qwen3.6-35B-A3B}   # pinned snapshot 995ad96e...
OUT=codex_runs/fullffn_tail05_frozen_router_20260712
cd "$ESFT_DIR"
test -f "$MODEL/config.json" || { echo "LAUNCH_BLOCKED: base model が無い"; exit 1; }
test -f data/v4_20260711.jsonl || { echo "LAUNCH_BLOCKED: v4 が無い"; exit 1; }
# patched trainer (2026-07-12) は cache キーに .tok{mode} + sidecar .index.json を追加した。
# 旧名 .pt の存在チェックは false-green になる (実害 1 回) — sidecar を見る。
test -f data/cache/v4_20260711.jsonl.seq7168.seed5934875.ccr0.0.max0.tokincremental.index.json || { echo "LAUNCH_BLOCKED: v4 cache (新命名 sidecar) が無い — --prepare-data-only を先に"; exit 1; }
test -f data/cache/mixed_v2.jsonl.seq7168.seed5934875.ccr0.0.max0.tokincremental.index.json || { echo "LAUNCH_BLOCKED: replay cache (新命名 sidecar) が無い"; exit 1; }
$VENV - << 'EOF'
import json
s = json.load(open('data/v4_render_check.json'))
assert s['errors'] == 0, f"render errors: {s['errors']}"
print(f"render check OK: {s['ok']} rows")
EOF
mkdir -p "$OUT"
ENVS=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring NCCL_PROTO=Simple)
echo "=== PHASE tail05_frozen_router start $(date -u +%s)"
env "${ENVS[@]}" "$VENV" -m torch.distributed.run --nproc_per_node=8 \
  train_fullffn_dcp.py --model "$MODEL" --method full-ffn \
  --expert-config configs/fullffn_probe.json \
  --train-data data/v4_20260711.jsonl --replay-data mixed_v2.jsonl --replay-ratio 0.30 \
  --output-dir "$OUT" --seq-length 7168 --router-top-k 32 --fused-ce \
  --router-tail-scale 0.5 \
  --random-concat-ratio 0 --optimizer adafactor --weight-decay 0.0 \
  --grad-accum 4 --per-device-batch-size 1 --max-steps 1000 \
  --eval-steps 100 --save-steps 300 --logging-steps 1 \
  --deterministic-fullffn \
  2>&1 | tee "$OUT/train_tail05.log"
echo "TAIL05_INTERVAL_DONE"
