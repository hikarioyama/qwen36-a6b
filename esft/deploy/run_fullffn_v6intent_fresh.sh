#!/bin/bash
# 注記 (2026-07-16): このスクリプトは並走セッションが独立に組み立てた v6 変種
# (sha 96806d7b...) 用の staging 記録で、実際には発射されていない。実走行
# (fullffn_v6divnames_fresh_20260715) は等価レシピの別ビルド (sha d6145158...) を
# 使用した — 両者は同一 322,861 行・_source タグ文字列のみ相違 (DEVLOG 2026-07-15 参照)。
# v6 fresh 区間 (2026-07-15, 燃料作り直し後の再出発):
#   stock base から fresh / router 完全凍結 / --router-tail-scale 0.5 / v6 燃料 / 1000 step。
# v6 = v5 − 旧 intent 全行 7,341 (mock_* テンプレ名、BFCL −31pt の汚染源)
#        + intent_r4_mocksurface 復元版 4,643 (diverse 名 + natural request、
#          生成は mock surface + 逆写像、選別 = exact-match 採択 92.9%)。
# 構成は run_fullffn_tail05_frozen_router.sh と同一 (燃料のみ変更 = 区間傾き比較が same-condition)。
set -euo pipefail
ESFT_DIR=${ESFT_DIR:-/mnt/docker-raid/models/esft}
VENV=${VENV:-python3}
MODEL=${MODEL:-$ESFT_DIR/base/Qwen3.6-35B-A3B}   # pinned snapshot 995ad96e...
FUEL=data/v6_20260715.jsonl
OUT=codex_runs/fullffn_v6intent_fresh_20260715
cd "$ESFT_DIR"
test -f "$MODEL/config.json" || { echo "LAUNCH_BLOCKED: base model が無い"; exit 1; }
test -f "$FUEL" || { echo "LAUNCH_BLOCKED: v6 が無い"; exit 1; }
# sha ゲート (組み立て時の値と一致必須)
sha=$(sha256sum "$FUEL" | cut -c1-16)
test "$sha" = "96806d7b74434e97" || { echo "LAUNCH_BLOCKED: v6 sha 不一致 ($sha)"; exit 1; }
# cache ゲート: trainer が実際に要求する sidecar 名を見る (旧名 .pt は false-green)
test -f data/cache/v6_20260715.jsonl.seq7168.seed5934875.ccr0.0.max0.tokincremental.index.json || { echo "LAUNCH_BLOCKED: v6 cache (sidecar) が無い — --prepare-data-only を先に"; exit 1; }
test -f data/cache/mixed_v2.jsonl.seq7168.seed5934875.ccr0.0.max0.tokincremental.index.json || { echo "LAUNCH_BLOCKED: replay cache が無い"; exit 1; }
$VENV - << 'EOF'
import json
s = json.load(open('data/v6_render_check.json'))
assert s['errors'] == 0, f"render errors: {s['errors']}"
print(f"render check OK: {s['ok']} rows")
EOF
mkdir -p "$OUT"
ENVS=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring NCCL_PROTO=Simple)
echo "=== PHASE v6intent_fresh start $(date -u +%s)"
env "${ENVS[@]}" "$VENV" -m torch.distributed.run --nproc_per_node=8 \
  train_fullffn_dcp.py --model "$MODEL" --method full-ffn \
  --expert-config configs/fullffn_probe.json \
  --train-data "$FUEL" --replay-data mixed_v2.jsonl --replay-ratio 0.30 \
  --output-dir "$OUT" --seq-length 7168 --router-top-k 32 --fused-ce \
  --router-tail-scale 0.5 \
  --random-concat-ratio 0 --optimizer adafactor --weight-decay 0.0 \
  --grad-accum 4 --per-device-batch-size 1 --max-steps 1000 \
  --eval-steps 100 --save-steps 300 --logging-steps 1 \
  --deterministic-fullffn \
  2>&1 | tee "$OUT/train_v6intent.log"
echo "V6INTENT_INTERVAL_DONE"