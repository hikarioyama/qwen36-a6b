#!/bin/bash
# co-activation ゲート実験 (gpu-host, 8-GPU): mixed_v2 と同一 config で、訓練中の
# gate.top_k を毎 forward {8,16,24,32} から一様サンプル (--router-topk-random)。
# 目的: k=32 拡張後の知識 -2pt 頭打ちが「expert が高-k 協調の訓練を受けてない
# (co-occurrence 不足)」問題かを、選抜 delta インフラで安く偵察する。
# eval/save は k=32 固定 (hook が module.training=False で max(set) に戻す)。
#
# 実行前提: 実技試験ペアが 8 枚占有中なら回すな。gpu-host 空きを確認してから。
set -euo pipefail
cd "$(dirname "$0")"

VENV=~/esft-work/venv/bin/python   # gpu-host の venv パスに合わせて調整
[ -x "$VENV" ] || VENV=$(command -v python)

# モデル snapshot (995ad..) を glob 解決。曖昧なら明示パスに置換。
MODEL=$(ls -d /mnt/docker-raid/huggingface/hub/*/snapshots/995ad* 2>/dev/null | head -1)
[ -n "$MODEL" ] || { echo "MODEL snapshot 995ad* not found"; exit 1; }

# 既存 mixed_v2 cache を流用 (cache_path_for が
#   mixed_v2.jsonl.seq7168.seed<SEED>.ccr0.max0.pt を再構成 -> 命中させる)。
# 命名が合わないと再 tokenise が走る。下の SEED/ccr/seq を実 cache に一致させること。
CACHE_DIR=/mnt/docker-raid/models/esft
TRAIN_DATA="$CACHE_DIR/mixed_v2.jsonl"
SEED=5934875

echo "[$(date)] COACT_GATE START  model=$MODEL"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$VENV" -m torch.distributed.run \
  --nproc_per_node=8 \
  train_esft.py \
    --model "$MODEL" \
    --expert-config configs/mixed_v1_token_k32_p0.2.json \
    --train-data "$TRAIN_DATA" \
    --data-cache-dir "$CACHE_DIR" \
    --output-dir runs/coact_gate \
    --router-topk-random --topk-random-set 8,16,24,32 \
    --seq-length 7168 --fused-ce --random-concat-ratio 0 \
    --optimizer adafactor --grad-accum 2 \
    --max-steps 900 --eval-steps 100 --save-steps 300 \
    --seed "$SEED"
echo "[$(date)] COACT_GATE EXIT rc=$?"
