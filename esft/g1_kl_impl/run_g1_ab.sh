#!/bin/bash
# G1 step 2: CE-only vs CE+KL A/B on the selected-delta (k=32) path.
# Three arms, run SERIALLY (each uses all 8 GPUs under plain DDP):
#   A   : CE only                       (--kl-beta 0, no teacher -> bit-identical delta path)
#   B1  : CE + 0.1 * KL(teacher_k8||student)
#   B2  : CE + 0.5 * KL(teacher_k8||student)
# Same selection config, v3 cache, seq7168, --router-top-k 32 (fixed), router FROZEN
# (no --train-router), Adafactor. checkpoints at 250 and 500.
#
# 前提: run_precompute_k8.sh が先に完走し KL_DIR/manifest.json が存在すること。
# 8 GPU 占有中なら回すな。
set -euo pipefail
cd "$(dirname "$0")"

VENV=~/esft-work/venv/bin/python
[ -x "$VENV" ] || VENV=$(command -v python)

MODEL=$(ls -d /mnt/docker-raid/huggingface/hub/*/snapshots/995ad* 2>/dev/null | head -1)
[ -n "$MODEL" ] || { echo "MODEL snapshot 995ad* not found"; exit 1; }

CACHE_DIR=/mnt/docker-raid/models/esft
SEQ=${SEQ:-7168}
CONFIG=${CONFIG:-configs/mixed_v1_token_k32_p0.2.json}
TRAIN_DATA=${TRAIN_DATA:-$CACHE_DIR/v3.jsonl}
KL_DIR=${KL_DIR:-$CACHE_DIR/k8_teacher_v3_seq${SEQ}_top64}
SEED=5934875
STEPS=${STEPS:-500}
CHUNK=${KL_TOKEN_CHUNK:-2048}
OUT_ROOT=${OUT_ROOT:-runs/g1_kl}

[ -f "$KL_DIR/manifest.json" ] || { echo "KL_DIR manifest missing: $KL_DIR (run run_precompute_k8.sh first)"; exit 1; }

run_arm () {   # $1=name  $2=kl-beta  $3=extra-kl-args
  local name=$1 beta=$2; shift 2
  echo "[$(date)] G1 ARM $name START (kl_beta=$beta)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$VENV" -m torch.distributed.run \
    --nproc_per_node=8 \
    train_esft.py \
      --model "$MODEL" \
      --method delta \
      --expert-config "$CONFIG" \
      --train-data "$TRAIN_DATA" \
      --data-cache-dir "$CACHE_DIR/cache" \
      --output-dir "$OUT_ROOT/$name" \
      --router-top-k 32 \
      --seq-length "$SEQ" --fused-ce --random-concat-ratio 0.0 \
      --optimizer adafactor --grad-accum 2 \
      --max-steps "$STEPS" --eval-steps 100 --save-steps 250 \
      --seed "$SEED" "$@"
  echo "[$(date)] G1 ARM $name EXIT rc=$?"
}

# A: CE only. No --kl-logits-dir -> kl_active False -> bit-identical to the plain delta path.
run_arm A 0.0

# B1 / B2: CE + beta*KL over the precomputed top-64 teacher support.
run_arm B1 0.1 --kl-logits-dir "$KL_DIR" --kl-beta 0.1 --kl-token-chunk "$CHUNK"
run_arm B2 0.5 --kl-logits-dir "$KL_DIR" --kl-beta 0.5 --kl-token-chunk "$CHUNK"

echo "[$(date)] G1 A/B complete -> $OUT_ROOT/{A,B1,B2}"
