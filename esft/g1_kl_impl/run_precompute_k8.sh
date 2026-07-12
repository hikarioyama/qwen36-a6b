#!/bin/bash
# G1 step 1: precompute the k=8 teacher's top-64 logits for the v3 packed cache.
# 8 independent per-GPU processes (NO torchrun), each owns a contiguous block range and
# writes its own resume-safe shards. Re-running skips shards already on disk.
#
# 実行前提: 8 GPU が空いていること。frozen forward のみ (grad なし)。所要は下の NOTES 参照。
set -euo pipefail
cd "$(dirname "$0")"

VENV=${VENV:-python}
[ -x "$VENV" ] || VENV=$(command -v python)

MODEL=$(ls -d /mnt/docker-raid/huggingface/hub/*/snapshots/995ad* 2>/dev/null | head -1)
[ -n "$MODEL" ] || { echo "MODEL snapshot 995ad* not found"; exit 1; }

CACHE_DIR=/mnt/docker-raid/models/esft
SEQ=${SEQ:-7168}
CACHE_FILE=${CACHE_FILE:-$CACHE_DIR/cache/v3.jsonl.seq${SEQ}.seed5934875.ccr0.0.max0.pt}
OUT_DIR=${OUT_DIR:-$CACHE_DIR/k8_teacher_v3_seq${SEQ}_top64}
TOPK=${TOPK:-64}
CHUNK=${CHUNK:-512}
WORLD=${WORLD:-8}

[ -f "$CACHE_FILE" ] || { echo "CACHE_FILE not found: $CACHE_FILE"; exit 1; }
mkdir -p "$OUT_DIR"

echo "[$(date)] PRECOMPUTE_K8 START model=$MODEL cache=$CACHE_FILE out=$OUT_DIR world=$WORLD"
pids=()
for r in $(seq 0 $((WORLD - 1))); do
  CUDA_VISIBLE_DEVICES=$r PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$VENV" precompute_k8_logits.py \
      --model "$MODEL" \
      --cache-file "$CACHE_FILE" \
      --out-dir "$OUT_DIR" \
      --rank "$r" --world-size "$WORLD" \
      --top-k "$TOPK" --chunk-size "$CHUNK" --teacher-top-k 8 \
      > "$OUT_DIR/precompute.rank$r.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "[$(date)] PRECOMPUTE_K8 EXIT rc=$rc  (per-rank logs in $OUT_DIR/precompute.rank*.log)"
exit $rc
