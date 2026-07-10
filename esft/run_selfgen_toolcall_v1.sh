#!/usr/bin/env bash
set -euo pipefail

# Detached production launcher.  The watcher binds this PID before CUDA work.
RUN_ID="${1:?usage: $0 RUN_ID}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="/tmp/${RUN_ID}.pid"
LOG_FILE="${ROOT}/esft/data/selfgen_toolcall_v1/${RUN_ID}/launcher.log"

mkdir -p "$(dirname "${LOG_FILE}")"
echo "$$" > "${PID_FILE}"
sleep 45
exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /usr/bin/python3 "${ROOT}/esft/selfgen_toolcall_v1.py" execute \
  --run-id "${RUN_ID}" --n 500 --best-of 4 --temperature 0.7 --max-new 512 \
  >>"${LOG_FILE}" 2>&1
