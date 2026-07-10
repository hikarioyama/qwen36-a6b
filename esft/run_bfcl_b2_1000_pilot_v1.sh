#!/usr/bin/env bash
set -o pipefail

# The short arm-delay gives the durable watcher a chance to bind this exact PID
# before either GPU is touched.  The launcher PID is intentionally written
# outside the not-yet-created result directory.
echo "$$" > /tmp/20260710_bfcl_b2_1000_pilot_v1.pid
sleep 45
exec env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0,1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  /usr/bin/python3 esft/bfcl_pilot.py campaign \
    --run-id 20260710_bfcl_b2_1000_pilot_v1 \
    --n 300 \
    --batch-size 4 \
    --max-new 512 \
    --noninferiority-margin 0.02
