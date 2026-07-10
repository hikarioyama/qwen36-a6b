#!/usr/bin/env bash
# GRPO [T]-phase launch (remote 2×RTX PRO 6000, torchrun DDP).
#
# Prereq: a rollouts jsonl from the [G] phase (rollout_gen.py / inc0_gen.py) and the
# SFT starting patch (esft-qwen-patch-v1, trained_as=residual-delta). The starting
# policy delta is initialised as (SFT effective slice − base), so KL references the
# true base with the delta disabled.
#
# GPU note: DO NOT run while the Japanese training tmux (jptrain) holds GPUs 0,1.
# This script assumes the remote training node is free.
set -euo pipefail

: "${VENV:?Set VENV to the virtual-environment bin directory}"
: "${PATCH:?Set PATCH to the starting patch file}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ESFT=${ESFT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
ROLLOUTS=${ROLLOUTS:-$ESFT/rl/rollouts/inc0_prefill.jsonl}
PROMPTS=${PROMPTS:-$ESFT/rl/data/grpo_prompts.jsonl}
OUT=${OUT:-$ESFT/rl/runs/grpo_c1}
REWARD_KEY=${REWARD_KEY:-lenient}
KL_BETA=${KL_BETA:-0.02}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
NPROC=${NPROC:-2}

# The expert config is embedded in the patch metadata; dump it to a temp json.
CFG=$(mktemp /tmp/grpo_expert_cfg.XXXX.json)
"$VENV/python" - "$PATCH" "$CFG" <<'PY'
import sys, json
from safetensors import safe_open
with safe_open(sys.argv[1], "pt") as f:
    ec = json.loads(f.metadata()["expert_config"])
json.dump(ec, open(sys.argv[2], "w"))
print("expert_config ->", sys.argv[2], "layers:", len(ec["experts"]))
PY

cd "$ESFT"
"$VENV/torchrun" --standalone --nproc_per_node="$NPROC" rl/grpo_train.py \
    --model "$MODEL" \
    --expert-config "$CFG" \
    --rollouts "$ROLLOUTS" \
    --prompts-data "$PROMPTS" \
    --init-patch "$PATCH" \
    --output-dir "$OUT" \
    --reward-key "$REWARD_KEY" \
    --kl-beta "$KL_BETA" \
    --learning-rate "$LR" \
    --epochs "$EPOCHS" \
    --max-seq-len 7168

# On >1-epoch reuse, add:  --seq-ratio-mode gspo --clip-eps 0.2
# The next [M] phase merges $OUT/delta_state.safetensors -> ckpt for the next [G].
echo "done -> $OUT/delta_state.safetensors"
