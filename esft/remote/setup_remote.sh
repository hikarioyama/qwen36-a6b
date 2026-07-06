#!/usr/bin/env bash
# ESFT remote bootstrap — sudo 不要、コンテナ内 non-root で完結する。
# 前提: python3.10+ / git or curl / NVIDIA driver passthrough (--gpus all) / 書込可能な作業 volume
set -euo pipefail

WORK="${ESFT_WORK:-$HOME/esft-work}"
mkdir -p "$WORK" && cd "$WORK"

echo "== [1/4] uv (self-contained, no sudo)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "== [2/4] venv + deps (pip wheel が CUDA runtime を同梱するので base image の CUDA は不問)"
uv venv --python 3.12 "$WORK/venv"
source "$WORK/venv/bin/activate"
# torch は sm_120 (Blackwell) 対応の安定版を明示
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers==5.7.0" accelerate datasets safetensors sentencepiece numpy "huggingface_hub[cli]"
# Phase 1 の top-k sweep / serve 用 (訓練だけなら省略可: SKIP_VLLM=1)
[ "${SKIP_VLLM:-0}" = "1" ] || uv pip install vllm

echo "== [3/4] GPU smoke"
python "$(dirname "$0")/gpu_smoke.py"

echo "== [4/4] モデル DL (HF から直接。~70GB, 中断可・再開可)"
if [ "${SKIP_MODEL_DL:-0}" != "1" ]; then
  hf download Qwen/Qwen3.6-35B-A3B --local-dir "$WORK/models/Qwen3.6-35B-A3B"
fi

echo "DONE. activate: source $WORK/venv/bin/activate"
