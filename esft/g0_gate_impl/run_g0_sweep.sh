#!/usr/bin/env bash
# G0 gate-shaping sweep: MMLU n=600 choice-logprob over 6 shaped configs + 2
# stock controls (k32, k8), serial on aux-host's 2 GPUs, then paired McNemar
# verdicts of each shaped arm vs the k32 stock control.
#
# G0 survives iff some shaped arm beats k32-stock by >= +1.0pt (McNemar p<0.05):
# training-free gate reshaping recovers the k32 knowledge regression.
#
# Deploy first (once):
#   scp gate_shaping.py g0_debug.py apply_harness_patch.py run_g0_sweep.sh aux-host:~/esft/
#   ssh aux-host 'cd ~/esft && ~/esft-work/venv/bin/python apply_harness_patch.py'
#   ssh aux-host 'cd ~/esft && G0_GPU=0 ~/esft-work/venv/bin/python g0_debug.py'   # sanity
#
# Run (GPUs must be free; the MAIN session launches this, not the impl agent):
#   ssh aux-host 'cd ~/esft && nohup bash run_g0_sweep.sh > reports/eval/g0_sweep.log 2>&1 &'
set -euo pipefail

PY="$HOME/esft-work/venv/bin/python"
HARNESS="$HOME/esft/eval_harness.py"
BENCH="mmlu"
N="${G0_N:-600}"
GPUS="${G0_GPUS:-0,1}"
REPORT_DIR="$HOME/esft/reports/eval"
mkdir -p "$REPORT_DIR"

# tag | topk | G0_SHAPE | G0_PARAM     (shape=- means stock control, no hook)
CONFIGS=(
  "g0_k32_base|32|-|-"
  "g0_k8_base|8|-|-"
  "g0_temp_0.5|32|temp|0.5"
  "g0_temp_0.7|32|temp|0.7"
  "g0_masscut_0.80|32|masscut|0.80"
  "g0_masscut_0.90|32|masscut|0.90"
  "g0_rankdamp_0.25|32|rankdamp|0.25"
  "g0_rankdamp_0.50|32|rankdamp|0.5"
)

echo "=== G0 sweep start $(date -Is)  bench=$BENCH n=$N gpus=$GPUS ==="

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag topk shape param <<< "$cfg"
  echo ""
  echo "--- [$tag] topk=$topk shape=$shape param=$param  $(date -Is) ---"

  if [[ "$shape" == "-" ]]; then
    # Stock control: strip any G0_* from the env so the OFF arm is provably clean.
    env -u G0_SHAPE -u G0_PARAM -u G0_DEBUG -u G0_DEBUG_CALLS \
      "$PY" "$HARNESS" \
        --model base --benchmark "$BENCH" --n "$N" --topk "$topk" \
        --gpus "$GPUS" --tag "$tag"
  else
    # Shaped arm: G0_DEBUG on so the first calls log eff-k / mass shift per GPU.
    G0_SHAPE="$shape" G0_PARAM="$param" G0_DEBUG=1 G0_DEBUG_CALLS=8 \
      "$PY" "$HARNESS" \
        --model base --benchmark "$BENCH" --n "$N" --topk "$topk" \
        --gpus "$GPUS" --tag "$tag"
  fi
done

echo ""
echo "=== paired verdicts vs k32 stock control  $(date -Is) ==="
BASE_ITEMS="$REPORT_DIR/g0_k32_base_items.json"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag topk shape param <<< "$cfg"
  [[ "$tag" == "g0_k32_base" ]] && continue
  echo ""
  echo "### $tag  vs  g0_k32_base"
  "$PY" "$HARNESS" --paired-verdict "$BASE_ITEMS" "$REPORT_DIR/${tag}_items.json" \
    --verdict-key correct || echo "  (paired verdict failed for $tag)"
done

echo ""
echo "=== G0 sweep done $(date -Is) ==="
echo "Verdict rule: a shaped arm with delta >= +0.010 and McNemar significant=True"
echo "beats the k32 regression training-free -> G0 SURVIVES."
