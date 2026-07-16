#!/bin/bash
# G3 傾き判定 v6fresh 版 (ローカル、2 GPU 必須): MMLU n=600 choice-logprob を 3 腕直列。
# 床は base@k8 (盛り防止)。腕: base@k8 / base@k32 / v6fresh1000@k32。
# 雛形 = run_g3_eval_local.sh (20260712)。腕のパスと TS のみ変更 (same-condition)。
set -euo pipefail
BASE=$HOME/models/esft/qwen36_35b_base
V6=$HOME/models/esft/v6fresh_1000_bf16
TS=20260716
cd $HOME/projects/qwen36-a6b/esft
test -f "$V6/config.json" || { echo "BLOCKED: v6fresh export 未完備"; exit 1; }
export CUDA_VISIBLE_DEVICES=0,1
for arm in "base 8 $BASE" "base 32 $BASE" "v6fresh1000 32 $V6"; do
  set -- $arm; name=$1; k=$2; path=$3
  tag="g3_${TS}_clp_${name}_k${k}_mmlu"
  echo "=== ARM $tag start $(date -u +%s)"
  python3 eval_harness.py --model base --model-path "$path" --benchmark mmlu \
    --topk "$k" --n 600 --choice-logprob --tag "$tag"
done
echo "=== 3 腕完了 → paired 計算"
python3 - << 'EOF'
import json, math, os
R = os.path.expanduser('~/esft/reports/eval')
def load(tag):
    items = json.load(open(f'{R}/{tag}_items.json'))
    return {it['item_key']: bool(it['correct']) for it in items}
b8  = load('g3_20260716_clp_base_k8_mmlu')
b32 = load('g3_20260716_clp_base_k32_mmlu')
v32 = load('g3_20260716_clp_v6fresh1000_k32_mmlu')
def paired(a, b, la, lb):
    keys = sorted(set(a) & set(b))
    n = len(keys)
    aa = sum(a[k] for k in keys); bb = sum(b[k] for k in keys)
    b01 = sum((not a[k]) and b[k] for k in keys)
    b10 = sum(a[k] and (not b[k]) for k in keys)
    delta = (bb - aa) / n
    if b01 + b10 > 0:
        z = (abs(b01 - b10) - 1) / math.sqrt(b01 + b10)
        p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    else:
        p = 1.0
    print(f"{lb} vs {la}: {aa}/{n} -> {bb}/{n}  delta={delta:+.4f}  "
          f"discordant {la}-only={b10} {lb}-only={b01}  McNemar p={p:.4g}")
paired(b8,  v32, 'base@k8',  'v6fresh@k32')
paired(b32, v32, 'base@k32', 'v6fresh@k32')
paired(b8,  b32, 'base@k8',  'base@k32')
EOF
echo "G3_V6FRESH_DONE"