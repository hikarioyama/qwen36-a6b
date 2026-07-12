#!/bin/bash
# G3 傾き判定 (ローカル、2 GPU 必須): MMLU n=600 choice-logprob を 3 腕直列。
# 床は base@k8 (盛り防止)。腕: base@k8 / base@k32 / joint1000@k32。
# 完走後に per-item paired (McNemar + CI) を cross-model 手動計算で出す。
set -euo pipefail
BASE=$HOME/models/esft/qwen36_35b_base
JOINT=$HOME/models/esft/joint1000_20260712_bf16
TS=20260712
cd $HOME/projects/qwen36-a6b/esft
test -f "$JOINT/config.json" || { echo "BLOCKED: joint1000 export 未完備"; exit 1; }
export CUDA_VISIBLE_DEVICES=0,1
for arm in "base 8 $BASE" "base 32 $BASE" "joint1000 32 $JOINT"; do
  set -- $arm; name=$1; k=$2; path=$3
  tag="g3_${TS}_clp_${name}_k${k}_mmlu"
  echo "=== ARM $tag start $(date -u +%s)"
  python3 eval_harness.py --model base --model-path "$path" --benchmark mmlu \
    --topk "$k" --n 600 --choice-logprob --tag "$tag"
done
echo "=== 3 腕完了 → paired 計算"
python3 - << 'EOF'
import json, math
R = __import__('os').path.expanduser('~/esft/reports/eval')
def load(tag):
    items = json.load(open(f'{R}/{tag}_items.json'))
    return {it['item_key']: bool(it['correct']) for it in items}
b8  = load('g3_20260712_clp_base_k8_mmlu')
b32 = load('g3_20260712_clp_base_k32_mmlu')
j32 = load('g3_20260712_clp_joint1000_k32_mmlu')
def paired(a, b, la, lb):
    keys = sorted(set(a) & set(b))
    n = len(keys)
    aa = sum(a[k] for k in keys); bb = sum(b[k] for k in keys)
    b01 = sum((not a[k]) and b[k] for k in keys)  # b だけ正解
    b10 = sum(a[k] and (not b[k]) for k in keys)  # a だけ正解
    delta = (bb - aa) / n
    # McNemar (exact に近い正規近似) + paired CI (差の二項近似)
    if b01 + b10 > 0:
        z = (abs(b01 - b10) - 1) / math.sqrt(b01 + b10)
        from math import erf
        p = 2 * (1 - 0.5 * (1 + erf(z / math.sqrt(2))))
    else:
        p = 1.0
    se = math.sqrt((b01 + b10) - (b01 - b10) ** 2 / n) / n
    print(f'{lb} vs {la}: {aa}/{n} -> {bb}/{n}  Δ{delta*100:+.2f}pt  '
          f'CI95 [{(delta-1.96*se)*100:+.2f},{(delta+1.96*se)*100:+.2f}]  '
          f'McNemar p={p:.4f}  (n={n}, same items, choice-logprob)')
paired(b8,  b32, 'base@k8',  'base@k32')
paired(b32, j32, 'base@k32', 'joint1000@k32')
paired(b8,  j32, 'base@k8',  'joint1000@k32')
EOF
echo "G3_EVAL_DONE"
