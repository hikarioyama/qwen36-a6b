# Router-only 訓練モード（2026-07-12）

## 目的と根拠

`--method router-only` は、Qwen3.6-35B-A3B の routed expert FFN、shared
expert、attention、embedding、LM head を全て凍結し、各 MoE layer の
`mlp.gate.weight` だけを更新する。`--router-top-k 32` は通常どおり gate に
設定されるため、学習時も k=32 の選択である。

この分岐は、joint 訓練 1,000 step で pre-top-k entropy が 5.145 から 5.328
へ単調に平坦化し、MMLU@k32 の k32 化コストが base@k8 比 -3.17 pt だったという
観測への機構仮説を検査するためのもの。後者は paired n=600 の既測定値であり、
router-only の結果ではない。新規ゲートは同一 machine / harness revision /
item order / seed / prompt mode / token cap / batch size の paired 条件で取り直す
まで `same-condition` とは呼ばない。

## 実装

- router-only は full-ffn と同じ FSDP `SHARDED_STATE_DICT` + DCP checkpoint
  経路を使う。optimizer は gate のみ、LR は生の `--learning-rate`
  （`--router-lr-mult` はこの mode では実効 1.0）、weight decay は 0。
- 最終 HF export は full state dict を保存する。凍結 FFN は base から一度も
  optimizer に渡らないため、base の値のまま export される。
- `--router-anchor-ref run-start` は従来どおり開始時 gate の CPU snapshot を使う。
  `--router-anchor-ref PATH` は PATH 内の HF safetensors から必要な
  `*.mlp.gate.weight` だけを直接読み、モデル全体や teacher をロードしない。
  PATH は `model.safetensors` と sharded `model.safetensors.index.json` の両方に
  対応する。
- router を訓練する全 method で、既存の eval forward の gate hook から全 MoE
  layer・全 eval token の pre-top-k softmax を集計する。追加 forward はない。
  eval loss の直後に以下を一行出す（entropy は nats、`r18`/`r932` は各 token の
  降順 rank 質量を layer 平均した値）。

  ```text
  [router-obs] step=N entropy=X.XXX r18=0.XXX r932=0.XXX
  ```

`[router-obs]` は学習時 validation blocks の観測であり、ベンチマーク結果では
ない。MMLU/GSM8K は n=600、HumanEval は n=164 を同一条件の paired harness で
別途記録し、HumanEval は `max_new=4096` と両 arm の truncation count も併記する。

## Fable 発射手順（300--500 step）

以下の `BASE` は revision と tensor fingerprint を確認済みの真の A3B base
directory とする。PATH anchor を同じ `BASE` に固定することで、途中 export や
resume を起点にしても原初 base への anchor を維持できる。実行前に Fable 側へ
`train_fullffn_dcp.py` を配備し、`preflight` と v4 render check を通す。

```bash
set -euo pipefail
ESFT_DIR=/mnt/docker-raid/models/esft
VENV=~/esft-venv/bin/python
BASE=/absolute/path/to/verified-Qwen3.6-35B-A3B-base
OUT=codex_runs/router_only_v4_20260712
DATA=data/v4_20260711.jsonl
cd "$ESFT_DIR"

test -f "$BASE/config.json"
test -f "$DATA"
"$VENV" codex_harness.py preflight
"$VENV" - <<'PY'
import json
result = json.load(open('data/v4_render_check.json'))
assert result['errors'] == 0, result
print('v4 render check:', result['ok'], 'rows')
PY

# v4 の token cache が無い場合だけ、単独 process で先に作る。
"$VENV" train_fullffn_dcp.py --model "$BASE" --method router-only \
  --expert-config configs/fullffn_probe.json --train-data "$DATA" \
  --output-dir "$OUT" --seq-length 7168 --router-top-k 32 \
  --optimizer adafactor --grad-accum 4 --per-device-batch-size 1 \
  --max-steps 300 --prepare-data-only

env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$VENV" -m torch.distributed.run --nproc_per_node=8 \
  train_fullffn_dcp.py --model "$BASE" --method router-only \
  --expert-config configs/fullffn_probe.json --train-data "$DATA" \
  --output-dir "$OUT" --seq-length 7168 --router-top-k 32 --fused-ce \
  --optimizer adafactor --learning-rate 1e-5 --weight-decay 0.0 \
  --router-anchor-weight 0.15 --router-anchor-ref "$BASE" \
  --grad-accum 4 --per-device-batch-size 1 --max-steps 300 \
  --eval-steps 100 --save-steps 100 --logging-steps 1 \
  2>&1 | tee "$OUT/train.log"
```

500 step に延長する場合は同じ条件のまま `--max-steps 500` にし、`checkpoint-300`
から再開する場合だけ `--resume-from-checkpoint "$OUT/checkpoint-300"` を足す。
anchor PATH、learning rate、world size、weight decay を変えた resume は checkpoint
marker が拒否する。発射前に、長時間・multi-GPU job 用の Grok preflight と job-event
watch を別途完了させること。

## CPU 検証

`esft/tests/test_fullffn_joint_trainer.py` は実モデルをロードせず、合成 tensor で
次を確認する。

- trainable Parameter と optimizer group が gate だけであること。
- 小さな synthetic safetensors の `PATH` anchor が gate weight と一致すること。
- eval observer が一回の既存 forward から entropy / rank mass を計算すること。
