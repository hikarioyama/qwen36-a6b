# Recipe 化: mock_* テンプレ名燃料で汚染された run (破棄前の記録)

2026-07-14、BFCL −31pt (機構 = 関数名複写忠実度の喪失、DEVLOG 参照) の確定を受け、
mock_* テンプレ名 selfgen を燃料に含む checkpoint 線を「汚染線」として破棄可と判断
(ユーザー許可)。破棄前に再現に必要な情報をここに固定する。weights は消えても、
このレシピ + repo のコード + 燃料 (vault 保全) で同一 run を再現できる。

## 共通構成 (両 run 同一)

- trainer: `esft/train_fullffn_dcp.py` (repo 管理)、8 GPU torchrun、FSDP + DCP checkpoint
- `--method full-ffn --expert-config configs/fullffn_probe.json --seq-length 7168
  --router-top-k 32 --fused-ce --router-tail-scale 0.5 --random-concat-ratio 0
  --optimizer adafactor --weight-decay 0.0 --grad-accum 4 --per-device-batch-size 1
  --max-steps 1000 --eval-steps 100 --save-steps 300 --logging-steps 1
  --deterministic-fullffn`
- replay: `mixed_v2.jsonl --replay-ratio 0.30`

## Run 1: fullffn_tail05_frozen_router_20260712 (= tail05, v4 出力)

- 出発点: stock Qwen3.6-35B-A3B から fresh、router 凍結
- 燃料: v4 corpus (322,262 行; mock_* intent selfgen r1-r3 + toucan + code 系)
- 結果 (measured): 1000/1000 完走。MMLU 84.83 (α0.5, 床超え点推定 ns)、JMMLU/GSM8K ns、
  **BFCL 53/100 vs base 84/100 (n=100 paired, McNemar p=1.2e-07) = 主目的軸 −31pt**
- 教訓: 「審判の不在は無罪ではない」。3 軸 ns でも主目的軸は壊れうる

## Run 2: fullffn_v5intent_from_tail05_20260714 (v5 → v5.1)

- 出発点: tail05 export (fp32) から fresh (= 汚染の継承)
- 燃料: v5_20260714.jsonl (sha256 743bb819…) step 0-300 → v5p1_20260714.jsonl
  (sha256 369e5533…, T4 gold 倍増版) step 300-371
- 経過: step 343 で燃料切替の空振り検出 → checkpoint-300 から v5.1 resume (loss 連続、
  model_match=True) → **step ~371 でユーザー決定により中止** (燃料が mock_* 欠陥を共有)
- 結果: 境界測定なし (中止)。loss は 0.55-0.66 帯で健全だった = 「loss が下がる」と
  「目的軸が良くなる」は別物、の実例

## 破棄対象 (recipe 化済み → 削除可)

- gpu-host: `codex_runs/fullffn_tail05_frozen_router_20260712` (1.1T)、
  `codex_runs/fullffn_v5intent_from_tail05_20260714` (249G)
- ローカル vault: tail05 系のコピーがあれば同様 (燃料 jsonl と eval 生データは**保持**)

## 保持するもの (削除しない)

- 燃料 jsonl 全て (vault) — 失敗の再現・分析材料
- eval 生データ (esft/reports/eval/codex_runs/) — −31pt の証拠
- B2 期 expert-patch (b2_1000_expert_patch.safetensors, 8.4G 級) — 非汚染 (BFCL 88/100 で無罪確認済み)
