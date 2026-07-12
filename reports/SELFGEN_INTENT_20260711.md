# 意図レベル tool-call selfgen v1 (実装記録、2026-07-11)

実装対象は `esft/selfgen_toolcall_intent_v1.py`。走行済み run の再現性を守るため、
`esft/selfgen_toolcall_v1.py` は変更していない。新実装は v1 を互換性の参照として import
し、期待 trace、JSON/schema 検証、mock executor、best-of-4 の「完全一致でのみ採用」という
契約を引き継ぐ。

## Tier

| tier | request と tools | trace |
|---|---|---|
| T1 | v1 の転写 request をそのまま使用 | v1 と同一 |
| T2 | ツール名・引数名を request に出さず、静的な引数値だけを自然文テンプレートに出す | v1 と同一 |
| T3 | T2 + 3–5 個の近似名・近似 schema の distractor | v1 と同一。distractor 呼びは plan alignment 不合格 |
| T4 | T3 + 4 stage。stage 0 は parallel、次は sequential、stage 2 の `UNAVAILABLE` から stage 3 recovery | trace-first で構築し machine verify |

初期 mix は `T1:0.1,T2:0.4,T3:0.3,T4:0.2`。最大剰余方式で整数 seed 数へ配分するため、
例えば n=10 は T1=1, T2=4, T3=3, T4=2 になる。tier ごとの採用率は同一条件の best-of-4
で集計するまで未測定であり、現時点に新しい pass-rate はない。

## 値出現と降格

`request_values` は初回 request に存在すべき静的引数値、`derived_values` は先行 tool result
から得る receipt/error code として seed に明示する。`validate_value_occurrences` は静的値の
canonical JSON 表記が `natural_request` に全てあることを検査する。テンプレート生成は最大 8 回で
失敗を止める。

paraphrase の書き戻しで検査に失敗した seed は転写 request へ fallback し、`tier=T1`、
`tier_original`、`tier_downgrade`、`paraphrase.status=fallback_transcription` を保存する。これにより
失格文を意図レベルデータとして混入させない。

## Fable 運用

以下はすべて CPU 操作である。GLM/Fable の実行自体はこの repo から起動しない。

```bash
# 1. trace-first seed を作る。prepare は GPU を使用しない。
python3 esft/selfgen_toolcall_intent_v1.py prepare \
  --run-id intent_r1 --n 5000 \
  --tier-mix 'T1:0.1,T2:0.4,T3:0.3,T4:0.2'

# 2. T2/T3/T4 だけを Fable/GLM に渡す JSONL として出力する。
python3 esft/selfgen_toolcall_intent_v1.py emit-paraphrase-batch --run-id intent_r1

# 3. Fable は各行に対して次の最小 schema を JSONL で返す。
#    {"seed_id":"seed-0001","natural_request":"..."}
python3 esft/selfgen_toolcall_intent_v1.py ingest-paraphrase \
  --run-id intent_r1 --input /path/to/fable_writeback.jsonl

# 4. cx/Grok の一意可解性審査用 JSONL を出力する（judge 文は学習データに入れない）。
python3 esft/selfgen_toolcall_intent_v1.py emit-audit-batch --run-id intent_r1

# 5. 審査でテンプレートを修正した後だけ、既存と同じ best-of-4 を実行する。
#    実行前に campaign の GPU preflight と job watcher 登録を完了する。
python3 esft/selfgen_toolcall_intent_v1.py execute --run-id intent_r1 --best-of 4
```

`paraphrase_batch.jsonl` は転写 request、静的値の canonical literal list、derived value の由来、
書き戻し schema を含む。`audit_batch.jsonl` は tools、natural request、期待 trace、および
`PASS|FAIL`/理由/代替 trace を返すための審査 prompt を含む。audit は少数サンプルから開始し、
不合格テンプレートを直してから全量化する。

`execute` は通常時のみ v1 と同じ k=8 true-stock、GPU 0/1、best-of=4 を要求する。生成 benchmark
の summary には tier 別 generated/accepted、rejection reasons、`truncation_count` を記録する。
今回の実装作業ではモデルをロードせず、GPU job も起動していない。

## CPU 検証

`python3 esft/tests/test_selfgen_toolcall_intent_v1.py` は次を確認する。

- T1 の v1 既存フィールド（tools、request、期待 trace を含む）が同一。
- T2/T3/T4 の静的値出現、ツール名・引数名の非露出。
- distractor を呼ぶ候補が expected trace として採用されないこと。
- default tier mix の n=10 配分。
- T4 の parallel + sequential + error recovery trace の mock-executor 検証。
- paraphrase の値欠落が転写 fallback と tier 降格記録になること。
