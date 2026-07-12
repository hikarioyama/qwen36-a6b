# Corpus Judge v1 — 機械検証済みデータの厳格二次選別

作成日: 2026-07-11  
対象: `selfgen_toolcall_v1`、`selfgen_ja_verifiable_v2`、decontam 済み Toucan parquet  
実行条件: CPU/ローカルファイルのみ。ネットワーク、SSH、GPU は使用しない。

## 結論

`corpus_judge_v1` は、機械検証を通った例をそのまま採用せず、Codex/cx
ジョブに厳格な意味品質判定をさせるための二段目フィルタである。ランナー
自身は判定モデルを実行しない。rubric と最大20件のレコードを一つの prompt
ファイルにし、後続 cx ジョブの JSONL 出力だけを検証・追記する。

この設計は「executor が成功した」ことと「有益な agent 学習信号である」ことを
混同しない。特に、正解 tool 名・全引数・順序を request に埋め込んだ転写例は、
実行可能でも reject である。

## Rubric

- [tool call rubric](../esft/judge/rubric_toolcall_v1.md) SHA-256
  `5ec6c5b3937df1697700638efaa24881f50e794ef82e8fb06c260a19618db7fe`
- [Japanese rubric](../esft/judge/rubric_ja_v1.md) SHA-256
  `7ec46ce0a54585d5762e637565a82f2474b6a42c1ab4e7ea9c64aaf6dddc25fa`

どちらも (a) 指示の自然さ・情報量、(b) 応答/tool call の整合性、(c) 反復・
破綻文・不自然な混在などの有害信号を確認する。出力は `accept` / `reject` /
`borderline` と一行理由である。迷い・証拠不足・評価不能は **reject** とする。
`borderline` は、核となる品質を満たし、軽微で明示可能な不確実性が一つだけ
残る場合に限定する。

## ランナー仕様

実装は [corpus_judge_v1.py](../esft/judge/corpus_judge_v1.py)。入力は一つの
JSONL または一つの parquet shard、rubric、`--start`/`--end`（ゼロ始まり・end
exclusive）を受ける。Toucan の JSON-in-string `messages`/tool schema と selfgen
の object 形式を正規化する一方、元データにある既存の quality assessment は
judge prompt に渡さない。先入観ラベルの漏れを防ぐためである。

`prepare` は prompt ファイルだけを生成する。既存 `judgements.jsonl` の id は
skip し、古い prompt は監査のため残すが、今回 cx に渡すべきファイルだけを
`pending_batches.json` に列挙する。`batch_manifest.jsonl` は各 batch の id 群と
rubric SHA を固定する。`append` は cx 出力の必須5列、verdict、理由の一行制約、
rubric SHA、prepared batch/id の対応、既存 id 重複を検証してから
`judgements.jsonl` に append する。したがって ledger の各行は次だけを持つ。

```json
{"id":"...","verdict":"accept|reject|borderline","reason":"...","rubric_sha":"...","batch_id":"..."}
```

`summary` は `n`、accept 率、3 verdict の数、理由文字列ごとの分布を出力する。
これは判定を再実行しない。

## 全量運用

まず source ごとに出力ディレクトリを分ける。例（toolcall、先頭100 record）:

```bash
python3 esft/judge/corpus_judge_v1.py prepare \
  --input esft/data/selfgen_toolcall_v1/20260711_toolcall_v11_prod5000/train.jsonl \
  --rubric esft/judge/rubric_toolcall_v1.md \
  --output-dir esft/judge/runs/toolcall_v11_prod5000 \
  --start 0 --end 100 --batch-size 10
```

`pending_batches.json` にある各 Markdown をそのまま独立 cx job に渡す。cx は
prompt が要求する JSONL だけを返す。結果を一時ファイルに置き、次で ledger に
追記する。

```bash
python3 esft/judge/corpus_judge_v1.py append \
  --input /path/to/cx-output.jsonl \
  --rubric esft/judge/rubric_toolcall_v1.md \
  --output-dir esft/judge/runs/toolcall_v11_prod5000
python3 esft/judge/corpus_judge_v1.py summary \
  --output-dir esft/judge/runs/toolcall_v11_prod5000
```

日本語 JSONL では `rubric_ja_v1.md` を使う。Toucan は一 shard ごとに同じ操作を
行い、`--input /mnt/vault/.../train-xxxxx.parquet` とする。vault 側へは書かない。

推奨 cx 分割は **10 records/batch**（本実装の許容範囲は10--20）。Toucan は tool
schema と会話が長いので 10 固定から開始し、prompt 長・判定欠落・理由の具体性を
spot check する。短い selfgen/日本語は15、十分短いことを確認済みの場合のみ20へ
上げる。judge の温度・モデル名・実行日時は各 cx job の外部 manifest に保存する。
この報告書の数値は評価ベンチの same-condition paired 比較ではなく、下記の固定
source・固定rubricによる品質監査の単一条件集計である。

## パイロット: toolcall v11 prod5000 の先頭100 records

対象は
`esft/data/selfgen_toolcall_v1/20260711_toolcall_v11_prod5000/train.jsonl` のファイル
先頭100 records（同一 source、同一 tool-call rubric SHA、手動 Codex 判定）。結果は
[pilot_judgements.jsonl](../esft/judge/pilot_judgements.jsonl) に保存した。これは
機械検証の accept/reject ではなく二次品質判定である。

| verdict | n | 割合 |
|---|---:|---:|
| accept | 0 | 0.0% |
| reject | 100 | 100.0% |
| borderline | 0 | 0.0% |

代表理由（n=100）: request が exact tool 名、全引数、並列/逐次/回復順序まで明記し、
assistant はそれを書き写すだけである。`single` (n=26)、`parallel` (n=25)、
`multi_turn` (n=24)、`error_recovery` (n=25) のすべてで同じ決定的欠点を確認した。
先頭100 records は id が連番とは限らず、`seed-0062` は upstream で除外済みのため
含まれないが、**物理ファイルの先頭100行**を判定対象にした。

この `accept=0/100` は、データ全量や将来の意図レベル selfgen の accept 率を推定
する統計値ではない。same-condition は上記 source/rubric/first-100 に限られ、paired
比較・truncation は生成 benchmark ではないため該当しない。
