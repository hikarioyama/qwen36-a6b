# 日本語 verifiable 指示追従・自己生成 v2

日付: 2026-07-11

## 実装

`esft/selfgen_ja_verifiable_v2.py` は、独立した v2 パイプラインである。v1 の既存ファイル、run directory、生成プロトコルは変更しない。true-stock `Qwen3.6-35B-A3B` k=8 が各 seed に対して best-of-4 を生成し、正規表現・JSON parse・決定的カウントだけからなる検証器が全制約を満たした候補だけを選ぶ。LLM judge は使わない。

`prepare` は seed・汚染指紋・generation引数を manifest に凍結する。`execute` は凍結した `best-of`・temperature・token cap と異なる引数を fail-closed で拒否し、GPU 0/1 に分割した checkpoint JSONL から部分再開できる。乱数は task ごとに派生させるため、再開の有無は未処理 task の候補列を変えない。生成失敗候補は通常保存しないが、`SELFGEN_DEBUG_RAW=1` のときだけ GPU ごとの失敗 raw と検証理由を保存する。`--fixture` は CPU 構造試験専用で、`fixture_validation.json` と `fixture_summary.json` を出すが、`train.jsonl` を作らない。

## 制約レジストリ

| 型 | 分類 | M-IFEvalとの関係 |
| --- | --- | --- |
| `char_range` | 文字数上限・下限 | 同型 |
| `sentence_count` | 文数 | 同型 |
| `paragraph_count` | 段落数 | 独自 |
| `keyword_count` | 指定語の回数 | 同型 |
| `forbidden_word` | 指定語の不使用 | 同型 |
| `script_only` | ひらがな・カタカナ・漢字の限定 | 独自 |
| `bullet_count` | 箇条書き項目数 | 同型 |
| `numbered_list_count` | 番号付きリスト | 同型 |
| `heading` | Markdown見出し | 同型 |
| `json_object` | JSON objectと必須キー | 同型 |
| `markdown_table` | Markdown表のデータ行 | 独自 |
| `polite_style` | です・ます調の統一 | 同型 |
| `plain_style` | 常体の統一 | 同型 |
| `ending` | 文末表現 | 独自 |

同型/独自は M-IFEval の抽象的な制約分類との対応であり、M-IFEval の prompt・指示文・例文は参照も転記もしていない。seed は互換性のある2–3制約を組み合わせる。native-only bundle をラウンドロビンで混ぜ、manifest に同型・独自の件数を記録し、独自型が常に少なくとも 1/3 になるよう fail-closed にしている。トピックは24領域、指示プレフィックスは44種で、`template_id` は各学習行の metadata に残る。

## 汚染規律

M-IFEval日本語、MMLU、GSM8K、HumanEval、JMMLU、BFCLの6セットを protected source とする。各 source のファイル hash を manifest に残し、Unicode NFKC・casefold後に空白/句読点を落とした文字列の exact 8-gram を構築する。日本語は分かち書きに依存せず文字8-gramで照合する。固定system文とuser指示を合わせた生成prompt投影は prepare 時と execute 後の選別時に再照合し、交差があれば reject する。source が一つでも欠ける・読めない場合は `BLOCKED` として prepare/execute を止める。

この実装は M-IFEvalの input 本文を seed構築へ渡さない。本文を読む唯一の経路は汚染拒否集合への変換で、テキストは保存・表示・template化しない。

## CPU テスト

`esft/tests/test_selfgen_ja_verifiable_v2.py` は全14制約型の正例/負例、独自型1/3・トピック/テンプレ多様性、正規化8-gram、6 protected eval source contract、fixture の prepare → execute → 選別を確認する。fixture は `n=20`、GPU未使用、学習データ `n=0`、truncation `n=0` であるため能力測定・paired verdict ではない。

## Pilot 起動（Fable実行用）

GPU 実走はこの変更では行わない。Fable はまず事前確認を行う。

```bash
python3 esft/selfgen_ja_verifiable_v2.py prepare \
  --run-id pilot_20260711 --n 500 --seed 20260711 --best-of 4
python3 esft/selfgen_ja_verifiable_v2.py preflight --run-id pilot_20260711
SELFGEN_DEBUG_RAW=1 python3 esft/selfgen_ja_verifiable_v2.py execute \
  --run-id pilot_20260711 --best-of 4
```

detached 実行では同じ command を `setsid ... > pilot_20260711.outer.log 2>&1 &` で起動し、checkpoint JSONL と manifest の両方を残す。

`preflight` が true-stock identity と GPU 0/1 の空き確認に失敗した場合は `BLOCKED` として停止する。実走結果は生成 corpus の件数、best-of、制約型内訳、reject reason、truncation count を記録する。これは訓練・能力評価ではないので、paired verdict は該当しない。
