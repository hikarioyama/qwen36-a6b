# 足場付き自己生成データパイプライン v1 — ツールコール軸

更新: 2026-07-11 JST  
対象run: `20260711_toolcall_v1_pilot500_r3`  
状態: **PREPARED / GPU preflight BLOCKED / generated n=0**

## 結論

v1 の実装、CPU検証、500件の生成入力凍結は完了した。真stock
`Qwen3.6-35B-A3B` revision `995ad96eacd98c81ed38be0c5b274b04031597b0`
（expert fingerprint `3a1ca2a61e9a86af44c5114d72a9033504d3a20e27c3c6838f4162b87e3aa315`）を
GPU 0/1 のみで使うよう固定し、B2 patch は参照も使用もしない。

ただし起動直前の `nvidia-smi` が exit 9（NVIDIA driver と通信不能）で失敗した。
GPU使用の可否を検証できないため、detached job は起動していない。したがって
訓練用 `train.jsonl`、棄却例、採用率はいずれも **n=0 / 未測定** であり、量産判断は
していない。

## 凍結済み入力と汚染防止

- 500 task、**197**種類のsynthetic mock API schema（約200）、20 domain。
- strata: single / parallel / multi-turn / error recovery が各 **125** task。
- 再帰は1周だけ（生成出力を次のseed/few-shotへ再投入しない）。
- BFCL はローカル19ファイル・集計 **8,726 function** から、arity/type の集計値だけを
  構造参考にした。問題文、関数名、説明、値はseedや出力に保存していない。
- 汚染検出はローカルBFCL 20評価ファイルを対象に、正規化8-gram一致または関数名一致を
  採用前に拒否する。照合対象のSHA-256一覧は
  `esft/data/selfgen_toolcall_v1/20260711_toolcall_v1_pilot500_r3/seeds.json` と manifest に固定した。
- ACEBenchはワークスペース内に存在しなかった（`acebench_present=false`）。未照合を隠さず
  manifest に記録し、後でローカル配置された場合は検出して照合対象に含める。

## 生成・検証契約

生成時だけ、厳格なsystem promptと3-shot、best-of-4を与える。出力候補の各assistant turnは
次をすべて通過しなければ採用しない。

1. 厳格JSON解析（repairなし）とtool-call外形検証。
2. 宣言schemaの関数名、required、追加引数、型、enum検証。
3. 決定的offline mock executorの再実行一致検証。
4. taskごとに凍結した期待tool planとの一致。multi-turnは前turn receipt、error recoveryは
   `UNAVAILABLE` code を次turn引数が正しく参照することまで検証。
5. BFCL/ACEBench汚染照合。

採用時だけ `train.jsonl` に、実tool schemaと素の user/assistant/tool 会話を保存する。
生成用system prompt・few-shotは保存しない（足場の蒸留）。棄却は `rejected.jsonl` に理由を
残す。生成打切り数もsummaryの `truncation_count` に記録する。

## 実装・検証済み

- パイプライン: `esft/selfgen_toolcall_v1.py`
- detached runner: `esft/run_selfgen_toolcall_v1.sh`
- `nohup` launcher と watcher arm: `esft/launch_selfgen_toolcall_v1.sh`,
  `esft/arm_selfgen_toolcall_v1_job.py`
- CPU test: `python3 esft/tests/test_selfgen_toolcall_v1.py` → **4/4 PASS**。
- `py_compile`, `bash -n`, `git diff --check` → PASS。
- GPU preflight: `/usr/bin/python3 esft/selfgen_toolcall_v1.py preflight --run-id 20260711_toolcall_v1_pilot500_r3`
  → **BLOCKED**, `nvidia-smi` exit 9。GPU 0/1を起動していない（GPU 2も未使用）。

## 再開条件と起動方法

ドライバが復旧し、preflightがPASSしてから次だけを実行する。

```bash
set -o pipefail
esft/launch_selfgen_toolcall_v1.sh 20260711_toolcall_v1_pilot500_r3
```

これは `nohup` で起動し、PIDを `nohup.pid` に保存する。45秒の待機中に、never-reused
job ID `local-selfgen-toolcall-v1-20260711_toolcall_v1_pilot500_r3` を
`tools/job-events-mcp/jobs.json` に正確なPIDで登録してwatcherを再起動する。起動後は
`job_watch_status` が `running` を返すことを確認してから待機する。完走時はmanifestの
`status=complete` と `train.jsonl` / `rejected.jsonl` / `summary.json` を確認してからeventをACKする。

この実行の前にユーザーが明示的にGrok preflightを要求した場合は、run条件を凍結し、その
レビューを完了してから同じコマンドを使う。現時点ではユーザーからGrok実行要求はなく、
ネットワーク不可条件にも従って実行していない。

## 検分例

実モデル生成が0件のため、採用例・棄却例はまだ存在しない。GPU復旧後の500件完走時には
`summary.json` の採用率・棄却理由分布と、`train.jsonl` / `rejected.jsonl` から各patternの
代表例をこのreportへ追記し、Fableの検分前には量産しない。

## 2026-07-11: execute 待機クラッシュの修正（再起動は未実施）

`launcher.log` を再検分したところ、GPU 0/1 はともに真stock のロード完了後、各
`[selfgen gpuN] 3/250` まで到達していた。一方で親プロセスは、ワーカーが全500件の
結果を最後にまとめて送るまでの設計にもかかわらず、`output.get(timeout=60)` の60秒で
`_queue.Empty` を送出して終了していた。このため、これはGPUロード失敗でもワーカー例外でもなく、
best-of-4・最大512 token・multi-turn の初回生成時間を固定タイムアウトが下回った親側の不具合である。

`esft/selfgen_toolcall_v1.py` は次のように修正済みである。

1. 親は `get_nowait()` とワーカープロセスの生存確認による待機ループへ変更した。生存中なら待機を継続し、端末sentinelなしでワーカーが終了した場合は exit code をログして異常終了する。ワーカーからの端末進捗が300秒ない場合だけ警告を出す。
2. ワーカーは `Exception` を含む失敗時に GPU 番号・例外文字列・traceback を error sentinel として親キューへ送る。親はこれをログして例外化する。
3. 各完了taskを `generation_records_gpu0.jsonl` / `generation_records_gpu1.jsonl` に fsync してから次へ進む。同じ run directory の execute は、このチェックポイントを凍結済み `seeds.json` と照合し、完了 `seed_id` を必ずskipする。最終 `train.jsonl` / `rejected.jsonl` も原子的に置換するため、完了直前の再実行で重複追記しない。

旧実装で3件まで生成した分は終了時一括送信前で、永続成果物は残っていない。したがってこの run の再開時は **0 completed / 500 pending** から始まるが、修正版以降の中断では上記checkpointから再開する。

CPU確認（GPU起動なし）: `python3 esft/tests/test_selfgen_toolcall_v1.py` は **9/9 PASS**
（checkpoint resume、worker error sentinel、待機loopを追加）、
`python3 esft/tests/test_eval_harness.py` は **21/21 PASS**。`py_compile` と
`git diff --check` もPASS。Fableが再起動を担当するため、この修正後にこちらから
preflight・launcher・GPUジョブの起動は一切行っていない。
