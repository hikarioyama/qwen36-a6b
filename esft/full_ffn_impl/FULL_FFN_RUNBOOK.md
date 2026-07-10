# Full-FFN 設計・実施・監視 Runbook

最終更新: 2026-07-10
対象: Qwen3.6-35B-A3B → Qwen3.6-35B-A6B (serve-time top-k 32)

## 1. 現在の結論

**200-step 本番はまだ開始しない。**

真stockを使ったFull-FFN probe v2では、学習・checkpoint保存・optimizer復元・再開stepの
実行までは成功した。しかし、連続実行のstep 6とcheckpoint-5から再開したstep 6で、
model/optimizerの全local-shard digestがbit-exact一致しなかった。

一致した項目:

- step 6 loss: `0.608`
- step 6 grad norm: `0.2521`
- scheduler state
- 8 rankすべてのRNG stateファイル
- optimizer state件数: `[40, 40, 40, 40, 40, 80, 40, 40]`

不一致だった項目:

- 8 rankすべてのmodel/optimizer full-byte digest
- 対応するrank 0のmodel/optimizer DCP shard（同サイズだが内容不一致）

まず不一致の発生点を「checkpoint-5読込直後」と「同一batchのstep 6更新」に分離する。
この診断が通り、新しいprobeがGREENになることが本番開始条件である。

## 2. 固定する設計

### 2.1 学習対象

- base revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- method: `full-ffn`
- trainable: 全40層のrouted expert FFNのみ
- trainable tensor数: 80
- trainable parameter数: `32,212,254,720`
- frozen:
  - router
  - shared expert
  - attention / linear attention
  - embedding / lm_head
  - layer norm
- routing: fixed top-k 32
- 禁止:
  - `--train-router`
  - `--router-topk-random`
  - full-FFNでのAdafactor以外のoptimizer

起動時に80 tensorの完全allowlist、parameter数、router trainable 0を全rankでfail-fast検査する。

### 2.2 データとbatch

- main: `v3.jsonl`
- replay: `mixed_v2.jsonl`
- block比: main 70% / replay 30%
- 実測:
  - v3 cache: 22,238 blocks
  - mixed_v2 cache: 59,468 blocks
  - mixed replay投入: 9,531 blocks
  - 合計: 31,769 blocks
- sequence length: 7,168
- per-device batch: 1
- GPU数: 8
- gradient accumulation: 4
- global batch: 32
- seed: `5934875`
- random concat ratio: 0

### 2.3 optimizerとscheduler

- optimizer: Adafactor
- learning rate: `1e-5`
- scheduler: constant
- warmup: 0
- weight decay: `0.0`
- `scale_parameter=False`
- `relative_step=False`
- `warmup_init=False`
- `beta1=None`
- max grad norm: 1.0

FSDPの既定`no_sync`でGA4を行うと、未shardの32.2B gradientを各GPUに保持してOOMする。
Full-FFNでは必ず次を指定する。

```python
accelerator_config={
    "gradient_accumulation_kwargs": {"sync_each_batch": True}
}
```

これによりglobal batch 32を維持したまま、各microbatchでgradientをreduce/shardする。

### 2.4 FSDP

- 8-way FULL_SHARD
- auto-wrap単位: `Qwen3_5MoeDecoderLayer`
- `use_orig_params=True`
- `sync_module_states=True`
- `cpu_ram_efficient_loading=True`
- `limit_all_gathers=True`
- `backward_prefetch=backward_pre`
- checkpoint state dict: `SHARDED_STATE_DICT`

環境変数`FSDP_STATE_DICT_TYPE=SHARDED_STATE_DICT`をmodel load前に設定し、Trainer生成後にも
実値をassertする。

現在はTransformersのgradient checkpointingを使用している。FSDP native
`activation_checkpointing`への切替は、現方式で再びmemory問題が出た場合の第一候補だが、
同時に両方を有効化してはならない。

### 2.5 checkpoint

形式はHF/Accelerate標準のFSDP sharded DCPとする。

各checkpointに必要なもの:

- `pytorch_model_fsdp_0/.metadata`
- `optimizer_0/.metadata`
- `scheduler.pt`
- `trainer_state.json`
- `rng_state_0.pth` ～ `rng_state_7.pth`
- `checkpoint_complete.json`

完全性markerは保存開始時に無効化し、全rankのsaveと検査が終わった後にrank 0がatomic publishする。
markerがないcheckpointからのresumeは禁止する。

Adafactorの保存では、派生値`RMS`だけを除外する。`scale_parameter=False`ではRMSはLRに
使われず、次stepで現在のparameterから再計算される。`step`と`exp_avg_sq`は必ず保存する。

Accelerate 1.14の標準SHARDED optimizer loaderは、fresh Adafactorの空stateをload先に使い、
保存済みstateを黙って読み飛ばす。そのため標準loaderは使わず、PyTorch DCP metadataから
canonical optimizer stateを構築して、`FSDP.optim_state_dict_to_load()`経由で復元する。

実測checkpointサイズは約249 GiB。200stepごとのcheckpointを全保持する。TTDCの
`/mnt/docker-raid`はHDDではなくNVMe RAID0である。選別後の長期保管物だけをローカル
`/mnt/vault` HDDへ転送する。

## 3. 実装ファイル

- `train_esft.py`: Full-FFN trainer、freeze audit、DCP save/load、resume検査
- `run_fullffn_probe.sh`: memory・gradient・checkpoint・exact-resume probe
- `run_fullffn_main.sh`: 200step単位の本番runner
- `IMPLEMENTATION_NOTES.md`: 旧設計メモ。数値や完了状態が本runbookと矛盾する場合は本runbookを優先

TTDC配備先:

- `/mnt/docker-raid/models/esft/train_fullffn_dcp.py`
- `/mnt/docker-raid/models/esft/run_fullffn_probe_dcp.sh`
- `/mnt/docker-raid/models/esft/run_fullffn_main_dcp.sh`

liveの`/mnt/docker-raid/models/esft/train_esft.py`は上書きしない。

## 4. これから実施する作業

### Phase A: bit-exact不一致の発生点を分離

1. checkpoint-5からmodel/optimizer/schedulerをloadするだけの診断runを作る。
2. update前に各rankで次を別々にdigestする。
   - trainable model shard
   - optimizer tensor state (`exp_avg_sq`)
   - optimizer scalar state (`step`)を型ではなく正規化した数値として記録
3. checkpoint-5保存時のdigestとload直後digestを比較する。
4. 判定:
   - load直後に不一致: DCP model/optimizer復元経路またはdtype変換を修正する。
   - load直後は一致、step 6後だけ不一致: FSDP collective/reductionまたはdata/updateの非決定性を分離する。
5. lossは4桁表示で判断せず、少なくともfloat64表記またはraw bitで比較する。

特に確認する項目:

- FSDPがbf16 parameterをfp32へupcastして保存・復元していないか
- 連続側とresume側でAdafactor `step`の型が異ならないか
- model digestとoptimizer digestを混ぜず、component別に比較できているか
- checkpoint-5のlogical tensor値とload直後のlocal shard値が一致するか

### Phase B: probe v3

Phase Aの修正後、使い回さない新job IDと新output directoryでprobeを再実行する。

GREEN条件:

1. 8 rankすべて起動
2. peak allocated `<70 GiB/GPU`
3. trainable parameter `32,212,254,720`
4. exact trainable allowlist 80/80
5. router/attention/embed等がfrozen、frozen gradientなし
6. expert gradient union coverage 80/80
7. checkpoint-5とcheckpoint-6が完全
8. checkpoint-5からoptimizer state、scheduler、RNGを復元
9. checkpoint-5 load直後が保存時と一致
10. 連続step 6とresume step 6で以下が一致
    - model digest
    - optimizer digest
    - scheduler
    - 8 rank RNG
    - batch identity
    - raw loss / update結果
11. `PROBE RESULT: GREEN`を出力

1項目でも満たさなければ200stepを開始しない。

### Phase C: 200-step pilot

probe v3 GREEN後、ユーザーへ結果を説明して明示承認を得てから開始する。

固定値:

```text
TARGET_STEPS=200
SAVE_STEPS=200
EVAL_STEPS=100
SEQ=7168
WORLD_SIZE=8
PER_DEVICE_BATCH=1
GRAD_ACCUM=4
GLOBAL_BATCH=32
LR=1e-5
WEIGHT_DECAY=0.0
TOP_K=32
```

本番前チェック:

- 8 GPUに他processがいない
- 配備trainer/runnerのSHA-256がローカルと一致
- true-stock revisionが一致
- data cacheのblock数が一致
- TTDC RAIDの空き容量が十分
- jobEventsに未再利用のjob IDを追加済み
- exact PID、command substring、log、成功marker、JSON checksを設定
- watcher再起動後に`job_watch_status=running`
- start sentinel作成前にGPUが0 MiB

### Phase D: step 200判定

step 200到達だけではpromoteしない。checkpointの完全性とresume可能性を再確認し、能力評価を行う。

最低限の評価候補:

- MMLU: n=600、choice-logprob、shuffle seed 0
- GSM8K: n=600
- HumanEval: n=164、max_new=4096、truncation併記

true-stock base@k8とのsame-condition paired比較を行う。McNemar p値だけで非劣性を主張せず、
事前に定めたmarginとpaired confidence boundで判断する。

### Phase E: 追加100step

step 200評価後、続行が妥当な場合のみ次を行う。

```text
TARGET_STEPS=300
RESUME=<run>/checkpoint-200
```

`max_steps=100`ではなくglobal targetの300を指定する。300は通常のsave interval 200の倍数では
ないため、終了時に完全checkpoint-300を強制保存する。以後も同じ方式で延長する。

## 5. 監視

### 5.1 jobEvents

長時間jobは`tools/job-events-mcp/jobs.json`へ事前登録する。

必須項目:

- 一度しか使わないjob ID
- exact PID
- `cmd_contains`
- `success_not_before_epoch`
- log path
- positive success pattern
- 必須成果物一覧
- `checkpoint_complete.json`のglobal step検査

定義変更後:

```bash
systemctl --user restart qwen36-a6b-job-events-watcher.service
```

start sentinelを作る前に、MCPの`job_watch_status`で対象jobが`running`になっていることを確認する。

監視中はmodel側の手動polling loopを作らず、`wait_for_job_event`を使う。通知はmodelを勝手に
起動するものではなく、active turnのwaitへ返るか、次のuser turnまでdurable queueに残る。

2026-07-10の自己診断では、completed manifestの一時jobが約29秒でMCPへ通知された。
Full-FFN v2の実failedイベントも自動通知されたため、通知経路は動作確認済み。

### 5.2 実行時に見る値

- 8 rankのfreeze audit
- FSDP state dict type
- loss / grad norm / LR
- expert gradient union coverage
- GPU allocated / reserved / temperature
- checkpoint書込容量と`.metadata`
- marker publish
- resume時のmodel/optimizer/scheduler loadログ
- terminal event

正常な長時間無出力の例:

- 約249 GiBのDCP checkpoint書込
- 約129 GiBのoptimizer state読込
- 全local-shard digest

プロセスが生きているだけで正常と決めない。PID、log更新時刻、checkpoint容量、GPU使用、
watcher観測時刻を突き合わせる。

### 5.3 event処理

event受信後の順序:

1. ユーザーへ`[MCP jobEvents]`付きでevent ID、job ID、status、時刻、要約を通知
2. logと成果物を直接検査
3. 結果と次の判断をDEVLOGへ記録
4. その後で`ack_job_event`

ACKは「通知を見た」ではなく、「結果を処理し判断を記録した」を意味する。

## 6. 障害対応

### CUDA OOM

2026-07-10 v1実測:

- GA4 + FSDP既定no_sync
- allocated 93.19 GiB
- 使用量約94.14/94.97 GiB
- MoE backward recomputeで896 MiB追加確保に失敗

修正済み: `sync_each_batch=True`。v2 peakは最大63.59 GiB。

再発時の順序:

1. `sync_each_batch=True`が実際にAcceleratorへ反映されているか確認
2. FSDP native activation checkpointingへ切替（Transformers側gradient checkpointingとは排他）
3. `backward_prefetch`を無効化
4. 最後にsequence length短縮を検討

k低下、router変更、optimizer変更をmemory対策として無断で行わない。

### checkpoint不完全

- markerなし: resume禁止
- `.metadata`欠損: resume禁止
- 8 rank RNG欠損: resume禁止
- world size不一致: exact resume禁止
- weight decay不一致: resume禁止
- optimizer state件数不一致: resume禁止

古いmarkerが残らないよう、上書き保存開始前にmarkerを削除する。

### resume不一致

loss表示が同じでもGREENにしない。次をcomponent別に調べる。

1. checkpoint保存時
2. load直後
3. 最初のresumed batch取得後
4. backward後
5. optimizer step後

model、optimizer、RNG、scheduler、batch hashを各境界で比較し、最初に差が出た位置を原因とする。

## 7. 実測履歴

### 旧INC-0

- peak 61.8 GiB
- GA1
- true-stockではないmodel pathを使用
- checkpoint exact-resumeを検証していない

参考値に留め、現行本番のGREEN根拠には使わない。

### v1 (true-stock)

- GA4
- FSDP no_syncによるfull gradient常駐
- 初回backwardでOOM
- checkpointなし

### v2 (true-stock)

- `sync_each_batch=True`
- peak max 63.59 GiB
- Phase 1は全項目GREEN
- checkpoint-5/6作成成功
- checkpoint-5からmodel/optimizer/scheduler/RNG復元成功
- resumed step 6実行成功
- exact full-state比較だけRED

## 8. 完了条件

Full-FFN作業を「準備完了」と呼べるのは次の状態である。

- bit-exact不一致の発生点を特定・修正
- probe v3が全条件GREEN
- 200step jobEvents定義がarmed/running
- GPU、容量、SHA、true-stock identityのpreflight完了
- ユーザーが200step開始を承認

200step完走後も、評価とpaired判定が終わるまでは成果モデルをpromoteしない。

## 9. 運用上のユーザー指定

- weight decayは`0.0`
- checkpointは200stepごと
- 必要に応じてcheckpointから追加100step
- 長期保管は選別してHDDへ移す
- Grokはユーザーが明示的に指示した場合だけ使用する
