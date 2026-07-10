# G1: k=8 教師 top-64 logits への KL 蒸留 — 実装ノート

## 目的(1行)
top-k を 8→32 に上げると知識ベンチが劣化する。k=8 経路(=base)の出力分布を教師とし、
k=32 生徒(同重み + 選抜 delta)を KL で合わせる。`loss = CE + β·KL(teacher_k8 || student_k32)`。

## 成果物
- `precompute_k8_logits.py` — base(k=8)で packed cache 全 block を no_grad forward、各 token
  位置の **top-64 logits + indices** を per-GPU シャードに保存(8 プロセス、resume-safe)。
- `train_esft.py` — 本番 trainer(delta+DDP+FLCE+`--router-top-k`)に KL 経路を追加。
  `--kl-logits-dir` / `--kl-beta` / `--kl-token-chunk`。既存挙動は完全に温存。
- `run_precompute_k8.sh` — 教師 logits 事前計算(8 GPU シャード並列)。
- `run_g1_ab.sh` — 3 腕 A/B(A: CE only / B1: β0.1 / B2: β0.5)を直列実行。
- `tests/test_kl_grad.py` — CPU grad ゲート + KL 正当性(方向・非負・store round-trip)。

## 設計判断

### なぜ「事前計算した教師 logits」なのか(live 教師でない)
delta 経路は 35B 全体が **1 GPU** に載る(packed 凍結 + 小 delta)。ここに凍結 35B 教師を
同居させると 2 モデル分で VRAM が破綻する(既存 `--kl-teacher` full-ffn scaffold が
`NotImplementedError` で拒否しているのと同じ壁)。教師出力を **一度だけ** ディスクに焼けば
訓練時に教師モデルは不要 → 壁を回避。top-64 だけ保存すれば full-vocab を持たずに済む。

### FLCE × KL の部分 lm_head 計算(この実装の肝)
FLCE(Liger fused-linear-CE)は `[seq × vocab]` の full logits を materialize しない。
KL には生徒の logits が要るが、**教師の top-64 indices の位置だけ** 部分計算する:

    stu_logits = einsum("ch,ckh->ck", hidden_chunk, lm_head.weight[idx_chunk])   # (c,64)

full `[seq × 248320]` を作らず、`token_chunk × 64 × H` の transient だけで済む
(`compute_kl_term`)。教師分布 `p = softmax(top64_logits)`、生徒 `q = softmax(stu top64_logits)`
を **同じ 64-support で切断正規化** し、`KL(p‖q) = F.kl_div(log_q, p, 'sum')`(teacher-anchored)。
CE は FLCE のまま(A 腕と同一実装)、KL は別項として加算。

### CE を A 腕と bit 一致させる
`KLTrainer.compute_loss` は `super().compute_loss(...)`(=stock Trainer)で CE を取得し、
その上に KL を足すだけ。よって CE のスケーリング(grad-accum の `num_items_in_batch` 正規化含む)は
CE-only 腕と厳密一致。KL 項は「非 pad token で平均 → block で平均」の正則化項で β が scale を吸収する
(grad-accum の token 正規化は KL には未適用。β の相対値で調整する設計)。

### teacher shard と block の対応付け(random_split 耐性)
`train_ds, val_ds = random_split(dataset, ...)` で block はシャッフルされる。そこで
`IndexedTensorDataset.__getitem__(i)` が `(input_ids[i], labels[i], i)` を返す。random_split の
Subset は base index を保つので **global block index i がそのまま collator を通って compute_loss
に届く**。`TeacherLogitStore.get(i)` が `shard_{(i//chunk)*chunk}.safetensors` の行 `i%chunk` を
`safe_open.get_slice` で 1 行だけ読む(full-shard load なし、handle キャッシュ)。
教師は **同じ cache ファイル** を同順で処理して焼くので block i が一意対応。
起動時に `manifest.num_blocks == cache.blocks` と `seq_length` 一致を assert。

### eval は CE のみ
`_kl_ce_forward` は **訓練時のみ** `hidden_states` を返す(eval で返すと Trainer が hidden を
logits として eval セット全体に concat しメモリ/形状が破綻する)。KL は `model.training` 時だけ
加算 → `eval_loss` は全腕で純 CE、A/B 比較が clean。

### KL の方向と support
- 方向: `KL(teacher ‖ student)`(教師に生徒を寄せる)。`F.kl_div(input=log_q_student, target=p_teacher)`。
- 位置: **非 pad の全位置**(`input_ids != pad_id`)。`labels==-100`(prompt 部)も含む
   ── 教師合わせが目的なので supervised token に限定しない(タスク指示どおり)。
- 教師/生徒は同一位置 t の next-token 分布を直接比較(CE のような shift は KL には不要)。

## サイズ / 時間試算(v3 cache, S=7168, top_k=64)
- block 数 N ≈ 2.55GB / (7168·8·2 bytes) ≈ **~22k blocks**。
- 教師サイズ/block = `S·64·(2+4)` = 7168·64·6 = **2.75 MB** → 全体 **~60 GB**
  (logits bf16 ~20GB + indices int32 ~40GB)。docker-raid に置く(`/tmp` 禁止)。
  int32 は vocab 248320 > 65535 のため必須(uint16 不可)。
- 事前計算: 22k/8 ≈ 2.75k blocks/GPU、frozen forward のみ。おおよそ **~1–2 h**(実測でログ確認)。
- 訓練: 500 step × 3 腕。KL の追加コストは部分 lm_head(下記)+ teacher IO(~2.75MB/step)で小。
  目安 **~2 h/腕 × 3 = ~5–6 h**(coact_gate 900step 実績から外挿。実測せよ)。

## メモリ増分(94GB edge に対して)
CE-only(delta+FLCE)比の追加:
- `hidden_states` 返却: `B·S·H·2` = 1·7168·2048·2 ≈ **29 MB**。
- 部分 lm_head chunk: `token_chunk·64·H·2` = 2048·64·2048·2 ≈ **536 MB** transient(+ gather rows 同等)。
  → ピーク **~1 GB 増**。94GB edge で危ければ `--kl-token-chunk` を下げる(1024→268MB)。
- teacher shard 行: 2.75 MB(無視可)。
- **seq7168 が入らない場合の fallback**: `run_*.sh` の `SEQ` env を 5120 に。ただし教師は seq 依存
  なので `run_precompute_k8.sh` を `SEQ=5120` で焼き直し(別 out-dir)→ `run_g1_ab.sh SEQ=5120` で対応する
  `KL_DIR` を指す。cache も seq5120 版が要る(`--prepare-data-only` で先に生成)。

## 品質ゲート(達成状況)
- **β=0 bit 同一**: `kl_active = bool(kl_logits_dir) and kl_beta>0`。未指定 or β=0 では
  IndexedTensorDataset/KLTrainer/KL forward すべて非活性 → 既存 delta 経路と同一。A 腕は
  `--kl-logits-dir` を渡さないので構造的に CE-only。
- **新パラメータなし**: KL は既存 delta のみに勾配を流す。教師 logits は勾配なし定数
  (`softmax(...).detach` 相当、`F.kl_div` target)。lm_head は凍結。
- **grad ゲート(CPU 1 バッチ)**: `python tests/test_kl_grad.py` で
  `test_grad_flows_to_delta` が KL→hidden→delta へ grad>0 を実証(PASS 済、grad_norm≈0.7)。
  実モデルでの確認は下記 GRAD_PROBE 手順。

### 実モデルでの grad ゲート手順(gpu-host、発射前)
```
GRAD_PROBE=1 <venv>/bin/python -m torch.distributed.run --nproc_per_node=8 train_esft.py \
  --model <MODEL> --method delta --expert-config configs/mixed_v1_token_k32_p0.2.json \
  --train-data /mnt/docker-raid/models/esft/v3.jsonl \
  --data-cache-dir /mnt/docker-raid/models/esft/cache \
  --output-dir runs/g1_probe --router-top-k 32 --seq-length 7168 --fused-ce \
  --optimizer adafactor --grad-accum 2 --max-steps 2 --eval-steps 100 --save-steps 100 \
  --kl-logits-dir <KL_DIR> --kl-beta 0.1
```
`[GRAD_PROBE] expert_gnorm=...(n>0)` が非ゼロ、`frozen[embed].grad=None` を確認。
β0 と β>0 で 1 step 目の loss を比べ、β>0 の方が大きい(KL 加算)ことも確認。

## deploy 手順(gpu-host)
train_esft.py は `/mnt/docker-raid/models/esft/`(esft_qwen パッケージがある dir)で走る前提。
```
scp train_esft.py precompute_k8_logits.py run_precompute_k8.sh run_g1_ab.sh \
    gpu-host:/mnt/docker-raid/models/esft/
scp -r tests gpu-host:/mnt/docker-raid/models/esft/tests_g1   # 任意
# 1) 教師 logits 焼き(8 GPU 空き確認後)
ssh gpu-host 'cd /mnt/docker-raid/models/esft && bash run_precompute_k8.sh'
# 2) grad ゲート(上記 GRAD_PROBE)→ GREEN なら
# 3) 3 腕 A/B
ssh gpu-host 'cd /mnt/docker-raid/models/esft && bash run_g1_ab.sh'
```
`VENV` は両スクリプトで `~/esft-work/venv/bin/python`(無ければ PATH の python)。
`CONFIG`/`KL_DIR`/`SEQ`/`STEPS`/`KL_TOKEN_CHUNK` は env で上書き可。

## eval(G1 判定)
各腕の best-eval delta patch を既存 `eval_harness.py`(`--topk 32`)で知識ベンチ評価し、
A(CE only)vs B1/B2 の知識スコアを **same-condition** で比較。KL が知識劣化を埋めるかを見る。
```
```
