# full-FFN FSDP trainer — implementation notes

全 routed expert FFN(32.2B)を gpu-host 8×96GB で FSDP full-shard 訓練する `--method full-ffn`
経路。計画 `clever-conjuring-clarke.md` Phase 1a-1e / 2a に対応。

## 成果物

| file | 役割 |
|---|---|
| `esft_patch_to_esft_full.py` | `esft_qwen/esft_patch.py` に **追記** する `to_esft_full()`(paste-ready、既存シンボル無改変) |
| `train_esft.py` | 本番 `train_esft.py`(gpu-host)の **完全置換版**。`--method full-ffn` 分岐を追加、delta/maskhook 経路は無改変 |
| `run_fullffn_probe.sh` | INC-0 memory probe(8-GPU max-steps 5、4 assert を自動判定) |
| `IMPLEMENTATION_NOTES.md` | 本ファイル |

`train_esft.py` は gpu-host 本番版(delta+DDP+cache+FLCE+router-mobile 済)を土台に、
`--method full-ffn` を **純追加分岐** として実装。delta/maskhook のコードパスは 1 行も
挙動変更していない(full-ffn 固有処理はすべて `is_full_ffn` ガード下)。

## deploy 手順(主セッションが実行)

1. `to_esft_full` を `esft_qwen/esft_patch.py` の `enable_router_training` の直後・
   `snapshot_router_weights` の直前に貼る(`esft_patch_to_esft_full.py` の中身)。純追加。
2. 本番 `train_esft.py` を本 repo の `train_esft.py` で置換
   (`diff` 推奨。差分は下記「変更点」の region のみ)。
3. probe を回す前提: `mixed_v2.jsonl`(または v3)の cache が `--seq-length 7168` で存在
   or `--prepare-data-only` で生成(probe step 1 が自動でやる)。
4. `run_fullffn_probe.sh` の env(`ESFT_DIR/VENV/MODEL/TRAIN_DATA`)を gpu-host 実パスに合わせて起動。
   **GPU 発射は主セッション。本実装エージェントは発射しない。**

### 変更点(train_esft.py、region 単位)

- docstring: 3rd method `full-ffn` の説明を追記。
- 新 CLI: `--method` choices に `full-ffn`、`--replay-data/--replay-ratio`、
  `--kl-teacher/--kl-beta`(すべて default off)。
- `is_full_ffn` 判定 + **model load 前** に `ACCELERATE_USE_FSDP=true` /
  `FSDP_CPU_RAM_EFFICIENT_LOADING=1` を export(cpu_ram_efficient_loading の前提)。
- full-ffn 時のみ `torch.distributed.init_process_group` を model load 前に初期化
  (`is_fsdp_enabled()` が PG 初期化を要求するため)。
- data: `mix_replay_blocks()` + `build_or_load_packed(..., data_path=)` 拡張(replay 混合)。
- `device_map`: full-ffn は `None`(FSDP shard)、delta/maskhook は従来 `{"":idx}`。
  load kwargs に `low_cpu_mem_usage=True`(full-ffn)。
- `handles`: `elif args.method == "full-ffn": handles = to_esft_full(model)`。
- optimizer: full-ffn は **自前 optimizer を作らない**(`optimizer=None`)。
  Trainer が FSDP wrap 後に `--optim` から構築。
- `TrainingArguments`: full-ffn 時のみ `fsdp="full_shard auto_wrap"` +
  `fsdp_config`(下記)+ `optim`(adafactor→"adafactor" / adamw→"adamw_torch")。
- Trainer 構築: `optimizers=` は optimizer 非 None のときだけ渡す。
- `FULLFFN_PROBE` callback + 訓練後 frozen-grad assert。
- 最終保存: full-ffn は `FULL_STATE_DICT`(offload_to_cpu, rank0_only)gather →
  `save_pretrained`。**この gather は collective なので全 rank で実行**(early-return より前)。

## 技術判断と根拠

### FSDP wrap はデコーダ層単位(experts 単独 wrap 不可)
`transformer_layer_cls_to_wrap=["Qwen3_5MoeDecoderLayer"]`。
実コード(`modeling_qwen3_5_moe.py`)確認済:
- decoder 層クラス = `Qwen3_5MoeDecoderLayer`(`GradientCheckpointingLayer` 継承、L826)。
- `Qwen3_5MoeSparseMoeBlock.forward` は 1 forward 内で `gate_up_proj` と `down_proj` の
  両方を使う。experts を単独 FSDP unit にすると、層 forward の途中で 2 packed tensor が
  同時に unshard されている保証が崩れる。デコーダ層で wrap すれば層 forward 全体で両方が
  all-gather 済みになり、expert matmul が unsharded weight を見る。
- packed 3D Parameter(dim0=expert 軸、per-expert module ではない)を前提。全 expert 訓練
  なので per-expert mask 不要 = flat-shard で問題なし。

### `use_orig_params=True` 必須(凍結混在)
1 デコーダ層内に **trainable(experts packed 2本)と frozen(router.weight,
shared_expert, shared_expert_gate, attn, 2×RMSNorm)が混在**。FSDP の FlatParameter は
1 flat 内で `requires_grad` 混在を許さない。`use_orig_params=True` で per-Parameter の
`requires_grad` を保持し、frozen は grad/opt state を持たず、trainable(experts)だけ
optimizer に載る。

### optimizer は Trainer に委譲(自前で作らない)
自前 optimizer を `optimizers=` で渡すと **FSDP wrap 前の unsharded param** を掴んで壊れる。
full-ffn は `--optim adafactor`(TrainingArguments)で Trainer が wrap 後に構築。
weight_decay は trainable(experts)のみ(他は frozen)。

### cpu_ram_efficient_loading + sync_module_states
`ACCELERATE_USE_FSDP=true` を **from_pretrained 前** に export し、rank>0 は meta device
ロード、rank0 だけ実 weight を materialize → FSDP が shard を scatter。host RAM peak を
~70GB(1コピー)に抑える。gpu-host は 1492GB free なので余裕だが、複数 rank の full copy
(8×70=560GB)を避けるのは行儀として有効。

## メモリ試算(per-GPU、8-way FULL_SHARD)

| 項 | 式 | /GPU |
|---|---|---|
| trainable param(bf16 shard) | 32.2B×2B / 8 | 8.05 GB |
| grad(bf16 shard) | 32.2B×2B / 8 | 8.05 GB |
| Adafactor state ⚠ | 後述 fp32 non-factored / 8 | ~16.1 GB |
| frozen 非expert(bf16 shard) | ~3B×2B / 8 | ~0.75 GB |
| activations(seq7168, grad-ckpt, FLCE) | 概算 | ~5–10 GB |
| all-gather buffer(1 デコーダ層分 experts) | 805M×2B | ~1.6 GB |
| **合計** | | **~40–45 GB** |

target `<70GB/GPU` に対し余裕。plan の 38–42GB 試算に **Adafactor state 16GB** を足した値。

- 32.2B = 40層 × 256 expert × 3.146M(gate_up 1024×2048 + down 2048×512)。
- bf16 param 64.4GB + grad 64.4GB。**DDP 全複製は不可**(128GB > 96GB)→ FSDP 必須。
- **⚠ Adafactor state**: FSDP flat shard は 1D。Adafactor は ndim<2 の param を
  **non-factored**(fp32 full 2次モーメント)で持つ → shard numel(4.03B)×4B = 16.1GB/GPU。
  「Adafactor だから state ~0」は **FSDP 下では成立しない**。それでも AdamW の
  +258GB(fp32 m+v full)よりは遥かに軽く、8-way でも AdamW は
  32.2B×8B/8=32GB/GPU で experts のみでも重い。Adafactor 継続が妥当。
- probe の ASSERT1 が実測(`torch.cuda.max_memory_allocated`)。もし >70GB なら逃げ順:
  grad-accum↑ → seq 短縮 → `reduce_dtype=bf16`(mixed-precision の grad reduce)→ CPUOffload。

## 保存形式と DCP fallback

- **途中 checkpoint**: `fsdp_config.state_dict_type="SHARDED_STATE_DICT"`(各 rank が自分の
  shard を書く。70GB rank0 gather を run 中に走らせない)。
  ※ HF/accelerate のバージョンがこのキーを拒否したら **落として** よい(Trainer 既定の
  FULL_STATE_DICT でも動く。ただし checkpoint 毎に重い gather)。
- **最終**: `FSDP.state_dict_type(FULL_STATE_DICT, offload_to_cpu=True, rank0_only=True)`
  で gather → `unwrapped.save_pretrained(state_dict=cpu_state)`。**普通の HF model dir** が
  出るので eval は無改造(`--model base --model-path <dir> --topk 32`)。
- **collective 注意**: `state_dict()` は all-gather collective。**全 rank が入る**必要があり、
  `if not should_save: return` より **前** に置いた(でないと rank0 が gather で無限待ち
  = deadlock)。rank0 のみ save、他 rank は空 dict で通過。
- **DCP fallback**(host RAM が 70GB gather を許さない場合): 最終も SHARDED のまま残し、
  別の大 RAM ノードで `torch.distributed.checkpoint`(DCP)+
  `dcp_to_torch_save` / `FSDP.state_dict_type(FULL)` off-line consolidation → save_pretrained。
  gpu-host は 1.5TB free なので通常不要。

## 忘却フック(構造のみ、既定 off)

- **replay(実装済・動作)**: `--replay-data <jsonl> --replay-ratio r`。
  main を pack 後、replay を同じ pack 経路(cache 共有)で pack し、block 数で比率 r になるよう
  seeded sample して concat → random_split。cheap で正しい。既定 r=0 で無効。
- **KL teacher(scaffold only、有効化すると `NotImplementedError`)**: `--kl-teacher <ckpt>
  --kl-beta b`。目的は frozen base topk=8 分布への self-distillation(CE+β·KL、FLCE 無効化)。
  **未実装の理由(honest)**: frozen 35B teacher を 32B-trainable FSDP shard と 8×96GB に
  同居させるには **teacher 自身の sharding(TP or 別 FSDP replica)か外部 teacher server**
  が要る。untested な重い経路を黙って回すと OOM/誤り。loss 合成の設計は残すが、有効化は
  明示エラーで止める。**必要になったら別途 teacher-serving 設計 + 専用 probe**。
  - 代替(cheaper だが仕様と別物): router-mobile の base-routing anchor
    (`--train-router --router-anchor-weight`)が既に「k=8 分布を 1 forward で pin」する
    正則化を提供している。co-adaptation 較正の軽い代用として先に試す価値あり。

## probe(INC-0)の 4 assert

`run_fullffn_probe.sh`(8-GPU、max-steps 5、FULLFFN_PROBE=1):
1. **peak < 70GB/GPU**: `max_memory_allocated` 行を全 rank で parse、max<70。
2. **trainable == 32.2B**: `ESFT trainable params (full-ffn): N` を parse、32.0–32.5e9。
3. **全 40 層 expert grad>0**: FULLFFN_PROBE の `grad_none==0 && grad_zero==0` かつ
   covered≥80(40層×2 packed)。
4. **frozen 不変**: (A, in-run)frozen param の `.grad` が全 None。
   (B, offline)保存 model dir の `*.mlp.gate.weight` を base と byte 比較(drift==0)。
   router は requires_grad=False かつ optimizer 非搭載なので B は構造的に保証されるが、
   実測で裏取り。

全緑で `PROBE RESULT: GREEN (GO)`、非緑なら exit 1。

## リスク / 未確定(probe が答える)

- ⚠ **Adafactor state 16GB/GPU** の実測(試算。ASSERT1 が確定させる)。
- HF/accelerate バージョン依存: `fsdp_config` のキー受理(特に `state_dict_type`,
  `cpu_ram_efficient_loading`, `backward_prefetch`)。拒否時は該当キーを落とす。
- gradient_checkpointing は TrainingArguments 側(`use_reentrant=False`)で有効化。
  FSDP の `activation_checkpointing` は **併用しない**(二重 wrap 回避)。
- FLCE forward swap(`model.model.language_model`)は multimodal
  `Qwen3_5MoeForConditionalGeneration` 前提(既存踏襲)。text-only を積む場合は
  backbone パスの見直しが要る。
- `--router-top-k 32` は full-ffn でも維持(rank9-32 expert に train 時 token=grad を流す)。

## deploy 時に主セッションが確認すべき点

1. **diff の当て先**: `to_esft_full` は `esft_patch.py` へ純追加(既存関数と衝突しない位置)。
   `train_esft.py` は完全置換 or region diff。delta/maskhook 経路の byte 不変を diff で確認。
2. **想定メモリ**: per-GPU ~40–45GB(<70 target)。**Adafactor 16GB が効く**点を意識。
   ASSERT1 実測が試算と乖離したら逃げ順(grad-accum→seq→reduce_dtype→CPUOffload)。
3. **probe を必ず先に**(INC-0)。GREEN 後に実技試験ペア完走を待って本走。
4. env バージョン: `fsdp_config` キー拒否が出たら該当キー削除で degrade 可(挙動は保つ)。
5. KL teacher は今は使えない(scaffold)。忘却対策は当面 replay + router-anchor で。
