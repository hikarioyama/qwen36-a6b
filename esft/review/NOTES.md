# ESFT → Qwen3.6-35B-A3B 移植ノート (Phase 0)

調査で確定した事実と、移植で必要になった設計判断を記録する。数値は測定条件付き。

## 環境

- venv: `~/quant-env` (system と同系だが独立)。`transformers==5.7.0`, `torch==2.10.0+cu128`。
  - `transformers.models.qwen3_5_moe` を含む(system 5.8.0 / `~/vllm-env` 5.7.0 も可)。
- tokenizer: ローカルスナップショット `models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/*` から取得
  (完全な `tokenizer.json` + `chat_template.jinja` を同梱。DL 不要)。素の `models--Qwen--Qwen3.6-35B-A3B`
  は `config.json` のみ。
- 全作業 CPU、`CUDA_VISIBLE_DEVICES=""`。実モデルの構造検証は **meta device**(メモリ非確保)で実施。

## Qwen3_5Moe の実クラス / 属性パス / routing 数式

モデルクラス: `Qwen3_5MoeForConditionalGeneration`(multimodal)。`AutoModelForImageTextToText` で解決。
テキスト実体は `text_config`。routed-MoE 層は全 40 層(`num_hidden_layers=40`、MTP は
`_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]` で load 時に無視され、モジュール木には現れない)。

モジュール経路(meta device で実測):

```
model.model.language_model.layers.{0..39}.mlp   -> Qwen3_5MoeSparseMoeBlock
    .gate            -> Qwen3_5MoeTopKRouter   (Parameter weight (256,2048)、属性 top_k=8)
    .experts         -> Qwen3_5MoeExperts      (packed 3D: gate_up_proj (256,1024,2048), down_proj (256,2048,512))
    .shared_expert   -> Qwen3_5MoeMLP          (intermediate 512)
    .shared_expert_gate -> Linear(2048,1)
```

- `gate_up_proj` shape = `(num_experts, 2*moe_intermediate, hidden)` = (256, 2*512, 2048)。
- `down_proj`    shape = `(num_experts, hidden, moe_intermediate)` = (256, 2048, 512)。
- **expert 軸は dim 0**(gradient マスクもこの軸)。
- テキスト専用 `Qwen3_5MoeForCausalLM` の場合は経路が `model.model.layers.{i}.mlp`。
  → 属性ハードコードでなく **クラス名 `*SparseMoeBlock` でモジュール走査**して両対応(`esft_qwen/common.py:find_moe_blocks`)。

### routing 数式(`Qwen3_5MoeTopKRouter.forward`、正確な順序)

```python
router_logits = F.linear(hidden, weight)              # (N, 256)
router_probs  = softmax(router_logits, dtype=float32, dim=-1)  # 全 256 で softmax が先
top_val, top_idx = topk(router_probs, top_k, dim=-1)  # その後 top-k
top_val = top_val / top_val.sum(-1, keepdim=True)     # top-k 内で再正規化(無条件)
return router_logits, router_scores(=top_val), router_indices(=top_idx)
```

- **softmax → top-k → 再正規化** の順。標準 softmax router(`router_aux_loss_coef=0.001`)。
- **`norm_topk_prob` は config フラグではない**。再正規化行は無条件に走る(= 常時 ON)。
  スモークテスト `a.top_k weights sum to 1` で実測確認済み。
- 統計収集は `.gate` に forward hook を張り、返り値の `(scores, indices)` をそのまま使う
  = **実モジュールを通すので数式は完全再現**。`--top-k N` は `config.num_experts_per_tok` を
  load 前に上書きして反映。

## ESFT 凍結ロジックの移植(最重要の設計判断)

上流 ESFT(DeepSeek-V2-Lite)は expert が**個別モジュール** `mlp.experts[i].gate_proj` なので、
非訓練 expert を buffer 化(`to_buffer`)して optimizer から外す。

Qwen3_5Moe は expert が**2 本の packed 3D Parameter**。`requires_grad` は Parameter 単位でしか
効かず、subset expert だけ凍結できない。よって:

1. 全パラメータ凍結 → 選択 expert を含む層の `gate_up_proj`/`down_proj` のみ `requires_grad=True`。
2. その packed Parameter に **gradient hook** を張り、非選択 expert の行(dim 0)の勾配をゼロ化。
3. optimizer で packed expert 群を **weight_decay=0 グループ**に入れる。

### なぜ (3) が必須か — weight-decay ドリフト

`base.yaml` は `weight_decay=0.1`。decoupled AdamW は勾配に関係なく全要素に `p -= lr*wd*p` を適用する。
packed テンソル全体を `requires_grad=True` にすると、勾配ゼロの非選択 expert も wd で 0 に向かって
ドリフトしてしまう(packed 化が生む固有の罠)。

→ **grad マスク(非選択の勾配=0)+ expert 群 wd=0** で、非選択 expert は Adam の 1次/2次モーメントも
0 のまま留まり、**ビット不変**に凍結される。スモークテスト `e.non-selected expert row bit-exact frozen`
(実 AdamW を 3 step 回して確認)で実証。

- 上流との差分: 訓練対象 expert への wd 0.1 は落とす(非選択凍結の不変性を優先)。軽い正則化の喪失のみ。
  代替案「step 後に非選択行を復元」は全 expert のコピー保持=MoE 本体丸ごとの追加メモリで却下。
- router / shared_expert_gate / attn / embed / norm は凍結(ESFT 仕様どおり、FFN expert のみ訓練)。
  `shared_experts`/`non_expert_modules` フラグで opt-in 可。

## patch 保存形式

選択 expert のスライスのみを safetensors 化(ESFT が訓練 expert モジュールだけ保存するのと等価)。
キー `layers.{L}.experts.{gate_up|down}.{E}`、metadata に `expert_config` を JSON 埋め込み(自己記述)。
load は packed テンソルへ in-place 書き戻し。roundtrip はスモークテスト `d` で検証。

## scoring(ESFT reference と一致)

- `gate_score[l,e]  += token が e に与えた routing weight`(affinity)
- `token_score[l,e] += 1/top_k`(e に routed された token ごと)
- 総 token 数で正規化 → 各行 expert 方向に和≈1(gate: 各 token weight は 1 に正規化済 / token: 各 token が
  top_k×(1/top_k)=1 を寄与)。スモークテスト `b` で和=1 を確認。
- 選択: score 降順で累積が `top_p` に達するまで(上流と同じく **追加前に current>=top_p を判定**)。
  ESFT-Token=(token, p=0.2)、ESFT-Gate=(gate, p=0.1)。

## データ

- profiling: 各ドメイン 32 blocks × 4096 tok = 131072 tok(chat template 適用、packed)。
- train jsonl: raw messages を再整形(tokenize は train 時)。math 394,996 / coding 111,272 / japanese 165,175 records。
- **tokenizer の罠**: この multimodal processor の `apply_chat_template(tokenize=True)` は
  非標準の BatchEncoding(`input_ids` に fast-tokenizer の `Encoding` を内包)を返し、素朴に列挙すると
  2 要素に化ける。回避: `tokenize=False` で文字列化 → `tok(text, add_special_tokens=False)`
  (chat template が special token を既に含むため `add_special_tokens=False`)。

## 検証済み / GPU 待ち

- CPU で実証済み: routing 数式再現、hook 捕捉、top-p 選択、scoring 正規化、凍結の bit 不変性、
  patch roundtrip、overlap 行列、実 35B のモジュール木検出(meta device)。→ `tests/test_smoke.py` 22/22 green。
- GPU 解放後のみ検証可: 実重みでの forward による実 routing 統計、実訓練の収束、
  patch を焼いた後の下流評価。
