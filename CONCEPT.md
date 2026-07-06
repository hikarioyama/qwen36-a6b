# DSv4-Flash 静的 K 拡張 + DoRA 適応 — 構想書 v1

**日付**: 2026-07-02
**目標**: K=6 (13.55B active) → K=13 (20.6B active) に静的拡張し、Qwen3.6-27B (dense) を明確に上回る
**状態**: 構想固め完了(文献調査 3 系統 + 実測アーキ解析済み)。INC-0 は GPU 解放待ち。

---

## 1. 実測で確定したアーキテクチャ事実 (measured, config + 全46 shard ヘッダ解析)

- 43層 全 MoE、256 routed + 1 shared/層、expert = w1/w3 [2048×4096] + w2 [4096×2048] = **25.166M params/expert**
- hash 層 = layer 0-2(gate に `tid2eid [129280, 6]` — **幅6の静的テーブル、K 拡張は構造的に不可能**)
- 学習ゲート層 = layer 3-42 の **40層**(gate.weight [256,4096] + gate.bias[256] = noaux_tc バイアス)
- router: `noaux_tc` + `sqrtsoftplus` + `norm_topk_prob=True` + `routed_scaling_factor=1.5`
- MTP module (mtp.0) も 256 expert の MoE 層を持つ(K 変更時は accept 率再測定必須)
- 総パラメータ ~285B、公式リリースが **expert_dtype=fp4 ネイティブ**(FP4 が master copy)、attn は FP8

### active params の式(比例計算は誤り)

```
active(K) = 13.55B + (K−6) × 1.007B     (ΔK 1本 = 25.17M × 40層)
```

| 目標 | K | 備考 |
|---|---|---|
| 17.6B | 10 | 旧案「K=10で20B」は不達 |
| **20.6B** | **13** | 妥協ライン 20B の正解 |
| 26B (真の倍) | 19 | |
| 40B (当初案) | 32 | |

### 設計上の必然: rank-conditional stop-gradient

「新規参入 expert(rank 7-13)だけ DoRA、top-6 常連は触らない」の**静的な expert 分割は存在しない**
(全 256 expert がトークンにより rank 1-6 にも 7-13 にも入る)。正しい実装:

- MoE combine `Σ wᵢ·Eᵢ(x)` で **rank 1-6 の寄与を detach**、rank 7-13 の寄与 + router のみに勾配
- DoRA adapter は原理上 全 256 expert に必要 → r=16 で trainable ~4.5B / optimizer ~36GB
- 絞り込み: 対象ドメインで rank 7-13 に高頻度で入る expert を INC-0 の routing ヒストグラムで特定し
  上位だけに adapter(文献裏付け: expert-level capacity mismatch, DR-LoRA 2601.04823)

---

## 2. 文献調査の結論 (3 系統, 2026-07-02)

### 2a. 無訓練 K 拡張は「崩壊しない、が利得もない」 (lit-inference-k)

- **最強アンカー: Elastic MoE (arXiv:2509.21892)** — DeepSeekV2-Lite(同系 fine-grained・同 renorm・訓練 K=6)の
  無訓練 K 倍増 6+2→12+2 で平均 **−0.13pt = 実質フラット** (measured)
- Matryoshka MoE (2509.26520): **K 増は無害、K 減は破滅**(MMLU 54→36)。増方向は安全側
- `norm_topk_prob=True` は K 不変の出力スケールを保つ = **K 拡張耐性の最良条件に既に乗っている**
- EMoE の訓練後は単調改善: V2-Lite 6+2: 53.20 → 12+2: **53.81**。適応訓練の uplift アンカー = **+0.6〜+2.7pt 平均**
- **文献にない穴(INC-0 で実測)**: `routed_scaling_factor=1.5`、`noaux_tc`+`sqrtsoftplus` の K 拡張時挙動、
  2.17 倍(2.0 超)への外挿、256-expert 級での直接測定

### 2b. トークン予算: 前例なし、最大リスクは「anti-selected expert」 (lit-adaptation-budget)

- 「固定 expert 集合で K だけ後から上げる」を測った論文は**空白**。全 upcycling 系は新規容量 + full training
- full-training の信号床: NVIDIA upcycling ablation で **0.1T (100B) tokens**(MMLU +0.9 iso-FLOP)。
  PEFT はこれより下がるはずだが「どれだけ下がるか」は誰も測っていない
- **距離2(最重要リスク)**: rank 7-13 は router が「選ばない」ことを最適化された **anti-selected expert**。
  前例の「役立つよう init された新 expert」と性質が逆。有用化に想定超のトークンが要る可能性 → pilot で最初に殺すべき仮説
- router 凍結は安定を買えない(Mechanistic Forgetting 2601.18699)— DoRA で軽く動かす方針は妥当、汎用回帰の監視必須
- 予算の方向感: 0.5B = go/no-go 信号検出 / 2-3B = PEFT 収束帯で net± 判定 / 5-20B = repurposing が本格に要るなら 20B は天井でなく床

### 2c. 競合案: ESFT (arXiv:2407.01906, DeepSeek 純正) — ROI では既定候補

- task 関連 5-15% expert のみ訓練、**~10^6-10^7 tokens** で専門 +16.6pt、汎用維持(FFT 超え)、K 拡張なし
- **目的がタスク品質(コーディング/数学/CTF)なら ESFT が確実な小勝ち**。
  K 拡張が勝てるのは「汎用能力を active FLOPs 増で底上げしたい」場合のみで、その単価は
  upcycling 実測で MMLU +0.9〜+2.3 / 100B-1T tokens と悪い

### 2d. 訓練インフラとコスト分母 (lit-peft-mfu)

- **NVFP4 frozen base 上の DoRA 既製スタックは存在しない** → FP4→BF16 dequant 訓練 or bitsandbytes NF4 + QDoRA。dequant 税 10-20%
- 8×H100 640GB 単一ノードで余裕(FP4 base ~160GB + adapter/optimizer ~40GB + 活性)、NVLink 内 EP で通信律速なし
- frozen-base の FLOPs = **4NT**(fwd 2NT + input-grad 2NT、weight-grad 不要)裏取り済み。grad-ckpt で ~6NT 相当に増える点は MFU に織込み
- MFU アンカー: DeepSeek-V3 極限最適化 = BF16 換算 ~43% / Megatron 標準 MoE ~46%。
  借り物スタックの現実線 = **15〜35%、中央 25% → 8×H100 で ~24,000 tok/s (estimated, n=0)**

---

## 3. コスト梯子 (さくら高火力DOK 8×H100 ¥2,988/h, N_act=20.6B, 4NT)

| トークン | MFU 35% | MFU 25% (中央) | MFU 15% |
|---|---|---|---|
| 0.5B (pilot) | ¥12k / 4.1h | **¥17k / 5.8h** | ¥29k / 9.6h |
| 3B (small) | ¥74k / 25h | **¥104k / 35h** | ¥173k / 58h |
| 5B | ¥124k / 41h | **¥173k / 58h** | ¥288k / 96h |
| 20B (full) | ¥494k / 165h | **¥692k / 232h** | ¥1.15M / 386h |

全て estimated (n=0)。**0.5B pilot が MFU 実測装置を兼ねる**(¥17k で分母が n=1 で確定)。
DOK 運用ゲート: 初回 1h (¥3k) で実 tok/s を測ってから延長判断。

---

## 4. 意思決定の階段 (INC gates)

### INC-0: 無訓練 K sweep — ¥0, ローカル, GPU 解放後, 半日〜1日
`vllm --hf-overrides '{"num_experts_per_tok": N}'` で K=6/8/10/13(hash 層が global K を読むなら patch 1枚)。
MTP off で clean A/B。取るもの:
1. naive K↑ 品質曲線(文献予測: −0.5〜+0.5pt。外れたら sqrtsoftplus/scaling_factor 固有項が犯人)
2. **rank 7-13 の renorm 後 gate 質量**(このレバーの物理上限。質量 X% なら naive 上限も X% 程度)
3. routing 頻度ヒストグラム → DoRA 対象 expert の選定
4. **Qwen3.6-27B との現在ギャップ実測**(このプロジェクト全体の EV を決める最重要数字)
5. K=13 の serve 速度実測(推定 150→~110 t/s)

**Kill 条件**: Qwen ギャップ >> +2.7pt(EMoE uplift アンカー上限)なら K 拡張単体では届かない → ESFT 路線 or 併用へ転換

### INC-1: V2-Lite 手法検証 — ¥0, ローカル
- rank-conditional stop-grad DoRA パイプラインを V2-Lite (15.7B, BF16 31GB, GPU 1枚) で構築
- **EMoE が同モデルで 53.20→53.81 を出している = 直接比較可能なベースラインが存在する**
- 通過条件: 無訓練フラット (−0.13) を有意に上回る uplift を n≥2 seed で再現

### INC-2: 0.5B pilot on DOK — ¥17k (中央推定)
- 見るもの: (a) held-out loss 低下 (b) router が rank 7-13 に質量を配り直せるか (c) 汎用ベンチ非崩壊 (d) MFU 実測
- **Kill 条件**: anti-selected expert 仮説が黒(loss 改善なし / router 質量が動かない)なら即撤退

### INC-3: 2-3B small — ¥104-173k
- K=13 net± の本判定。Qwen ギャップに対する到達見込みで full の go/no-go

### Full: 5-20B — ¥173k-1.15M
- INC-3 の傾きが要求する場合のみ

---

## 5. 冷徹な EV サマリ

- **勝ち筋の形**: naive K↑ はフラット(文献直接測定)→ **利得は 100% DoRA 適応が生む**。そのアンカーは EMoE +0.6〜+2.7pt
- **最大リスク**: anti-selected expert の repurposing コスト(文献空白、INC-2 で最初に殺す)
- **競合案 ESFT**: タスク品質目的なら ~10^7 tokens でほぼ無料・+16.6pt・低リスク。
  **「なぜ ESFT でなく K 拡張か」に答えられる目的定義(汎用底上げ)が go の前提条件**
- 併用形もある: K=13 + 「rank 7-13 高頻度 expert への ESFT 的集中訓練」= 本計画の histogram 絞り込み版と実質同型

## 6. 未決事項

- [ ] 目的の確定: タスク品質(→ESFT 優位)か汎用容量(→K 拡張)か
- [ ] INC-0 実行(GPU 解放待ち。訓練中は触らない)
- [ ] hf-overrides の通り道確認(vLLM DSv4 modeling、hash 層への波及)— コード読みだけなら今できる
- [ ] 訓練データレシピ(コーディング/数学/CTF、拒否句フィルタ)の具体化は INC-1 通過後
