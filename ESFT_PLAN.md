# 路線B: ESFT for DSv4-Flash — 実装計画 v1

**日付**: 2026-07-02
**出典**: arXiv:2407.01906 (EMNLP2024) + github.com/deepseek-ai/ESFT (MIT) + aws-samples/sample-ESFT + DES-MoE (2509.16882) の一次情報深掘り
**位置づけ**: K 拡張案(CONCEPT.md)の競合/先行路線。タスク品質(コーディング/数学/CTF)目的ならこちらが既定

---

## 1. ESFT の正確な仕様(paper/repo 一次情報)

### 選択基準(2種)
- **ESFT-Token**: `r_i = 平均( [expert i が top-K に入った] / K )` — routing 頻度。**p=0.2** で累積選択
- **ESFT-Gate**: `g_i = 平均 gate weight` — affinity 平均。**p=0.1** で累積選択
- 層ごとに score 降順で累積 ≥ p まで選択。V2-Lite 実績: 層あたり 2-4/64 個(3-6%)

### 凍結範囲(esft.py `to_esft` 確認済み)
- **訓練**: 選択 routed expert の FFN 全重みのみ(**full weight、adapter でない**)
- **凍結**: 非選択 expert / shared expert / **router・gate** / attn / embed / norm / lm_head(config flag で shared・非expert も開けられるが公開 config は全 false)

### 実数
- **profiling は 32 samples × 4096 tok = 131k トークンだけ**
- 訓練: LR 1e-5 / batch 32 / seq 4096 / **≤500 step**(100 step ごと eval で best 採用)→ 上限 ~6.5e7 tokens
- V2-Lite 結果: 専門 33.6→50.2(FFT 51.0 に肉薄、LoRA 44.9 超え)、**general 60.6-61.5 維持**(FFT は -3.6)、trained params 450M-1.85B、ストレージ 2.6-3.2GB

## 2. V4-Flash 移植の設計判断

1. **ESFT-Token を第一候補にする**。noaux_tc は選択(bias 込み)と gate weight(bias なし sqrtsoftplus + norm)を分離するが、Token 基準は「実際の top-k 結果」を数えるだけなので scoring 関数に非依存で clean。ESFT-Gate を使うなら p=0.1 は分布形状が違うため再チューニング必須
2. **要書き換え 3 点**: (i) V4-Flash modeling に topk_idx/topk_weight ロギング hook(stock transformers に `log_expert_weights` は無い、DeepSeek 同梱 modeling の patch)、(ii) to_esft のクラス名/属性パス(DeepseekV2MoE → V4-Flash MoE、.experts/.shared_experts)、(iii) TOP_K=6
3. **fp4 特有の配線**: 選択 expert は BF16 master で訓練、他は fp4 凍結のまま。訓練後、選択 expert だけ NVFP4 再量子化(llm-compressor)して既存 serve スタックへ。**往復品質税は要 A/B**
4. hash 層 (0-2) は routing 固定なので除外(40 学習ゲート層のみ対象)
5. AWS sample-ESFT が Qwen3-MoE 128-expert fine-grained で素の top-p 選択が動く実例(ms-swift 統合、**TP 非対応・EP/DP/PP のみ** — 単ノード 8×H100 なら EP で問題なし)

## 3. GO/NO-GO ゲート(これが全て)

**ゲート A: routing 集中度**(¥0、ローカル、131k tok の forward だけ)
- 各層で top-p 累積が小部分集合(数十/256)に収束するか
- **リスクの機構**: noaux_tc の aux-loss-free load balancing は expert 使用を能動的に平坦化する。V2-Lite (64 expert) より集中が弱い可能性。256 fine-grained での直接実証は文献に無し(最寄りは AWS の 128-expert)
- **集中しない → ESFT の ROI 消滅 → K 拡張路線 or 撤退**
- タスク間(coding/math/CTF)で選択集合が十分異なるかも同時に確認(ESFT 第2前提)

この profiling 装置は K 拡張路線の INC-0 routing ヒストグラムと**完全同一**。1回作れば両路線に効く。

## 4. コスト(estimated, n=0)

| 項目 | 費用 |
|---|---|
| Profiling + 選択 | ¥0(ローカル、fp4 160GB は 192GB に載る) |
| 訓練 1 domain(≤6.5e7 tok, 8×H100) | **~¥5-10k / 1-2h** |
| 3 domain(coding/math/CTF 別パッチ) | **~¥30k 総額** |
| Eval(専門 + general 回帰) | ¥0(ローカル) |

trainable の見込み: 集中度 2-4% なら 5-10 expert/層 × 40層 × 25.17M ≈ **5-10B params**、optimizer ~80-160GB → 単ノード余裕。V2-Lite 比で選択数が膨れたら(集中が弱い兆候でもある)DoRA-on-selected に切替。

## 5. 運用形: ドメイン別 expert パッチ

選択 expert の重み差分だけ保存(V2-Lite 実績で ~3GB/patch)→ **serve 時にドメインでパッチ swap**。単一モデル多ドメイン統合をやるなら DES-MoE (2509.16882) が後続(ALR + dynamic assignment、interleave 訓練で static 割当が崩れる問題への回答)だが、v1 はパッチ swap で十分。

## 5.5 転回案: ターゲットを Qwen3.6-35B-A3B に変更(2026-07-02, ユーザ提案)

**動機**: DSv4 (285B total) は訓練がクラウド前提。35B-A3B なら**全工程ローカル ¥0**。

**config 実測** (`Qwen--Qwen3.6-35B-A3B`, text_config): 40層 / hidden 2048 / **256 experts × top-8** / moe_inter 512 (per-expert 3.15M) / shared expert 512 / hybrid linear-attn / MTP あり / `router_aux_loss_coef` あり = **標準 softmax + aux-loss router(noaux_tc でない)** / `output_router_logits` フラグが**標準装備**。

**ESFT 適合性が DSv4 より高い点**:
- スケールが ESFT 論文 (V2-Lite 2.4B active/16B total) とほぼ同 régime(3B active/35B total)→ 転移確度最高
- 標準 softmax router → 論文の選択基準がほぼそのまま。noaux_tc の bias 分離問題が消える
- `output_router_logits` 標準装備 → profiling hook がほぼタダ
- AWS sample-ESFT が Qwen3-MoE 系対応済み(Qwen3.6 = Qwen3_5Moe クラスへの小移植は必要)

**リソース試算 (measured config × 机上)**:
- 選択 5/10/15% → trainable 1.5/3.2/4.8B、optimizer 24/50/77GB、patch 3.0/6.3/9.6GB
- BF16 weights 70GB + optimizer → **2×RTX PRO 6000 (192GB) に収まる**、6.5e7 tok 訓練 ~2.4-4h ローカル
- profiling 131k tok / eval も全部ローカル → **現金コスト ¥0**

**目標の再定義**: 「35B-A3B(throughput 用)を coding/math/CTF で 27B dense(quality 用)級に引き上げ、品質をスループット速度で得る」。両モデルともローカルにあるので現在ギャップの実測も ¥0。

**残るゲート**: (A) 256 fine-grained での routing 集中度(DSv4 と同じ問い、ただし aux-loss router なので文献に近い)、(B) 27B dense との現ギャップが ESFT uplift(paper +16.6pt from own baseline)で届く距離か。**ゲート B の性質に注意**: paper は「自分の baseline からの上昇」であり「9× active の dense 兄弟を跨ぐ」ことの保証ではない。

**DSv4 との関係**: この Qwen run 自体が DSv4-ESFT の INC-1(手法検証)を兼ねる。パイプラインの形は同一、成功して DSv4 に欲が出たら ¥30k で移植。

## 7. 確定方針: top-k 拡張 × ESFT の統合(2026-07-02, ユーザ決定「両方」)

**統合の要**: top-k を K* に上げた状態で profiling → ESFT すると、**「新規参入エキスパートの協調学習」問題が ESFT の枠内で自然に解ける**。
- ESFT-Token @ K* は「top-K* に頻繁に入る expert」を選ぶ = 新参(rank 9-K*)の高頻度組も選択集合に自動で入る
- ESFT は full-param 訓練なので、新参 expert は自分の出力を(renorm 後の小さい gate weight を補償する方向含め)自由に適応できる
- → DSv4 案で必要だった rank-conditional stop-gradient のカスタム forward が**不要になる**。router は ESFT 流儀で凍結(gate 可訓練は ablation arm として保留)

**フェーズ計画(全部ローカル)**:
- **Phase 0(GPU 不要、今できる)**: profiling データ 4 ドメイン(math/coding/日本語/要件定義)+ 訓練レシピ(math 先行)、ESFT repo の Qwen3_5Moe 移植(ロギング hook / to_esft パス / TOP_K)
- **Phase 1(推論のみ、1晩)**:
  1. ゲート B: 27B dense との現ギャップ(math/coding)
  2. naive top-k sweep: K=8/12/16/24/32 品質曲線 → 膝 K* 決定(事前分布: K*=16±8)
  3. ゲート A: profiling @ K=8 と @ K* → 集中度 + ドメイン間重複行列
- **Phase 2(訓練、1晩)**: math で 2-arm A/B — **ESFT@8 vs ESFT@K***(各 ~3-4h)。これで「active 増の限界利得」を same-condition で単離。事前分布: arm 差 +0.6pt 級(EMoE の K 内限界利得)、ESFT 自体は +数〜15pt 級(論文)
- **Phase 3**: 勝った arm で coding → 重複行列を見て日本語/要件定義の設計(独立パッチ or 複合ドメイン)→ 必要なら gate 可訓練 arm、DSv4 移植(¥30k)

**成功判定**: 専門ベンチで ESFT@K* > ESFT@8 > base@8 が CI 分離、汎用ベンチ非劣化。serve 時は K* の速度税(K=16 で ~1.29×)を払う点は決定済みとして記録。

## 6. 実行順(GPU 解放後)

1. ロギング hook 実装 + profiling データ準備(coding/math/CTF 各 32×4096 tok)— **GPU 不要、今できる**
2. ゲート A 実測 → 集中度レポート(層別 top-p カーブ、タスク間 overlap)
3. GO なら: to_esft 移植 + 訓練データレシピ(拒否句フィルタ込み)→ DOK 1-2h × 3 domain
4. NVFP4 再量子化 → serve → Qwen3.6-27B ギャップ + general 回帰の A/B
