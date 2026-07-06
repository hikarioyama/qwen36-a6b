# DEVLOG — Qwen3.6-35B-A3B → A6B (k=32) 強化キャンペーン

**大目的**: Qwen3.6-35B-A3B(MoE、総35B / active 3B、256 experts/層、native top-8)の推論時 top-k を 32 に引き上げた「35B-A6B」を作り、知能・一貫性・コーディング/agentic の3軸で base(k=8)を統計的有意に上回る。北極星 = Terminal-Bench。ハードゲート = ベンチ汚染ゼロ、汎用非劣化、数字は (n, same-condition, CI) 付きのみ。

**測定規律**: 同一条件 A/B + paired McNemar + cluster bootstrap。shuffle seed 0 固定。measured と hypothesized を区別。負の結果も記録する(むしろ律速項を指す clue として歓迎)。

---

## 2026-07-02 — 構想確定・ターゲット転回

- ターゲット選定: クラウド前提の候補を棄却し、**全工程がローカルで回る Qwen3.6-35B-A3B に確定**(訓練 trainable 1.5-4.8B、2×RTX PRO 6000 で完結)。
- 手法確定: **top-k 拡張 × ESFT(Expert-Specialized Fine-Tuning, 2407.01906)の合わせ技**。routing 頻度上位の expert FFN だけを delta(残差)方式で訓練、router は凍結。
- 文献テーゼ: 「naive な k 増は効かない/壊れる(EMoE 2509.21892, Matryoshka 2509.26520)+ 増やした容量は訓練して初めて効く」— この2段目がうちの計画そのもの。

## 2026-07-02/03 — Phase 0: 訓練基盤実装

- `esft/` 実装。**Qwen3.6 MoE の要注意事実**: experts は packed 3D Parameter(`gate_up_proj (256,1024,2048)`)なので requires_grad で凍結できない → grad hook で非選択 expert 行ゼロ化 + expert 群 wd=0。bit-exact 凍結を CPU で実証。
- delta 方式で trainable 32B→2.46B(勾配 64GB→4GB)。スモーク 22/22。
- 敵対レビューが FATAL を発見: train_esft が k=8 のまま訓練(rank9-32 に勾配が流れない)→ k=32 override 実装で修正。
- gate-mass 分析: 追加 rank9-32 が renorm gate 質量の **54.0%** を運ぶ(rank1-8=46%)。「4倍容量は名目だけ」を否定、k=32 の妥当性を実測で支持。

## 2026-07-03 — pilot 2本と方向転換

- math ESFT@32 pilot(300 step): GSM8K が base@8≈0.90 の near-ceiling で**非情報的**。教訓: headroom の無いベンチで効果測定するな。
- ユーザ方向づけ: 「なぜ数学にこだわる、コーディングが一番」→ coding 優先へ。**北極星 = Terminal-Bench 確定**、比較の物差しより「到達可能な最強を作る」(build, don't compare)。
- coding pilot で **AdamW OOM → Adafactor 解**(delta 方式は optimizer state 律速: AdamW 8B/param ≈ 19.7GB。Adafactor で state ~0)。
- 良質データ 13本 DL(Terminal-Corpus、When2Call、OpenCodeReasoning-2 等、license 全確認)。Claude セッションログのリークデータ(いわゆる Fable traces)は品質・法務両面で **REJECT**。

## 2026-07-04 — agentic SFT 完走・2つの壁を実測で分解

- **VRAM 壁の真因 = CE loss の logits 実体化**([seq×vocab 248320] fp32、seq8192 で 8.1GB)。attention でも linear-attn でもない(fla を疑ったのは早合点=盛り、traceback を読めば一発だった)。**Liger FLCE 統合**で解決(FLCE vs 参照 CE |diff|=1.19e-7)。
- **RAM 壁**(63k 軌跡のトークン化で 120GB) → int32 streaming pack で 25GB に。
- agentic SFT 本走: Terminal-Corpus 63,621 軌跡(SWE-bench_Verified + TB2 exact-match decontam 0 drop)、DDP seq7168、509 step 完走。patch 4.9GB。
- vLLM SM120 修理(cu130 nightly + flashinfer JIT の5段障害潰し)、単発 176.7 tok/s (n=1, cold, k=8, TP2)。

## 2026-07-05 — RL 基盤・データ考古学

- SWE-RL(2502.18449)verbatim 報酬関数(48 tests)。**SWE-smith-trajectories の patch 列は破損(shuffle、repo 一致 2%)を発見** → gold patch を instance_id join で再構築(300/300 整合)。RL データ v1 = 5,175件、decontam 0 drop。
- INC-0 rollout 384×8: lenient bo8 = **0.621**(rejection-FT の弾薬は十分)。
- **形式問題の真因特定**: Qwen template は gen prompt 側が `<think>` を供給する(モデルは開始タグを書かない)→ **assistant prefill `<think>\n` が解**(paired Δlenient +0.394 CI[+0.148,+0.647])。GRPO では prefill 再結合が必須要件。

## 2026-07-06 — 混合コーパス mixed_v1・eval インフラ完成・訓練起動

- **mixed_v1**: 415.9M tok(agentic 64 / coding 12.3 / toolcall 11.2 / math 10.2 / 封筒 2.3%)。汚染ゲート4層(word-13gram ∪ 正規化 exact ∪ 短問題 containment ∪ HE entry_point 署名 purge ∪ TB instruction 本文 178 exact ∪ JMMLU JP≥5 shingle)で**残留 0**。JP filter(JP/CJK≥10 で drop)ハードゲート化。
- Phase1 cache の破損(retroactive think strip → user ヘッダ欠落)を **preserve_thinking one-shot render** で修正、invariant スキャナで検証。
- eval 側: MMLU は think 溢れ trunc が測定を壊す → **choice-logprob 化**(hidden+lm_head 手動適用、full logits 64.5GiB OOM 回避)。MMLU first-N のアルファベット順偏り → shuffle seed 0。
- 訓練起動: aux-host 2×PRO6000、seq7168 / fused-CE / Adafactor / grad-accum 8 / 3150 step。幽霊 2.5GB VRAM リークで p0.2(833 experts)が OOM → **p0.18(730 experts)で稼働**。
- 運用事故と対策: 二重起動事故・eval wedge 7時間沈黙 → **起動権限を main に一本化、agent は read-only 監視、30分 heartbeat 常設**。pgrep 自滅 3回 → bracket パターン必須。

## 2026-07-07 — eval matrix 完成間近・k8>k32 問題・coding patch の死・文献検証

### eval matrix(n=600 intel / 164 HE / 500 MBPP、paired McNemar)

| アーム | MMLU | GSM8K | HumanEval | MBPP |
|---|---|---|---|---|
| base@k8 | **0.8433** | **0.8933** | 0.866 | (走行中) |
| base@k32 | 0.8067 | 0.8650 | 0.841 | 0.798 |
| patch(agentic)@k32 | 0.8133 | 0.8850 | **0.9024** | 0.828 |
| coding特化@k32 | 0.805 | 0.820 | 0.762 | (走行中) |

- **発見1: naive k32 は知識系を実劣化させる**(MMLU −3.7pt / GSM8K −2.8pt vs k8)。agentic patch は部分回復(GSM8K p=.029 有意、vs base@k32)+ HumanEval で有意勝ち(p=.041 vs base@k32)。**複雑タスクほど転移**。
- **発見2: coding 特化 patch は全面失敗**。GSM8K 0.820(base@k32 にすら p=.002 有意負け)、HumanEval 0.762(agentic patch に p=.0001 大敗)。**機構 = 生成長の転写**: median 186 tok(base 2588 / agentic patch 1064)で reasoning が焼き殺された。静的コード 111k の「問題→即答」形式がスタイルごと転写された。**教訓: SFT は能力より先に振る舞いを書き換える。ドメイン特化 patch 路線は死、混合+agentic が本線。**
- **発見3: agentic patch は「良い圧縮」**。base の 1/3 の思考長で最高精度 + 途切れ解消 + 実質3倍速。
- **gpu-host 編入**: 8×RTX PRO 6000(768GB VRAM)提供を受け、大規模側の計算制約が消滅。venv/repo/cache 配備済(HF DL は HF_HUB_DISABLE_XET=1 必須の罠)。
- **外部リサーチの敵対検証**(Grok 報告 → Opus 8体 + Fable 判定): 幻覚 arXiv ID ゼロ、ただし盛り/誤読 5件(ESFT「FFT 9pt+劣化」は実際 −3.6pt、replay 相場 10-30% は実際 1-5% で足りる、他)。**戦略級知見: (a) Matryoshka 論文が Qwen3-30B-A3B を直接測って k 変更の急劣化を示す(うちの実測と整合)(b) 弾性訓練でも実証レンジは native の 2-3×= k=32(4×)は文献の外の extrapolation**。GSPO は「ほぼ必須」ではなく、うちの RL v1(on-policy 1-step、router 凍結)は問題を構造的に半分回避済み。vLLM #36872(姉妹モデル FP8 + native MTP で accept 61%→0% 崩壊)= MTP graft は再現ゲート先行。
- **決定(ユーザ裁定)**: ①gpu-host 投入は corpus v2(replay 2-4%: Nemotron chat/science、汚染ゲートフル再走)完成後 → p0.2 本走。aux-host p0.18/v1 は完走させ corpus×expert 予算の A/B に。②ローカル次弾 = k sweep(base@k16/k24)→ ckpt600 知識回復チェック → TB2.0 3アーム夜間。

### 3台運用ドクトリン v1(2026-07-07、ユーザ提起で策定)

役割固定: **gpu-host = 主砲**(GPU数が効く律速工程のみ: 訓練本走/GRPO/大量rollout/pack)、**aux-host = 安定炉**(長時間単発ジョブ、A/B control 腕、データ工場 CPU)、**ローカル = 計測室**(全 eval + TB。測定環境をここに固定して same-condition を守る。ユーザ優先)。

選択基準(上から順に): ①律速工程に最強リソース、律速外に gpu-host を使わない ②役割表を破る時は理由を DEVLOG に書く ③重い run の前に必ず安いゲート(INC-0 原則)④1 GPU 1 ジョブ・訓練同居禁止・**空き GPU を埋めるための仕事は作らない**(遊休には安く独立な保険仕事のみ)⑤同格なら「早く次の判断をくれる」実験を優先 ⑥起動権限 main 一本化・借り物の作法。

### 走行中(2026-07-07 06:00 JST)
- aux-host: mixed_v1 訓練 step ~560/3150(78.8s/it、loss 0.462、eval_loss 0.4877@300)
- ローカル: eval matrix 最終アーム(coding_k32 MBPP)
- corpus v2 replay レーン構築(Opus agent、merge 前に Fable 検証で停止)
