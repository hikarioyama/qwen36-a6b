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

### eval matrix 完成(n=600 intel / 164 HE / 500 MBPP、paired McNemar)

| アーム | MMLU | GSM8K | HumanEval | MBPP |
|---|---|---|---|---|
| base@k8 | **0.843** | **0.893** | 0.866 | 0.866 |
| base@k32 | 0.807 | 0.865 | 0.841 | 0.792 |
| patch(agentic)@k32 | 0.813 | 0.885 | **0.902** | 0.828 |
| coding特化@k32 | 0.805 | 0.820 | 0.762 | 0.718 |

**base@k8 比 paired McNemar(北極星は「base@k8 に有意勝ち」)**:
- patch@k32 vs base@k8: MMLU −3.0(p=.010 **負け**)/ GSM8K −0.8(p=.49 **引分**)/ HumanEval +3.7(p=.31 **引分**)/ MBPP −3.8(p=.027 **負け**)
- **正直な現在地: agentic patch はまだ base@k8 に有意勝ちしたベンチが1つも無い**(2引分・2負け)。これまでの「勝ち」は全て vs base@k32(naive)相手。base@k8 超えは混合訓練(走行中)+ INC-0/GRPO の宿題。
- coding特化@k32 は全ベンチで base@k8 に有意負け(MBPP −14.8pt)。ドメイン特化 patch 路線の死を再確認。

- **発見1: naive k32 は知識系を実劣化させる**(MMLU −3.7pt / GSM8K −2.8pt vs k8)。agentic patch は部分回復(GSM8K p=.029 有意、vs base@k32)+ HumanEval で有意勝ち(p=.041 vs base@k32)。**複雑タスクほど転移**。
- **発見2: coding 特化 patch は全面失敗**。GSM8K 0.820(base@k32 にすら p=.002 有意負け)、HumanEval 0.762(agentic patch に p=.0001 大敗)。**機構 = 生成長の転写**: median 186 tok(base 2588 / agentic patch 1064)で reasoning が焼き殺された。静的コード 111k の「問題→即答」形式がスタイルごと転写された。**教訓: SFT は能力より先に振る舞いを書き換える。ドメイン特化 patch 路線は死、混合+agentic が本線。**
- **発見3: agentic patch は「良い圧縮」**。base の 1/3 の思考長で最高精度 + 途切れ解消 + 実質3倍速。
- **gpu-host 編入**: 8×RTX PRO 6000(768GB VRAM)提供を受け、大規模側の計算制約が消滅。venv/repo/cache 配備済(HF DL は HF_HUB_DISABLE_XET=1 必須の罠)。
- **外部リサーチの敵対検証**(Grok 報告 → Opus 8体 + Fable 判定): 幻覚 arXiv ID ゼロ、ただし盛り/誤読 5件(ESFT「FFT 9pt+劣化」は実際 −3.6pt、replay 相場 10-30% は実際 1-5% で足りる、他)。**戦略級知見: (a) Matryoshka 論文が Qwen3-30B-A3B を直接測って k 変更の急劣化を示す(うちの実測と整合)(b) 弾性訓練でも実証レンジは native の 2-3×= k=32(4×)は文献の外の extrapolation**。GSPO は「ほぼ必須」ではなく、うちの RL v1(on-policy 1-step、router 凍結)は問題を構造的に半分回避済み。vLLM #36872(姉妹モデル FP8 + native MTP で accept 61%→0% 崩壊)= MTP graft は再現ゲート先行。
- **決定(ユーザ裁定)**: ①gpu-host 投入は corpus v2(replay 2-4%: Nemotron chat/science、汚染ゲートフル再走)完成後 → p0.2 本走。aux-host p0.18/v1 は完走させ corpus×expert 予算の A/B に。②ローカル次弾 = k sweep(base@k16/k24)→ ckpt600 知識回復チェック → TB2.0 3アーム夜間。

### 3台運用ドクトリン v1(2026-07-07、ユーザ提起で策定)

役割固定: **gpu-host = 主砲**(GPU数が効く律速工程のみ: 訓練本走/GRPO/大量rollout/pack)、**aux-host = 安定炉**(長時間単発ジョブ、A/B control 腕、データ工場 CPU)、**ローカル = 計測室**(全 eval + TB。測定環境をここに固定して same-condition を守る。ユーザ優先)。

選択基準(上から順に): ①律速工程に最強リソース、律速外に gpu-host を使わない ②役割表を破る時は理由を DEVLOG に書く ③重い run の前に必ず安いゲート(INC-0 原則)④1 GPU 1 ジョブ・訓練同居禁止・**空き GPU を埋めるための仕事は作らない**(遊休には安く独立な保険仕事のみ)⑤同格なら「早く次の判断をくれる」実験を優先 ⑥起動権限 main 一本化・借り物の作法。

### corpus v2 replay レーン(2026-07-07、検証合格)
- Nemotron-Post-Training の chat/science split から replay レーン2本: general 8.0M tok(5,790 recs)+ knowledge 4.0M tok(2,139 recs)= mixed_v1 の 2.89%。文献の 1-5% band 内。
- **Fable 独立検証**(agent 自己申告を裏取り): decontam.py audit 再走で両レーン residual 0、JP≥10=0、スキーマ純度 100%。生成長 median general 4.5k / knowledge 5.2k 字 = coding patch を焼き殺した即答186tok の真逆で、reasoning-heavy な振る舞いを補強する側。
- 裁可: system_prompt 非注入(Llama-Nemotron 制御トークンは Qwen に異物)/ reasoning on-off 両保持 + `<think>` verbatim / 加算 merge(415.9M→427.9M)。
- **honest note**: knowledge レーンは MMLU-style 科学多肢選択。decontam は exact/13gram/containment のみで **paraphrase レベルの近似は構造上スコープ外**(コーパス全体に共通の既知限界)。embedding-sim 硬化を追加するかはユーザ裁定待ち。

### gpu-host 8-GPU: mixed_v2 p0.2 本走 起動(2026-07-07 07:00 JST)
- **probe8 で p0.2(833 experts、trainable 2.62B)が 96GB/GPU に収まることを実測**(aux-host では幽霊リークで不可だった)。8-rank NCCL 正常、**21.6 s/step**(aux-host 2-GPU 78.8 の 3.65倍)。
- **転送の壁と大技**: aux-host→どこでも 0.9 MB/s(borrowed 機の住宅回線 上り≈7Mbps、tailscale は direct で relay ではない=物理限界)。mixed_v2 生 4.2GB を送ると 78分。**回避 = mixed_v2 の 99%(v1 部分)は既に gpu-host に packed cache で存在**。新規は replay 2レーン 54MB だけ → それだけ zstd 転送(31秒)→ gpu-host で pack(7929rec→1440 blocks、6秒)→ 既存 v1 cache と torch.cat → mixed_v2 cache(59468 blocks、supervised 62.3%、manifest 整合)。**4.15GB の転送を回避**。
- 起動: 8-GPU DDP、seq7168 / fused-CE(FLCE)/ Adafactor / ga2 / p0.2 config / 3150 step / save300。全GPU 94-100%・90.5GB。ETA ~19h。runs/mixed_v2_esft_k32_p02。
- **これで aux-host(p0.18/v1、55h)と gpu-host(p0.2/v2、19h)の2本が並走** = expert 予算(0.18 vs 0.2)× corpus(replay 有無)の best-shot アーム。build-the-best 方針で、最終は base@k8 比で判定。
- 運用教訓: 転送チェーンを2本同時に走らせて aux-host 回線を食い合い共倒れ + audit 起動 ssh の `&` が返らずスクリプトがハング → 両方 kill して単一目的の綺麗な転送に。**細い共有リンクでは転送は直列化・単一責務に**。

### Terminal-Bench 北極星ベースラインのハーネス修理(2026-07-07)
前セッションの base_k8 TB run が「0/24 solved + 12 error」で放置されていたが、解剖したら**有効な測定ですらなかった**。系統的にバグを剥がした:
1. **12「エラー」= SIGTERM キャンセル**(exception.txt が `_handle_sigterm→KeyboardInterrupt`)。前回セッションが run を kill しただけ、`finished_at: null` の未完。0/24 は幻。
2. **port 18000 = 前セッションの死んだ SSH トンネル**(aux-host:8000 宛、今は訓練中で無応答)。ローカル serve は 8001 → ハーネスを 8001 直叩きに変更。
3. **NotFoundError = モデル名不一致**(local serve `qwen36a3b` vs ハーネス `Qwen3.6-35B-A3B`)。serve を `--served-model-name Qwen3.6-35B-A3B` に統一。
4. **RuntimeError = Docker イメージ pull の transient reset**(IPv6 経路で大レイヤが `connection reset`)。ubuntu:22.04 が retry で通ることを確認 → transient と判定、再走で通過。
5. **AgentTimeoutError = モデルが遅いだけ**。trajectory 精査で **base@k8 は実際に賢く駆動していた**(llm-inference タスクで 174 step / 20 コマンドバッチ、ファイル読み→Python 実行→baseline_packer 解析まで到達)。reasoning 重めで default timeout に間に合わず。→ `--agent-timeout-multiplier 3.0` で完全ベースライン起動。
**教訓: 「0/24」を「モデルが弱い」と読まず解剖したのが正解**(memory の「壊れると『モデル弱い』と誤読」警告どおり)。ハーネスは生きていた。runs: base_k8_full(89タスク、n_conc4、timeout 3x、夜間)。serve=serve_bf16.sh 8(TP2、8001、Qwen3.6-35B-A3B)。

## 2026-07-08 — gpu-host v2/p0.2 完走・知識系初判定

- **訓練完走**: 3150 step / 18h20m / rc=0 / peak 91.55GiB(probe 予測どおり)/ 20.96 s/it。patch 5.2GB(1666 tensors = 833 experts)。best step 3100 の ckpt は save 間隔の狭間 → 最終 3150 の重みを patch 化(eval_loss ほぼ同値)。
- **知識系判定(gpu-host 同一マシン paired、n=600、choice-logprob)**:
  - MMLU: base@k8 0.840 vs **v2patch@k32 0.820**(−2.0pt、p=0.14 **引き分け**)
  - GSM8K: base@k8 0.887 vs **v2patch@k32 0.865**(−2.2pt、p=0.26 **引き分け**)
- **解釈(honest)**: naive k32 は MMLU で有意負け(p=.002)だったのが、v2 混合+replay で**統計的引き分けまで回復**。ただし「勝ち」ではなく、点推定は依然 −2pt。GOAL の非劣化ゲート(MMLU −1pt 以内)は点推定で未達(CI [−4.4, +0.4] で判定保留)。GSM8K は v1 agentic patch(local 0.885)から数値上の上積みなし — **replay 1-5% では知識ギャップを閉じ切れない可能性**が濃くなった。次の診断 = ckpt 軌道(900/1800/2700)で「まだ伸びてる途中」か「頭打ち」かを判定 → 頭打ちなら full-FFN エスカレーション条件に接近。
- 並走中: v2patch と base@k8 の code 系(HE/MBPP)を gpu-host GPU4-7 で測定中。TB base@k8 は 54+/89(solved 0 継続)。
- cross-machine 注記: base@k8 は gpu-host で MMLU 0.840 / GSM8K 0.887(local 0.843/0.893)— マシン間で −0.3〜−0.6pt ずれる実測。**同一マシン paired の再測定は正解だった**。

### 走行中(2026-07-07 07:00 JST)
- gpu-host: mixed_v2 p0.2 本走 8-GPU(上記、step 0→、ETA 19h)
- aux-host: mixed_v1 p0.18 本走 step ~600/3150(corpusv2 解放で CPU 競合解消、78s/it に復帰見込み)
- ローカル: k sweep(k24 測定中、k16=MMLU 0.828 確認済)
- 保留: layer1 domain別 NLL(ckpt300/600、転送競合で中断→gpu-host 本走後に再開)
