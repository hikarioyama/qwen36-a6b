# MoE RL微調整 包括的文献リサーチ報告 (GROK_RL_RESEARCH_PROMPT.md 対応)

**日付**: 2026-07-08  
**目的**: プロジェクトの現行RL設計 (GRPO + router凍結 + delta-only + rule-based類似度報酬 + rejection-FT前段 + 1-step on-policy + optional GSPO) を文献で裏取り・改善。サブエージェントを最大展開してA〜Fトピック網羅収集。  
**前提 (プロジェクト実測)**: naive k=32劣化、delta-SFTで一部回復頭打ち (MMLU plateau)、MBPP -10pt (SFTスタイル圧縮)、Terminal-Bench低成績。計算: 96GB GPU時分割 (serve/train)、vLLM rollout、実行報酬不可→類似度報酬、INC-0 rejection予定。  
**制約遵守**: 2024+中心、code公開優先、measured/hypothesized区別、vendorブログと査読区別、自己申告vs再現区別。矛盾点は目立たせ両面記載。

## A. MoE特有のRL不安定性と対策

### 主要論文
- **Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers (R3: Rollout Routing Replay)**, Ma et al., arXiv:2510.11370 (2025)  
  1-2行要点: 推論エンジン (SGLang等) のrouting分布を記録し訓練時にreplay。train-infer KLを大幅低減 (Qwen3-30B-A3B MoEで1.535e-3 → 密モデル並み)、extreme token (ratio>2) ~1桁減、collapse防止、GSPO/TISをoutperform。on/off-policy両対応、速度ロスなし。  
  自己申告数字: KL/ extreme token 実測 + 訓練曲線安定 + math/SWEタスク性能向上。第三者再現: フレームワーク統合 (sglang/Miles/verl) で部分追試進行中、独立小規模再現は限定的。  
  **エビデンス強度**: 複数設定実験 (self-reportedだが診断指標豊富)。  
  **我々の設定への適用性: 高** (router凍結の代替/補完としてreplayが有効。1-step on-policyはmismatch源を減らし相性良。256 expert k=32 override時のrouter volatility対策に直結。delta-onlyとも相性可)。

- **Group Sequence Policy Optimization (GSPO)**, Zheng et al. (Qwen Team), arXiv:2507.18071 (2025)  
  1-2行要点: token-level IS/clip/optimをsequence-level (seq likelihood ratio + seq clip)に置き換え。GRPO比で効率・安定・性能向上、特にMoE RLを本質的に安定化 (複雑replay不要、router適応も可能に)。Qwen3改善に寄与。  
  自己申告: MoEでGRPOが必要としていたreplayを不要化、clipped token効率向上。第三者: フレームワーク実装で確認事例増加。  
  **エビデンス強度**: 自己申告 + Qwen3結果、アルゴ比較。  
  **我々の設定への適用性: 高** (MoE特化安定化の直接証拠。optional GSPOは文献支持。router凍結下でもseq-levelがmismatch耐性高く、1-stepと組み合わせ易い。token vs seqの差がMoEで顕著)。

- **The Stability Gap: Why Top-K Routing Breaks RL** (blog/analysis, 2025) + 関連 "Towards Stable and Effective RL for MoE" (RSPO, arXiv:2510.23027 2025頃)  
  1-2行要点: Top-Kの離散性によりgradient blackout (非選択expertに勾配0) と first-order approx失敗 (PPO/GRPO surrogate無効)。router shift/driftがIS volatilityを増幅。Freezeは安定化するがadaptivity犠牲。RSPOはrouter-aware soft rescalingでfreeze/replayより優位 (adaptivity保持 + 安定)。  
  自己申告: 数学解析 + 小規模/大規模曲線 (Countdown/MATH/Code)。第三者: 分析として広く引用。  
  **エビデンス強度**: 解析 + 実測相関 (routing driftとcollapse前兆) / 中 (小規模コントロール実験あり)。  
  **我々の設定への適用性: 高〜中** (凍結の代償を明示的に警告。プロジェクトのrouter凍結は安定に寄与するが、256 expertでrouting適応余地を失うリスク。soft rescalingやGSPO併用推奨。1-stepはIS破綻軽減)。

- **ReLibra / Pr2 / IcePop/TIS 関連 (2025-2026)**: Replayをload-balanceやmaskingに活用。R3がreplay系で最も直接的安定化報告。  
  **適用性: 中** (システム面で有用、アルゴ主眼のプロジェクトでは二次)。

**矛盾点 (router凍結)**: RSPO論文で「硬直的freezeはrouterのRL objectiveへの適応を害しunsatisfactory」と明示的否定的実験結果。GSPOは「seq-levelでfreeze/replay不要にrouter適応可能」と主張。凍結支持は主にSFT/ESFTや単純安定策。両面: 凍結はmismatch/ISを構造的に避けるが表現力制約 (特にk-expansionで余剰expert協調が必要な場合)。

**我々の現行 (router凍結 + delta-only)**: 安定化には有効寄与するが、文献は「凍結+補正」or「GSPOで適応許容」のハイブリッドを好む傾向。測定必須 (routing entropy, IS clip frac, train-infer KL)。

## B. GRPOとその改良の到達点

### 主要論文
- **DeepSeekMath: ... (GRPO導入)**, Shao et al., arXiv:2402.03300 (2024)  
  1-2行: critic不要のgroup-relative advantage (G responsesでmean/std正規化)。MATH/GSM8Kで大幅向上、メモリ効率。  
  数字: GSM8K 82.9→88.2%, MATH 46.8→51.7% (base比)。自己申告。  
  **エビデンス強度**: 高 (広く採用・再現)。  
  **適用性: 高** (プロジェクトbaseline。G=8程度、(r-mean)/std、std=0 skipは忠実)。

- **DAPO: An Open-Source LLM RL System at Scale**, Yu et al. (ByteDance), arXiv:2503.14476 (2025)  
  1-2行: GRPOのentropy collapse/zero-adv/不安定をdynamic sampling + decoupled clip (higher upper) + 修正で解決。長CoTに強い。  
  数字: AIME2024 ~50 (naive GRPO ~30)。verlフルオープンソース。  
  **エビデンス強度**: 自己 + コミュニティ再現 (verl使用)。  
  **適用性: 高** (小規模8GPUでも安定化に有効。dynamic samplingはstd=0 group問題に直結。clip設定の相場として参考)。

- **Understanding R1-Zero-Like Training: A Critical Perspective (Dr.GRPO)**, Liu et al. (Sea AI Lab), arXiv:2503.20783 (2025)  
  1-2行: GRPOのlength bias (特に誤答で長文化) + std正規化バイアスを指摘。Dr.GRPO (meanのみ + 修正集約) でtoken効率向上・同等性能。  
  数字: 7BでAIME 43.3% (8xA100 27h)。code公開。  
  **エビデンス強度**: 解析+実験、広く議論・一部再現。  
  **適用性: 高** (length bias/entropy collapse対策として。プロジェクトのstd divideはDr.GRPOが「バイアス」とする点に矛盾。1-step純REINFORCE寄りで親和性高く、採用推奨)。

- **Group Sequence Policy Optimization (GSPO)**, Zheng et al., arXiv:2507.18071 (2025)  
  (Aでも記載) seq-levelでMoE安定・効率向上。  
  **適用性: 高** (MoE + 長応答/ agenticに特に)。

- **その他**: VinePPO (arXiv:2410.01679, 2024; credit assignment改善), RLOO系 (REINFORCE leave-one-out, シンプルgroup baseline有効), AVSPO (advantage collapse対策 virtual sample, +4-6pt)。  
  **純REINFORCE-with-group-baseline**: 多くの場合十分 (RLOO/Dr.GRPO系)。ratio/clipはscale/off-policy時に必要だが、1-step on-policy + verifiable rewardでは過剰の場合あり。  
  **矛盾点 (GRPOで十分か)**: naive GRPOはentropy collapse/length bias/adv collapseで失敗多発 (DAPO/Dr.GRPO/GSPOが修正)。プロジェクトの「on-policy 1-step + optional GSPO + std=0 skip」はconservativeでvalidだが、Dr.GRPO (std除去) やDAPO dynamic/clip-higherを追加で安定性向上余地大。KL β=0.02はconservative (多くのreasoning RLで0推奨)。

**group size/KL/clip相場**: G=4-16 (8 common for small scale), β=0〜0.02 (verifiable rewardで小/0), clip 0.2 or decoupled higher-upper。

## C. rejection fine-tuning と RL の接続

### 主要論文・系譜
- **STaR (Self-Taught Reasoner)**, Zelikman et al., arXiv:2203.14465 (2022基盤)  
  1-2行: 自己生成rationale (CoT)で最終正解をフィルタ → SFT反復 (失敗時はground-truthヒントでrationalization)。reasoning bootstrapの原典。  
  数字: CommonsenseQAで+12.5ptなど。  
  **エビデンス強度**: 初期実証 (小規模)。  
  **適用性: 中** (math/reasoning bootstrapとして; agentic codingでは多様性不足リスク)。

- **RAFT (Reward rAnked FineTuning)**, Dong et al., arXiv:2304.06767 (2023)  
  1-2行: per-promptでKサンプル → RM/検証器でrank/filter (高報酬/正例のみ) → SFT反復。RL不要の安定アライメント手法。  
  数字: LLaMA-7BでPPOに匹敵/上回る報酬 (K=32)。  
  **エビデンス強度**: 複数比較・再現あり。  
  **適用性: 高** (verifiable reward下のデータ curationとして)。

- **RFT / rejection sampling FT 系 + ReST** (Yuan 2023, Gulcehre 2023など) + 2024-2026 follow-up  
  1-2行: 自己生成 → 検証通過正例のみ (またはranked) でSFT。ReSTはoffline growing batch + scoring。  
  **エビデンス強度**: 複数再現 (math/tool/agentic)。

- **A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce**, Xiong et al., arXiv:2504.11343 (2025) — **最重要再現/比較論文**  
  1-2行: RAFT (positives-only) vs GRPO/PPO/Reinforceの直接ablation。RAFTは早期に競合/優位だがentropy collapseで探索劣化し、GRPOに後れを取る。GRPOの利点の多くは「全誤りpromptのimplicit filtering」。Reinforce-Rej提案。  
  数字例 (Qwen2.5-Math-7B-base, 複数ベンチavg): Base ~23.6, RAFT 49.9, RAFT++ 52.5, GRPO 53.9。entropy/KL曲線でpositives-onlyの急激な狭まりを実測。code公開。  
  **エビデンス強度**: 厳密コントロールablation (entropy plot含む)。第三者再現価値高。  
  **我々の設定への適用性: 非常に高** (INC-0 best-of-8正例SFT → GRPOの計画に直撃。entropy collapseの定量証拠)。

**entropy collapse / 探索縮退 (実証)**: 正例のみの自己生成データで訓練するとpolicy entropyが急落 (出力分布狭窄、多様なreasoning path減少)。Xiong論文で明確: RAFTは早期精度向上するがentropy/KL悪化、pass@1↑でもpass@Kやgroup内varianceが停滞し、後段RLの相対advantage信号が弱まる。特に長horizon agentic/coding (複数戦略・回復・探索が必要)で深刻。  
**回避策** (文献): 
- 温度高め + explicit diversity (semantic/n-gram dedup)。
- 負例/拒否サンプルの慎重混合 (全誤りは有害、部分信号やfiltered negative推奨)。
- 反復数制限 (通常数回でplateau監視)。
- curriculum / process reward / tree search (ReST-MCTS*)。
- RL段階でonline filtering (GRPO group処理自体が有効)。
- entropy regや探索ボーナス。
- DeepSeek風: RLでdiscoveryした後でrejection samplingして新データ生成 → 再RL (alternating)。

**反復数・切り上げの相場**: 文献では少回 (2-5回程度) で飽和しやすい。marginal gain低下 + entropy/diversity低下 (生成のrepetitive化、group variance減少) でRL移行を推奨。  
**プロジェクトINC-0 (best-of-8自己生成 → 正例のみSFT) への適用**: 中〜高 (verifiable codingに自然)。品質ブーストと低コスト初期化として有効。ただし「完全正例only」の直後のGRPOでentropy narrowingが後段探索を鈍らせるリスクがXiong等で実証済。プロジェクトの「62%正解あり」は良質だが、完全positives-onlyを避け多様性残す設計が必要。

**矛盾点 (rejection-FTのentropy collapse)**: 
- 肯定的側: RFT/RAFTは早期に強力な品質向上を提供し、GRPOの「良い開始点」になる (DeepSeek cold-start要素、Xiongで早期競合)。
- 否定的側: positives-onlyは分布を狭め、RLが本来提供すべき相対信号・探索を損なう (Xiong: entropy collapseでGRPOに抜かれる；SFTが探索を圧縮する事例と整合)。vendor (DeepSeek-R1)はRL-first or alternatingを強調し、rejectionは主にpost-RL curationに使う。プロジェクト計画は「INC-0で自己生成+正例焼く」点でこのリスクを直視する必要あり。SFT続きでは直らない問題をRLに委ねる判断は正しいが、前段で多様性を殺さない工夫が文献から強く要請される。

**コード/agentic特化**: テスト/実行verifiable reward下でRFT自然に機能。tool-use (Hint-RFT) やagenticでpass@1向上報告あるが、多様なtrace (tree/process) がhard taskに不可欠。codingではflaky testや複数正解戦略で純positivesの多様性低下がより問題化しやすい。

### プロジェクト現行との整合・推奨 (Cから)
INC-0はゼロコスト検証として優れているが、Xiong級の知見を反映して「1反復 + 温度/多様性明示制御 + 可能なら軽い負例混合」で実施し、entropy/diversity指標を監視してから本GRPOへ。alternating (一部GRPO後でrejection curation) も有力代替。

## D. 報酬設計 (コーディング/agentic/端末タスク)

## D. 報酬設計 (コーディング/agentic/端末タスク)

### 主要論文
- **SWE-RL: Advancing LLM Reasoning via RL on Open Software Evolution**, Wei et al., arXiv:2502.18449 (2025, NeurIPS) + facebookresearch/swe-rl code  
  1-2行: ソフトウェア進化データ (PR) + ルールベース類似度報酬 (ground-truth patchとのdifflib SequenceMatcher ratio, format -1) でGRPO。Llama3-70B → 41.0% SWE-bench Verified (当時<100Bベスト級)。SFTはOOD劣化、RLはin-domain + OOD (MATH +10pt超、MMLU維持/向上) 一般化。  
  自己申告: 41% + OOD table (HumanEval+ 79.9 vs base 76.2 / SFT 73.2 等)。code完全公開 (reward.py exact match)。第三者: 再現進行、SWE-bench標準化で検証されつつ。  
  **エビデンス強度**: 高 (実データ、ablation、code)。  
  **我々の設定への適用性: 非常に高** (報酬ロジックがプロジェクトreward.pyの直接参照。SEARCH/REPLACE + diff similarity + format -1 + GRPO + KLが忠実。no-exec時の最強実践例。ハッキング対策 (no-op penalty, normalize) も参考)。

- **Self-play SWE-RL (SSR)**, Wei et al., arXiv:2512.18552 (2025/26)  
  1-2行: 最小仮定 (sandbox repoのみ) でself-play (bug inject + solve)、hybrid rule+exec報酬。+10.4 SWE-bench Verified。  
  **適用性: 高** (将来exec導入時の指針。anti-hack設計参考)。

**ルール類似度 vs 実行**: ルール (SWE-RL)はスケール容易・インフラ軽いがproxy misalignmentリスク。実行は直接的だがコスト/ハック (test edit等)。ハイブリッド推奨。  
**報酬ハッキング実例**: format only、partial solve、test削除、length hack、データ漏洩。防御: KL、strict format (-1)、verifiable、filter、no-op penalty、KLで汎用能力監視。  
**format報酬要否**: 構造パースが必要な場合必須 (SWE-RL/R1でformat accを-1で強制)。プロジェクトの<think> prefill問題に直結。  
**KLと汎用能力低下**: KLでdrift制御。reward↑だがMMLU↓の事例あり (SWE-RLではRLがSFTのOOD劣化を回避)。  
**適用性全体: 高** (rule類似度 + GRPOはSWE-RL blueprint。exec格上げ時の移行容易)。

## E. RLで「増えた容量/眠っていた重み」を活性化できるか (仮説核心)

### 主要論文
- **MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning ...**, Ko et al., arXiv:2603.24984 (2026)  
  1-2行: deterministic top-Kの代わりにGRPOでexpert selection policyを最適化。routing entropy向上 (1.05→1.82)、多様なexpert combo活性化、task specialization向上 (+1-2pt)。  
  **エビデンス強度**: 実測 (entropy/JSD/可視化)。  
  **我々の設定への適用性: 非常に高** (RLが固定routingを越えてdormant/underused expertをwakeし、balanced利用を実現する直接事例。k-expansionの余剰expert協調に最適)。

- **Does Reinforcement Learning Really Incentivize Reasoning Capacity ... Beyond the Base Model?**, Yue et al., arXiv:2504.13837 (2025)  
  1-2行: RLVR (GRPO含む)はpass@1↑するが、baseのlarge-k pass@kを下回り、pathsはbase sampling distributionに既に存在。RLはsampling efficiencyを上げるelicitation中心、新規boundary拡大は稀 (distillationの方がcreation寄り)。  
  **エビデンス強度**: pass@k coverage分析 (複数モデル/ベンチ)。  
  **我々の設定への適用性: 高** (RLは「増えた容量をより良く使う」elicitation。SFTでは届かない「既存の潜在能力を引き出す」仮説を支持するが、根本new能力創造の過大期待は戒め)。

- **SFT Memorizes, RL Generalizes** (Chu 2025) + **RL Fine-Tuning Heals OOD Forgetting in SFT** (Jin 2025)  
  1-2行: SFTはmemorize/forget (OOD早期ピーク後低下)、RLは一般化・回復 (healing)。SVDで方向rotationがforget/healと相関。  
  **適用性: 高** (SFT plateau後のRLが「抑制された容量を回復」するメカニズム証拠。k=32追加expertの「眠り」をRLで起こす根拠)。

**その他**: DeepSeek-R1 (pure RLでemergent reflection、arXiv:2501.12948) vs 上記elicitation論争。MoEでRL routing最適化例複数。  
**矛盾**: R1系「emergent new patterns」 vs Yue「bounded by base、elicitation」。SFT「new教える」vs「memorize/forget」。プロジェクト仮説 (SFT届かずRLで自然協調探す) は後者群 + MoE-GRPOで強く支持。測定: expert utilization (entropy, JSD, activation分布) pre/post必須。

**active増 + RL相乗**: 間接支持 (larger capacityで探索空間拡大、RLがroutingを最適化)。直接k増ほどRL伸びしろ大の定量は薄いが、論理的に整合。

## F. スケーリングと実務

### 主要知見
- **verl / vime / OpenRLHF 等フレームワーク (2025実践)**: Qwen3-30B-A3B MoE (類似規模) で8GPU (H200/GB200級、96GB) 上GRPO/GSPO実動報告。coloc例: step ~147-252s、MFU~0.4、EP=8 + offload。rollout律速 (G=8-16でtok/s支配)。  
  自己申告数字: 各種step time / MFU。第三者: GitHub issues + ブログで一部確認。  
  **エビデンス強度**: 実務報告 (再現性中)。

- **GSPOのインフラ利点**: seq-levelでMoE router volatility耐性高く、replayオーバーヘッド削減、infra簡素化 (Qwen3貢献)。  
  **適用性: 高**。

- **KL ref trick (delta/LoRA)**: base (delta-off) forwardでKL計算 (プロジェクトのdelta toggleと同一)。KL=0 or tiny (verifiable rewardで一般的)。offload + PEFTパターンでメモリ節約。  
  **適用性: 非常に高** (プロジェクトの「同一モデルdelta無効化でref複製不要」は文献支持パターン)。

- **RL >> SFT 定量**: verifiable domain (math/coding/agentic)でRLがSFT plateau超え or OOD elicitの報告多数 (+10pt前後例)。SFTで十分なケース: 高品質curatedデータ・単純imitation。条件: reward verifiable + 探索必要時。  
  **「RL圧倒的に重要」言説**: 条件付き (verifiable + 長CoT/agenticで真価)。SFTはformat/cold-startに有用。

- **vLLM/SGLang MoE issue**: routing override (k=32) 要検証、logprob/train-infer mismatch (R3/GSPOで緩和)、EP/TP設定敏感、AllToAll comms。Blackwell/FP8は一部成功報告 (accuracy維持 + 速度)。  
  **8xGPU実務**: rollout throughput実測必須 (プロジェクト壁の式通り)。分離 (serve/train) vs colocでtradeoff。time-share時はbuffer/decoupled設計有効。

**矛盾/注意**: naive GRPO MoEでcollapse多発 → GSPO or R3推奨。プロジェクトのtime-shared + delta + vLLMは実践例と整合 (小規模MoEで証明済みパターン)。

## 代替設計案 (自分がこのRL段を設計し直すなら、根拠論文つき 1-3個)

1. **GSPO primary + router-aware soft correction (freezeは最小限 or ハイブリッド)**: GSPO (arXiv:2507.18071) でseq-level安定化をベースに、RSPO風 router-shift rescaling (arXiv:2510.23027) でrouter適応を許容。R3 replayをオプションで補完。根拠: freezeのexpressivity損失を避けつつMoE安定 (RSPO直接比較でfreezeより優位)。delta-onlyはexpert側に集中、routerは軽く適応。1-step on-policy維持。

2. **INC-0 rejectionを強化 + Dr.GRPO + dynamic sampling**: best-of-Nで正解+多様性サンプル (entropy collapse回避策を明示)、その後Dr.GRPO (arXiv:2503.20783, std除去 + 修正) + DAPO dynamic (arXiv:2503.14476)。KL=0スタート (verifiable reward)。根拠: length/adv collapse実証 + token効率、SFT plateau後のRL加速。プロジェクトのstd divideをDr.GRPOで置き換えテスト。

3. **MoE-aware routing RL要素を追加 (MoE-GRPO風)**: 必要ならrouterも軽くRL信号 (またはmodality/task guidance) でdormant expert活性化を明示狙う + EMoE風 co-activation (arXiv:2509.21892 for k-expansion)。ただし安定優先でexpert delta主軸維持。根拠: MoE-GRPOでrouting entropy実測向上 + k-expansion直接論文 (EMoE)。

優先: 1 or 2から小規模INC-0拡張でA/B検証 (same-condition n>=2)。

## 読むべき優先度 top 10 (理由つきランキング)

1. **SWE-RL arXiv:2502.18449 (2025)** + code: 報酬設計の完全一致。rule類似度GRPOのblueprint。実測41% + OOD一般化が核心。
2. **GSPO arXiv:2507.18071 (2025)**: MoE RL安定化の最直接的改良。プロジェクトMoE + optional GSPOに即適用。
3. **R3 arXiv:2510.11370 (2025)**: MoE router mismatchの診断とreplay解決。凍結の文脈で両面理解必須。
4. **A Minimalist Approach... (Xiong et al. arXiv:2504.11343, 2025)** + code: RFT/RAFT vs GRPOの厳密ablation。positives-only entropy collapseの定量証拠と「GRPO利点の本質はfiltering」。INC-0計画に最重要。
5. **Dr.GRPO arXiv:2503.20783 (2025)** + code: GRPOバイアス (length/std) の批判的解析。1-step設計の改善指針。
6. **DAPO arXiv:2503.14476 (2025)**: 実務スケール安定化パッケージ (dynamic/clip)。8GPU級に有用。
7. **ESFT arXiv:2407.01906 (2024)** + GitHub: router凍結 + 選抜expert deltaのSFT基盤。プロジェクト全体設計の根拠。
8. **MoE-GRPO arXiv:2603.24984 (2026)**: RLでdormant expert活性化の直接実測。仮説Eの最強支持。
9. **DeepSeekMath GRPO arXiv:2402.03300 (2024)**: 原典。group advantageの基礎。
10. **Yue et al. "Does RL Really Incentivize ... Beyond Base" arXiv:2504.13837 (2025)** + EMoE (2509.21892): elicitation限界 + k-expansion collaboration問題。測定の重要性とTop-K病理。

**補足**: 全論文でcode/実装可能なものを優先。プロジェクトは「短めINC-0 (多様性制御) → GSPO/Dr.GRPO強化 (必要時R3/hybrid)」が文献から最も整合的。常にsame-condition実測 (n>=2、entropy/diversity指標含む) で進路決定を。

---
**生成メモ**: サブエージェント7個並列展開 (全トピックA-F + 補完) で網羅。最終Cサブエージェント (Xiong等詳細) を統合。self-reportedと再現/分析、measured/hypothesizedを区別。router凍結 / GRPO / rejection entropy collapseの矛盾を両面明記。実務 (verl/vLLM/8GPU/SWE-RL) もカバー。追加深掘り時は特定subagent resumeやbrowse可能。