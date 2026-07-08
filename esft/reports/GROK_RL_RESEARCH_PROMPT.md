# 依頼: MoE モデルの RL 微調整に関する包括的文献リサーチ

## 大目的(全判断の基準)

MoE 言語モデル(256 experts/層、40層、native top-8 routing)の推論時 top-k を 32 に引き上げた「拡張容量」構成を、強化学習(RL)で仕上げる。ベースは Qwen3.6-35B-A3B 相当(総35B / native active 3B → top-32 で active ~6.6B)。**RL の役割仮説**: 教師あり微調整(SFT)は「模範解答の写経」なので、top-8 時代の文体に縛られ、増えた24 experts(gate 質量の54%を運ぶ)の最適な協調を引き出せない。RL は目標が「タスク成功」なので、増えた容量にとって自然な協調をモデル自身に探させられる — これが SFT の頭打ちを超える理論的根拠(仮説)。

このリサーチの目的: **我々の RL 設計を文献で裏取り・改善・必要なら差し替える**。現行選択は仮説にすぎない(後述)。より良い経路を示す文献があれば遠慮なくそちらを推してほしい。

## 実測済みの事実(前提として使ってよい。paired、exact McNemar、n=600/164)

- naive top-32(訓練なし): MMLU 0.843→0.807(有意)、GSM8K 0.893→0.865。JMMLU(日本語) 0.768→0.750(非有意=日本語は英語知識より頑健)。
- 選抜 expert delta-SFT(top-p 0.2 で 833/10,240 expert を選び residual delta 2.62B のみ訓練、router 凍結)+ 混合コーパス(agentic 62%/coding 12%/toolcall 11%/math 10%/replay 3%、428M tok): MMLU 0.820(vs base@k8 −2.0pt、p=0.14=統計的引き分け)、JMMLU 0.768(base@k8 と完全同点)。**checkpoint 軌道で MMLU が step1200/2100/3150 = 0.825/0.823/0.820 と頭打ち → 選抜 delta の容量天井**。
- コーディング: HumanEval は base@k8 と同点、**MBPP は −10pt(p<.0001)= SFT が生成長を median 2852→531 tok に圧縮するスタイル転写。SFT の続きでは直らないと判断 → RL の領分**。
- Terminal-Bench(実行ベース agentic ベンチ): 素の base@k8 も我々の patch も現状 solved 0(測定中、素の Qwen は長考で時間切れ多発)。

## 現行の RL 設計(決定でない。覆すの歓迎)

- アルゴリズム: **GRPO**(group-relative、1プロンプト 8 rollout、group 内 (r−mean)/std を advantage、std=0 group skip)。TRL/veRL 不使用の自前ループ。
- **router 凍結のまま RL**(選抜 expert の residual delta のみ更新)。理由: MoE-RL の不安定性(訓練中に活性 expert が変わり重要度比が揺れる)を構造的に回避する狙い。
- KL 参照: **同一モデルで delta を無効化**して base logp を計算(参照モデル複製が不要)。KL β=0.02 から。
- 報酬: 第1段は**ルールベース類似度**(生成 patch と正解の類似度、実行不要)でループの正しさを検証 → 第2段で**実行ベース報酬**(コンテナ内でテスト実行、pass/fail)に格上げ。
- 助走: GRPO 前に **rejection fine-tuning**(best-of-8 で自己生成→正解のみ SFT。best-of-8 で 62% の問題に正解あり)。
- 段階: SFT(済)→ rejection-FT → GRPO。
- 計算: 96GB GPU ×8 のサーバー1台(serve 4 + train 4 の分業想定)。Blackwell/SM120。

## リサーチトピック

### A. MoE 特有の RL 不安定性とその対策
- MoE を RL(GRPO/PPO 系)で訓練する時の固有問題(router collapse、expert 負荷不均衡、rollout と train の活性 expert 不一致 = importance ratio の破綻)と、各対策の比較エビデンス。
- **router 凍結**は MoE-RL 安定化に有効か? それとも routing replay / router-aware importance rescaling のような「揺れを許容して補正する」方が優れるか。凍結の代償(表現力の制約)を測った研究はあるか。
- GSPO(sequence-level importance ratio)、Routing Replay、R3(train/inference router alignment)、router-logit IS rescaling 等の直接比較。small-scale(GPU 8枚以内)で最も安定なのは。

### B. GRPO とその改良の到達点
- GRPO / DAPO / Dr.GRPO / RLOO / GSPO / VinePPO / GPG 等の 2024-2026 の比較。実務で「まず試すべき」デフォルトは。
- group size(rollout 数)、KL β、advantage 正規化(std で割る是非=Dr.GRPO の主張)、clip 設定の相場。
- 「on-policy 1-step(importance ratio なし)の純 REINFORCE-with-group-baseline」は GRPO の特殊形として妥当か、それとも ratio/clip は実用上必須か。

### C. rejection fine-tuning と RL の接続
- STaR / RAFT / RFT / rejection sampling FT → GRPO の系譜。rejection-FT を GRPO の前段に置く効果と、**entropy collapse(正解のみ訓練で探索が縮退し後段 RL に不利)** の実証と回避策。
- rejection-FT をどこで切り上げて GRPO に渡すべきか(iteration 数の相場)。

### D. 報酬設計(コーディング/agentic/端末タスク)
- ルールベース類似度報酬 vs 実行ベース報酬(unit test pass)の比較。SWE-RL 型類似度報酬の再現報告と落とし穴。
- 報酬ハッキングの実例と防御(端末/コード RL 特有: テストを消す、簡単な部分だけ解く、format だけ合わせる 等)。
- format 報酬の要否、KL による base からの逸脱制御と「reward 上がるが汎用能力(MMLU 等)が落ちる」現象の対処。

### E. RL で「増えた容量/眠っていた重み」を活性化できるか(この仮説の核心)
- **SFT では届かない性能を RL が引き出す**という主張の機構的エビデンス。RL は新しい能力を教えるのか、既にある能力を引き出す(elicit)だけなのか、の議論(2024-2026 で活発なはず)。
- MoE で「あまり使われない expert を RL で活性化して性能を上げた」直接事例(例: routing の偏りを叩いて眠った expert を起こす系)。
- 拡張容量(active param 増)と RL の相乗効果を測った研究。active を増やすほど RL の伸びしろが大きい、という主張の裏付け or 反証。

### F. スケーリングと実務
- 8×GPU 級で MoE-RL を回す実務レポート(serve/train 分業、rollout スループット律速、cycle あたりコスト)。
- vLLM/SGLang での MoE + RL rollout の既知 issue。
- 「RL は SFT より圧倒的に重要」という一般的言説の、定量的な裏付け(RL の絶対的な gain 幅の相場)と、それが成り立つ/成り立たない条件。

## 出力形式(厳守)

1. トピック A〜F ごとに: 論文リスト(タイトル + arXiv ID + 年)、各1〜2行の要点、**我々の設定への適用性(高/中/低 + 理由)**。
2. 論文の自己申告の数字と、第三者による再現・追試の有無を区別。
3. 我々の現行設計と**矛盾する知見は目立つ形で**報告(隠さない・丸めない)。特に「router 凍結」「GRPO で十分か」「rejection-FT の entropy collapse」の3点は賛否両論を必ず両面。
4. 各知見に**エビデンス強度**(単一論文のみ / 複数再現あり / ベンダー主張のみ)を明記。
5. 最後に「自分がこの RL 段を設計し直すなら」の代替案を1〜3個、根拠論文つき。
6. 読むべき優先度 top 10 を理由つきでランキング。

## 制約
- 2024年以降を中心に、基礎(PPO/GRPO/STaR 原典)は含めてよい。コード公開ありを優先。断定を避け、measured と hypothesized を区別。ベンダーのブログは査読論文と明確に区別。
