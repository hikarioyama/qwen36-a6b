# Master Plan — Qwen3.6-35B-A3B → Agentic (Terminal-Bench) via ESFT@k=32 + GRPO

最終更新: 2026-07-04。この文書が現行の唯一の計画。GOALS.md(math/McNemar 時代)は superseded。

---

## 0. GOAL(北極星)

> **Qwen3.6-35B-A3B を active 6.6B(k=32)のまま、汚染ゼロで agentic に強くする。**
> 物差し = **Terminal-Bench 2.x の resolution rate**。素の baseline から ESFT@k=32 + agentic SFT(+GRPO)で、**NVIDIA が terminal-corpus で実証した 25–30% 帯**への到達を狙う。**10日目標・上限14日でリリース。**

**成功基準(measurable / same-condition)**
1. **主(能力)**: Terminal-Bench 2.x resolution が強い agentic 帯(stretch 25–30%)に届く。pass@k, k≥5。
2. **健全性チェック**: 訓練後が未訓練 baseline を CI 分離で上回る(= 訓練が効いた証拠。goal 本体でなく sanity)。
3. **構造仮説**: k=8 ESFT control に対し k=32 が Terminal-Bench で優位(Phase 4)= 「4倍 active 容量を訓練で使い切った」実証。27B は 9x の active compute、効率で並べば構造的に勝ち(build-don't-compare)。
4. **非劣化**: MMLU/JMMLU −1pt 以内。

**🔴 ハードゲート(非交渉)**: ベンチ汚染ゼロ(訓練前 exact-match decontam 済 + eval 前 n-gram audit)。

**正直な注記(盛らない)**: baseline は**未測 → 最初に取る**(現在地)。**27.4% は NVIDIA の到達点であって我々の保証値でない**、25–30% 帯は stretch。数字には常に `(measured / hypothesized, n=?, same-condition?)` を添える。長期 agentic 汎化は Phase 3(GRPO)が担う。

---

## 1. 手法スタック(レバー、順序つき)

| # | レバー | 中身 | 位置づけ |
|---|---|---|---|
| L0 | base | Qwen3.6-35B-A3B(256 experts / native top-8) | 土台 |
| L1 | **active 拡張 + ESFT** | top-k 8→32、選抜 expert のみ full-weight 訓練、router 凍結、residual-delta | 構造的差別化 |
| L2 | **agentic SFT** | verified 軌跡(Terminal-Corpus + SWE)で多turn agent loop を焼く | agentic の driver |
| L3 | **GRPO × ルールベース報酬** | SWE-RL 方式、報酬=生成 patch と oracle の類似度(**実行なし=Docker 不要**) | 汎化を上乗せ |
| L4 | (option) co-activation 訓練 | 確率的-k で expert 協調を学習(EMoE)。速度フリーなので純上積み | k=32 を "使える強さ" に |
| L4' | (INC-0) rejection-sampling FT | N生成→高報酬 best 選抜→既存 SFT で焼く。GRPO 前の最安ゲート | 安く当たりを見る |

**正直な機構**: 「active 増→agentic」は**条件付き**。naive 増は EMoE の scaling wall で劣化。効くのは L1+L2+L3(+L4)が揃った時。容量は enable、agentic を deliver するのはデータ+RL。構造仮説(k=32>k=8)は phase 4 の control で測って決着させる。

---

## 2. データ(回収済み + decontam ゲート)

**回収済み(aux-host `~/esft/data/hf/`、全て実在検証・license clean・スクレイプ除外)**:
- **nvidia/Nemotron-Terminal-Corpus**(7.7GB, cc-by-4.0)= 本命。Qwen3 を Terminal-Bench 3.4%→27.4% にした張本人。226k=Math/Code/SWE を端末化 + 140k=合成 Terminal-Task-Gen。
- **SWE-bench/SWE-smith-trajectories**(4.0GB, MIT)= SWE-agent-LM-32B(40.2%)の SFT データ。test-pass 軌跡のみ、`resolved`+`patch` 付き(→ **GRPO の oracle patch 源**)。
- **nebius/SWE-rebench-openhands-trajectories**(2.0GB, cc-by-4.0)= Qwen3-Coder 生成・fresh(汚染低)。
- nvidia/Nemotron-Terminal-Synthetic-Tasks(984MB)/ SWE-Gym/OpenHands-SFT-Trajectories(小・高信号)。

**🔴 decontam ゲート(訓練前・非交渉)**:
- 標的 = Terminal-Bench 2.1 タスク集 + SWE-bench Verified instance/repo リスト。
- 手法(SWE-Bench Illusion 2506.12286 / SWE-rebench 2505.20411 に準拠): (1) task-id / repo 完全一致除去 (2) 問題文 n-gram 重複除去 (3) path-from-issue プローブで leak 検査 (4) 可能なら post-cutoff/fresh 優先。
- 特に Terminal-Corpus の 226k 適応ストリーム・SWE 軌跡の repo 重複を重点。除去件数を記録。

---

## 3. Eval(物差し)

- **Terminal-Bench 2.1** = 主。harness=**Harbor**(`uv tool install harbor`)、agent=terminus-2(OpenAI 互換で Qwen3.6 を差す)。採点=resolution rate、pass@k(論文 k≥5)。**ローカル機で回す**(Docker 必須、§4)。
- baseline を最初に取る(素の Qwen3.6)= 現在地。
- 副: SWE-bench(可能なら)、HumanEval/MBPP(静的回帰)、MMLU/JMMLU(汎用非劣化 −1pt ゲート)。
- **cross-domain 回帰行列**: 各 patch を全ベンチで測り偏りを検査。
- `<think>` パーサ罠を 1-3 タスクで先に確認(壊れると「モデル弱い」と誤読)。

---

## 4. アーキテクチャ(train remote / eval local)

| マシン | 役割 | 理由 |
|---|---|---|
| **aux-host**(借り物 2×RTX PRO 6000, `ssh aux-host`) | 訓練(ESFT SFT + GRPO) | GPU + ESFT パイプライン。**Docker 無し・vLLM 壊れ** |
| **ローカル(D1/F1)** | Terminal-Bench eval | Docker あり + Qwen3.6-vLLM serve 済 |

橋渡し = 訓練済み patch(~4GB delta)だけ Tailscale 転送。ローカルは 70GB base を保有。**ローカルは指示があるまで触らない(ユーザ作業中)。**

---

## 5. 段階ロードマップ(10-14日)

```
Phase 0 [進行中] coding SFT pilot (aux-host, 500step)   ← パイプライン実証。math は pilot 完了
Phase 1 agentic SFT: Terminal-Corpus(decontam済)@k=32 + SWE軌跡   ← 本命 SFT
        └ 並行: ローカルで Terminal-Bench baseline(素の Qwen3.6)
        └ 完了→ patch 転送→ ローカルで Terminal-Bench 再測 = SFT の効き
Phase 2 [INC-0] rejection-sampling FT: 自己生成→高報酬選抜→SFT   ← GRPO 前の最安ゲート
        └ 伸びれば GRPO へ投資、伸びなきゃ地図が1本正確に
Phase 3 GRPO × ルールベース報酬(SWE-RL)   ← 本 RL、§6
Phase 4 構造仮説の決着: k=8 vs k=32 control + (option)co-activation   ← 「active増→agentic」を測る
Phase 5 出荷: NVFP4 焼き(≥99% fidelity, 実ベンチ exact)+ cross-domain回帰 + release packaging
        └ redo バッファ(~2日)
```

---

## 6. GRPO × ルールベース報酬(L3)— 詳細

**なぜこの形か**: フル実行報酬 RL は Docker 実行環境が要る → 借り物に無い。SWE-RL(2502.18449, 実測 41.0% SWE-bench)は**報酬を「生成 patch と正解 patch の類似度」にして実行を回避**した。純 Python 計算 → **Docker 不要で借り物 GPU 上で回る。** これが「借り物で RL 可能」の正体。

- **アルゴリズム**: GRPO(1プロンプトに N rollout 生成 → 各報酬 → group 内正規化 → policy gradient)。framework 候補 = TRL GRPOTrainer / veRL。
- **報酬(ルールベース)**: 生成 patch/solution と oracle の類似度(SWE-RL: sequence similarity。拡張案: AST/変更ファイル一致で reward hacking を抑制)。
- **データ**: oracle patch を持つ SWE-smith/SWE-Gym/SWE-rebench + Terminal-Corpus の oracle 軌跡。
- **policy**: Phase 1 の ESFT@k=32 SFT モデル(RL は base の上で効く、弱い base に RL は薄い)。
- **律速リスク = rollout 生成スループット**(vLLM 壊れ。GRPO は 1問×8-16サンプル×多ステップ生成。serving を要解決 or veRL 内蔵生成)。
- **reward hacking ガード**: (a) 人の目視レビュー(rollout を数十件、"本当に解いてるか")(b) 補助チェック(patch が apply するか・正しいファイルを触るか)。
- **INC-0(Phase 2)を先に**: 完全な GRPO ループを数日かけて建てる前に rejection-sampling FT(既存 SFT パイプラインで回る)で「自己生成の高報酬データが伸ばすか」を安く確認。

---

## 7. 正直な EV / リスク

- **構造仮説(k=32>k=8)**: conditional。phase 4 の control で Terminal-Bench 上で測る。効けば「4倍容量を訓練で使い切る」実証。
- **baseline 控えめ**(一桁〜15%)= 出発点。
- **RL の律速 = serving スループット**(最大の技術リスク)。
- **汚染 = 整合性ゲート**(割ったら全て leakage 疑い)。
- **NVFP4 fidelity = 出荷側リスク**(MoE fp4 焼きは scale 破壊の前科)。
- **速度税は無問題(ユーザ明言)**→ k=24 退路不要、k を 50(A9B天井)方向へ上げる余地も生きる。

---

## 8. 分業(人 vs 機械)

| 人(ユーザ, ~12h + ローカル機) | 機械(俺 + aux-host) |
|---|---|
| ローカル Terminal-Bench eval 立ち上げ(**君のマシン必須**) | 訓練パイプライン(SFT/GRPO)、データ prep |
| rollout の reward-hacking 目視レビュー(人の判断が効く) | decontam ゲート実装、報酬関数、GRPO ループ配線 |
| golden seed / eval セット吟味 | Terminal-Corpus → messages 変換、profiling→config |

**手作業ラベリングは律速でない**(報酬は自動)。人手は「マシン必須」「人の判断が効く」所へ。

---

## 9. Phase 1 詳細仕様(実データ実測で詰め、2026-07-04)

### 9.1 データ実測(aux-host `~/esft/data/hf/`、tokenizer=Qwen3.6 で計測、n=60/set)

| dataset | 行数 | tok/traj p50 | p90 | max | turns p50 | role 構造 | 形式 |
|---|---|---|---|---|---|---|---|
| Terminal-Corpus/code | 31,960 | 19,606 | 25,815 | 41,052 | 14 | user/assistant | **terminus-2**(JSON action) |
| Terminal-Corpus/swe | 31,661 | 28,287 | 52,216 | 61,018 | 26 | user/assistant | **terminus-2** |
| SWE-smith | 3,260×8 | 23,656 | 60,313 | 118,137 | 53 | system/user/assistant | OpenHands(bash tool) |
| SWE-rebench | 67,074 | 38,650 | 56,496 | 74,317 | 125 | +**tool** role | OpenHands |

- Terminal-Corpus: teacher=DeepSeek-V3.2、agent=**terminus-2**(= eval harness と同一形式)、観測は user turn に畳まれる。
- SWE-smith: `patch`(oracle diff)+ `resolved=True` → **GRPO oracle 源**。SWE-rebench: `model_patch` + 品質フラグ(`pred_passes_gen_tests`)。

### 9.2 決定事項(第一原理)

- **D1 形式 primary = Terminal-Corpus(terminus-2)。** eval を terminus-2 harness で回す→ action 方言を揃える=clean transfer。SWE系(OpenHands)= GRPO oracle + 二次。方言混在は eval で誤形式吐くリスク。
- **D2 seq_length = 24576、1軌跡=1サンプル(cross-traj packing 禁止)。** 観測(user/tool)mask、assistant のみ supervise。code p50=19.6k は収まる、swe p50=28k は大半収まる。`>24k` の tail は truncate、**drop 件数を必ず log**(silent cap 禁止)。理由: 2048 では act→observe→adapt の弧が1つも context に入らず、教えたいループを破壊する。
- **D3 expert 選抜 = agentic トークンで再 profile。** coding 選抜(781 experts)は流用しない。collect_router_stats --top-k 32 を Terminal-Corpus サンプルで回し config 生成。agentic は別の expert を叩く。
- **D4 optimizer = Adafactor(baseline)vs Muon(A/B)。** 同一 delta config で eval_loss + peak mem + wall 比較。Muon = momentum のみ 4B/param(AdamW の半分、OOM した 19.7GB→~9.8GB で入る見込み)、行列専用でうちの delta にど真ん中。~2x は pretraining 値で SFT-on-delta 未実証→期待値は小幅、要 DDP 対応実装(Moonshot ZeRO-1 版流用、~半日)。critical path 外。
- **D5 chat template = terminus-2 wire 形式を厳密再現。** harness の prompt 構築と一致を smoke で確認(不一致だと「モデル弱い」と誤読)。

### 9.3 decontam ゲート(非交渉、訓練前)

- **標的リスト**: princeton-nlp/SWE-bench_Verified(instance_id + repo)、Terminal-Bench 2.1 task registry(terminal-bench GitHub / Harbor)。
- **手法**: (1) repo/instance 完全一致 drop(Terminal-Corpus/swe `task`・SWE-rebench `repo`+`instance_id`・SWE-smith repo)(2) task 文 13-gram 重複 drop (3) path-from-issue プローブを標本で leak 検査 (4) Terminal-Corpus/code の `source`(OpenCodeReasoning 等)が Terminal-Bench 派生でないか確認。**source 別 drop 件数を全記録。**

### 9.4 ゲート(大工事前の最安チェック)

- **INC-1(VRAM)**: seq=24k で 5 step、peak mem 実測。OOM なら 16k へ or checkpointing 強化。**フル run 前に必須**(serve GPU テストが窓を空けた後)。
- **INC-2(形式 smoke)**: 1-3 軌跡 → ローカル terminus-2 harness で eval → `<think>`/形式パース破綻がないか。

### 9.5 実行順序

1. decontam ゲート(CPU、serve と並行可)→ clean set + drop log
2. serve GPU テスト完了 → router profiling(agentic)→ config 生成
3. INC-1 VRAM ゲート(seq=24k)
4. 変換(conversations→訓練形式 + mask 仕様)
5. 訓練(Adafactor)+ Muon A/B
6. patch → ローカル転送 → Terminal-Bench(**baseline 先取り**→ 効き測定)

### 9.6 VRAM 壁の実測 = CE loss logits(2026-07-04、traceback で確定・要訂正)

- **真因(traceback 確定)**: 7.58GB OOM は **`transformers/loss/loss_utils.py:67 ForCausalLMLoss`** = cross-entropy の **logits materialization**([seq × vocab≈152k]、fp32 upcast + softmax 中間)。seq8192: 8192×152k×4≈5GB ×~1.5 ≈ 7.58GB。**壁は attention でも linear-attn でもない、loss。** seq が上がると logits が線形に膨らむ。
- **⚠️ fla 誤帰属を訂正(盛りの反省)**: 最初 fla の "fast path not available" 警告を見て linear-attn fallback と早合点したが、traceback を読めば loss。**`fla` は的外れ、かつ SM120 で dead lever** ── fla gated_delta_rule の tilelang backend が `#include <tl_templates/cuda/instruction/mma.h>` でコンパイル失敗、**4096(fla無しでは通る)すら crash させた**。→ fla + causal-conv1d は **uninstall 済**、torch fallback に復帰。
- **真の fix = Liger fused linear cross-entropy**。`liger-kernel==0.8.0` 導入済、**`LigerQwen3MoeSwiGLUMLP` あり=Qwen3-MoE 対応**。FLCE は full logits を materialize せず LM-head+CE を融合 → loss メモリが seq ほぼ非依存 → **seq 16k-32k 解禁見込み**。統合は train_esft.py 手術(loss 差し替え)要 = **明日 attended**(無人で壊さない)。
- **第2レバー**: per-GPU baseline が **~88GB**(device_map="auto" を rank ごとにロード)。FSDP or 1-GPU-per-rank で下げれば更に seq 余地。
- **今夜**: CE 律速で seq ~4096-6144、確実な run を確保(format+短horizon agentic の pipeline 実証+評価可能 patch)。長コンテキスト本命は Liger 統合後。

---

## 参照文献(全て arxiv 実在確認済、ハルシネ無し)
2602.21193 NVIDIA Terminal-Corpus / 2601.11868 Terminal-Bench / 2412.21139 SWE-Gym / 2504.21798 SWE-smith /
2505.20411 SWE-rebench / 2506.12286 SWE-Bench Illusion / 2502.18449 SWE-RL / 2603.21357 AgentHER /
2509.21892 EMoE / 2508.18672 Optimal Sparsity / 2407.01906 ESFT。
