# GRPO × delta-ESFT 設計(v1、2026-07-05)

## 目的
Phase 1 SFT patch(agentic_esft_k32)を出発 policy に、SWE-RL 方式ルールベース報酬で GRPO。
訓練対象は expert delta(2.46B)のみ、router/その他は凍結(SFT と同一枠組み)。

## 制約(実測に基づく)
- aux-host 2×RTX PRO 6000(各96GB)。**訓練だけで 91/96GB**(seq7168 DDP)→ vLLM server と訓練の同居は不可能。
- vLLM serve は SM120 修理中(経路a: 新wheel / b: source build)。
- Docker 無し → 実行報酬不可 → SWE-RL 類似度報酬(確定済)。

## アーキテクチャ: 交互フェーズ(iterative GRPO)
GPU を時分割する。TRL/veRL は使わず、train_esft.py の部品を再利用した自前ループ。
理由: (1) 同居不可なので TRL colocate/server 方式の前提が崩れてる
(2) packed 3D expert + delta hook + router top_k override は TRL の重み同期と噛み合わない
(3) 自前なら delta トグルで KL 参照がタダ(下記)。

```
loop cycle c = 1..C:
  [G] merge(base + delta_c) → ckpt_c を vLLM serve(TP2, k=32 override)
      prompt batch P_c(~256-512問)× N=8 rollout 生成(temp~1.0, OpenAI API)
      reward.py で報酬 → group 正規化 advantage
      (vLLM の logprobs も保存: 監視用。1-step 更新なら ratio 不要=on-policy)
  [T] server 停止 → 訓練プロセス起動(DDP 2GPU)
      completion トークンの logp を現 policy(delta ON)で計算
      loss = -(adv * logp_sum) + β·KL(policy ‖ base)
      ★KL の base logp は **delta を一時無効化した同一モデル**で計算
        (delta 方式の構造的特典: 参照モデルの複製 70GB が不要)
      1-2 epoch だけ回して delta_{c+1} 保存
  [M] merge → ckpt_{c+1}、cycle 諸指標を log(報酬分布、format-fail率、KL)
```

- cycle 切替オーバーヘッド: vLLM 起動 ~5-10min + merge ~数min。
  gen が数十min〜h スケールなので税率は低い(実測で確認)。
- 逐次改善: vLLM sleep/wake(0.24+ の sleep_mode)が使えれば reload 税を削れる。
  ただし delta→重み更新の in-place 反映(collective_rpc)は v2 の最適化、v1 はフル reload。

## 訓練側の詳細
- micro-batch: 1 completion = 1 サンプル(seq = prompt+completion ≤ 24k 想定、
  FLCE 統合済みなので logits 壁なし。DDP 上限 seq7168 の制約は **logp 計算にも効く**
  → prompt+rollout が 7168 超の場合の扱い: v1 は truncate して件数 log、
  v2 で単一GPU logp pass(12288 まで)or FSDP を検討)
- optimizer: Adafactor(SFT と同じ、state ~0)
- advantage: group 内 (r - mean)/std、std=0 group(全同報酬)は skip(信号なし)
- format-fail -1 が group 全滅させる問題: SEARCH/REPLACE 形式で合格率を確保(reward 側で対応済)
  + 全 -1 group は skip 対象に自然に入る

## INC-0(先行ゲート、GRPO ループ実装前に回す)
rejection-sampling FT: ckpt_1(=SFT patch)で N=8 生成 → 報酬 top-1(かつ similarity>閾値)
を SFT データ化 → **既存 train_esft.py でそのまま焼く**(新規実装ほぼゼロ)。
これで「自己生成+報酬選抜データが効くか」を1晩で判定してから GRPO 本体に投資。
INC-0 の rollout 生成・報酬計算は GRPO と同一部品=捨て仕事なし。

## 壁の式(rollout スループット律速)
cycle_gen_time ≈ P×N×tok_per_rollout / serve_throughput
例: 384問×8×~2k tok ÷ (35B-A3B@k32 TP2 の aggregate ~1-2k tok/s 仮定) ≈ 1-1.7h/cycle
→ C=10 cycle で ~15-20h + 訓練時間。serve throughput の実測が最初の仕事(修理完了後)。
数字は全部 hypothesized、serve 実測で更新する。

## リスク台帳
1. k=32 override が vLLM の hf-overrides で rollout 時も効いてること(訓練と同条件)を必ず検証
   (ローカル FP8 smoke では反映確認済み。aux-host の修理後 build でも再確認)
2. 報酬ハッキング: 目視レビュー(ユーザ担当)+ files_jaccard/applies の分布監視
3. KL β: SWE-RL は明示 KL なし(clip のみ)。v1 は β 小さめ(0.01-0.05)から、
   reward 上がって MMLU 落ちるなら β 引き上げ
4. chat template: rollout 生成と logp 計算で byte 一致必須(過去の教訓: train/serve 表現一致)

## 追記 2026-07-06: 形式プローブの確定知見(GRPO 設計に直結)
- **開始 `<think>` タグ問題**: Qwen3.6 の chat template は generation prompt 側で `<think>\n` を供給する(モデルは開始タグを「もらう」側で、自分では出さない)。SWE-RL verbatim の strict parser は completion 内に `<think>...</think>` を要求するため、素の serve 出力は構造的に fmt_ok=0 になる(near-miss 50% = 開始タグ以外は完成、の正体)。
- **対策 = assistant prefill**(`<think>\n` を continue_final_message で供給): fmt_ok 10.9% / 真封筒 9.4% / trunc 6% / paired Δlenient **+0.394 CI[+0.148,+0.647]**(n=64、vs P)。形式を強制しても中身は落ちず、むしろ CI-clean で改善。
- **GRPO 実装への要求**: rollout 生成と報酬の間で prefill を必ず再結合してから score_record に渡す(full assistant = prefill + completion)。学習ターゲットの token 化でも prefill 分を含める。max_tokens は 10000(4096 は trunc 支配)。
- **形式税の直接証拠**: getmoto pr_4860 = lenient_best 1.000 で有効封筒 0 本(直せるのに包装できない)。能力でなく形式が律速。
- **保留レバー**: reward.py の empty-thought 却下を緩めれば fmt_ok +~15% だが、swe-rl verbatim 原則から逸れるため未適用(封筒 SFT 後は非空 think が焼けてる想定なので不要になる見込み)。
