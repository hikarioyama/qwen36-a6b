# 35B-A6B 最終目標とやることリスト (living document)

更新: 2026-07-11 15:30 JST(状況が変わり次第このファイルを書き換える。最新の判断はここを正とする)

## 究極の目標

**35B-A6B を、ツールコール・コーディング・一貫性・日本語の4軸で、ベースモデルから有意に強化する。**

- 「有意」= 同一問題・同一seed の paired 比較で CI95 が 0 を跨がない改善。盛りなし、`(n=?, same-condition?)` を常に添える。
- 一般能力(MMLU/GSM8K 等)は非劣化ゲートで守る。4軸を上げて一般を落とすのは不採用。

## 現在地 (2026-07-11 午後更新)

- **B2 系列は死亡確定**(ユーザー確認済み)。4軸プロファイル: ツールコール +4.0pt 有意↑ / 指示追従 −13.7pt 有意↓ / MMLU 有意↓。死因 = router 凍結 + k32。
- **joint grad ゲート v3 PASS (GPU 実機)**: router union 40/40、expert 80/80、attn/embed 凍結クリーン、router LR 8e-7、fresh vs resume digest 全 8 rank bit 一致。→ **200-step probe 発射済み (2026-07-11 午後、gpu-host、決定論 on、eval-steps 50 / save-steps 100、最後に HF full model export)**。完了後は kill 基準表 (DEVLOG 2026-07-08(4)) で判定。
- **データ戦略確定**(ユーザー承認): ①足場付き自己生成 + 機械選別(生成=ローカル、各軸の合格判定器が本体、1周のみ、eval 汚染除去必須)+ ②外部の検証済みデータを混合。**訓練は gpu-host、データ生成はローカル**。selfgen pilot500 はローカル GPU0/1 で生成走行中。外部主力候補 Toucan-1.5M + ToolMind は vault へ DL 中。

## リソース方針 (2026-07-10 夜、ユーザー指示)

- **GPU はできるだけ余らせない。空いたら究極目標に寄与する作業を即割り当てる。**
- aux-host は当面使用不可。使えるのは **gpu-host(8×96GB)とホームラボ(RTX PRO 6000 ×2)** のみ。GPU 2(5070Ti)は表示用で計算に使わない。
- **恒久分担 (07-11 確定): ローカル = データ生成工場 + eval 計測室 / gpu-host = 訓練炉。**
- 現在の割当 (07-11 午後): gpu-host 8GPU = **200-step joint probe 走行中** / ローカル GPU 0,1 = **selfgen pilot500 生成走行中** / ネットワーク = Toucan-1.5M (21.8GB) + ToolMind (4.0GB) を /mnt/vault/corpora/ へ DL 中。
- **決定論 env 速度コスト実測済み**: overhead **+8.9%**(det on 61.96 vs off 56.87 s/it、n=2/腕、ABBA、same-condition)。200-step は on 推奨(+17分)、本走の on/off は選択肢3案でユーザー判断 → `reports/FULLFFN_DET_SPEED_AB_20260711.md`。

## やることリスト(優先順)

### T1. B2-750 評価 【完了 — B2 系列は採用候補から除外を推奨(最終確認はユーザー)】
- 結果 (`reports/B2_750_EVAL_20260710.md`、n=600/600/164 paired、同一protocol): MMLU **−2.50pt [−4.53,−0.47] McNemar p=0.024 で有意悪化**、GSM8K +0.33pt 未確定、HumanEval −1.22pt 未確定。B2-1000 は 750 より全項目で数値上勝るが、750↔1000 の直接 paired 差は全て未確定。
- 事前ルール適用: B2-750 も非劣化ゲート不成立(MMLU は有意にマイナス)。**B2 系列(v3 + forward KL β=0.5、teacher k8 top-64、router凍結)は採用候補から除外を推奨**。checkpoint 750/1000 は保全、recipe と死因は DEVLOG 記録。
- 死因の読み(mechanism): 内部 CE loss は 500→1000 で単調改善したのに能力ベンチは base 以下のまま(MMLU で有意悪化、JMMLU パイロットでも負側) — 「router 凍結に対する expert 最適化は間違った丘」という 2026-07-08 Grok 診断と整合する負の結果。次の登り口は router 可動 joint(T2/T3 の Full-FFN 路線)。

### T2. Full-FFN 決定論 probe arm B 【完了 — GREEN】
- 検分の結果、**arm B は 2026-07-10 05:31–05:38 UTC に実行済みで GREEN** と判明(前セッションが実行、DEVLOG 反映前に引き継ぎ境界を跨いだ)。Codex の「SSH BLOCKED」は新規再実行の試行が塞がれただけで、実体は完了済み。
- 6段階アサート全 MATCH: load_model / load_optimizer / RNG / batch_loss(32) / clip後 gradient / post_optimizer(各8 rank)。決定論設定は per-rank ログで確認。証跡と解釈は `reports/FULLFFN_DETERMINISM_ARMB_20260710.md`。
- 結論: bit 不一致の発生源は grouped-mm backward / NCCL reduction の再起動間非決定性で確定し、決定論 env で消える。**exact-resume ゲート GREEN、200-step 本番の技術的ブロッカーなし**。
- 速度コスト計測済み (+8.9%)、200-step はユーザー GO 済みで**発射済み** → T6 へ。

### T3. Full-FFN joint trainer 配備 【完了 — grad ゲート v3 PASS】
- trainer 改修 (二重 opt-in の joint、router LR group、anchor KL forward hook 方式) を gpu-host に配備。gradgate v3 で GPU 実機 PASS: router union 40/40 / expert 80/80 / attn/embed 凍結クリーン / fresh vs resume digest 全 8 rank bit 一致。詳細 `reports/FULLFFN_JOINT_TRAINER_20260711.md` + DEVLOG 2026-07-11 午後。

### T6. 200-step joint probe の完了検分と paired 評価 【走行中 — 最優先】
- 走行構成: gradgate fresh 段と同一条件 + max-steps 200 / eval-steps 50 / save-steps 100 / 決定論 on / FULLFFN_PROBE off。出力 `gpu-host:codex_runs/fullffn_joint_200step_20260711/`、完了マーカー `JOINT_200STEP_DONE`。
- 完了検分: eval_loss 軌跡 (50/100/150/200)、checkpoint-100/200 の実在、HF full model export の完全性。
- 評価: HF export をローカルへ転送 → 4軸 + MMLU/GSM8K の paired 評価 (k32 と k8 の両方)。
- **kill 基準表 (DEVLOG 2026-07-08(4))**: k32 MMLU ≥0.832 かつ k8 劣化 ≤0.8pt → 4a 続行 / k32 MMLU <0.825 → 4b full-FFN 増強 / k8 劣化 >1.5pt → router 再凍結。

### T7. selfgen pilot500 の検分と量産判断 【生成走行中、ローカル GPU0/1】
- 完走後: `reports/SELFGEN_TOOLCALL_V1_20260711.md` 更新、採用/棄却例の品質検分 (accepted/rejected/採用率は完走後にのみ記録)、良ければ量産 (n=5000 級) 起動。
- 次軸のパイプライン (日本語 verifiable 指示、一貫性) は pilot 検分後に設計。

### T8. 外部コーパスの取得と汚染除去 【DL 走行中 → vault】
- Toucan-1.5M (21.8GB) + ToolMind (4.0GB) → `/mnt/vault/corpora/`。DL 後: eval セット (MMLU/GSM8K/HumanEval/JMMLU/BFCL/M-IFEval) との n-gram 汚染照合 + 除去ログ manifest 化。
- 注意: ToolMind open_datasets には APIGen-MT (CC-BY-NC) 由来ファイルあり — B 群分離規律の対象。

### T4. 4軸評価ハーネスの整備 【バックログ、着手前にユーザー相談】
- 究極目標の4軸(ツールコール / コーディング / 一貫性 / 日本語)のうち、いま paired で測れるのはコーディング(HumanEval)だけ。残り3軸の評価セット選定が未着手:
  - ツールコール: BFCL 系 or 内製 agentic セット
  - 日本語: JMMLU / JHumanEval 等
  - 一貫性: 定義から要検討(長系列 self-consistency? persona 維持?)
- これは評価protocol の新規凍結を伴うので、**セット選定はユーザーと決めてから**。夜間は候補調査(read-only)まで。
- **候補調査完了 (2026-07-10 夜)**: Grok 調査 + Opus 2体の一次ソース検証済み。結論とライセンス注意点・未決3点は `reports/EVAL_4AXIS_CANDIDATES_20260710.md`。推奨: BFCL v4 非live + ACEBench / JMMLU + llm-jp-eval 決定的部分 / M-IFEval(日) seed 分散 + paraphrase 一致率。
- **BFCL パイロット再開・完了 (2026-07-10)**: pinned Gorilla + ローカル `bfcl-eval` wheel、公式 AST parser/checker、Qwen native tool template adapter で、GPU 0/1 のみ・base→B2直列を実測。非live / 非external-API の shuffle seed 0 `n=300` で base@k8 0.7900 (237/300, trunc 1)、B2-1000@k32 0.8300 (249/300, trunc 0)、paired Δ `+0.0400`、CI95 `[+0.0111,+0.0689]`、McNemar `p=0.01182`。ツールコール軸への正の pilot evidence だが、protocol 未凍結・B2採用根拠ではない。pinned checker が selected Java/JavaScript 40件を `KeyError('string')` で両armとも0点化する既知の上流不整合を検出したため、cross-language 結論は禁止。将来は上流修正版をpinして再走、または Python-only subset を事前固定して再走するかをユーザーと決める。詳細は `reports/BFCL_PILOT_20260710.md`。
- **M-IFEval(日) seed 分散 pilot 完走 (2026-07-11 v5)**: 依存解消後、v4 が launcher 終了で子プロセスごと停止 → v5 を detached (nohup) で再起動し完走。base@k8 pass 0.5628 vs B2-1000@k32 0.4256、**paired Δ−13.7pt CI95 [−22.9,−4.6] で有意悪化**。seed 間一致率は差なし (Δ−0.35pt ns)、B2 の pass rate seed SD は base の3.7倍。protocol 未凍結・採用判定なし。詳細 `reports/MIFEVAL_PILOT_20260711.md`。
- **B2-1000 の4軸パイロット総覧が完成** (全て paired vs 真stock、protocol 未凍結): ツールコール **+4.0pt 有意↑** / コーディング −3.7pt ns / 日本語知識 (JMMLU) −1.7pt ns / 日本語指示追従 (M-IFEval) **−13.7pt 有意↓**。一般 MMLU も B2-750 で有意↓。→ **B2 recipe は標的コーパスの軸だけ効き、他を広く壊す** — B2 系列除外の推奨を補強する決定的なプロファイル。

### T5. 日本語軸 JMMLU パイロット 【完了、n=300 paired】
- JMMLU を既存 paired ハーネスに組み込み、真stock base@k8 vs B2-1000@k32 をローカルGPU 0/1でbase→B2直列に測定。base 0.7500、B2 0.7333、Δ −0.0167、paired CI95 [−0.0451,+0.0117]、McNemar p=0.3593、両arm truncated=0。
- 日本語軸の差は未解決で、margin 0.02のnon-inferiorityもINCONCLUSIVE。目的(a)のハーネスbring-upは達成したが、(b)は補助材料に留まり、protocol未凍結・採用判定には使わない。
- report: `reports/JMMLU_PILOT_20260710.md`

### 夜間の自走範囲(明示)
- やる: T1 の監視・検分・report 検証、T2 の実行と判定、本ドキュメントと DEVLOG の更新、T4 の候補調査(read-only)。
- やらない: モデル採用判定の確定、追加訓練の開始、Full-FFN 200-step/本走、checkpoint の削除・上書き、git reset/clean。

## ガードレール(常設)
- 数字は同一条件 paired + CI95 のみ信用。単発・条件違いは参考値扱い。
- 負の結果も DEVLOG に死因つきで記録(実験データ削除禁止)。
- gpu-host/aux-host のジョブは runner 最終 marker + 全成果物で完了判定(`set -o pipefail`)。
