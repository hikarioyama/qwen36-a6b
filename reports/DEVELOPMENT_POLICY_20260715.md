# モデル開発方針 — 35B-A6B キャンペーン (2026-07-15 版)

> 正本は `reports/ROADMAP.md` (生きた方針書) と `DEVLOG.md` (実測の正)。
> この文書はその要約スナップショット。矛盾があれば ROADMAP / DEVLOG が勝つ。

## 1. 何を作っているか (上位目標)

Qwen3.6-35B-A3B (MoE、通常は各トークンで expert を 8 個使う) を、**expert を 32 個使う設定 (k=32) に拡張した「35B-A6B」**に育てる。

- 手法の柱: **router 凍結** (expert の選び方は変えない) + **α ダイヤル** (増えた expert の寄与を推論時に較正する係数) + **full-FFN 訓練**
- 勝利条件: 4 軸 (ツールコール / コーディング / 一貫性 / 日本語) で base@k8 を **同一条件 paired 測定で有意に**上回る。BFCL (ツールコールの外部ベンチ) 非劣化が hard gate。

## 2. これまでに確定したこと (measured)

| 事実 | 数字 (n, 条件) |
|---|---|
| 借金完済: k=32 でも base@k8 と同等まで回復 | MMLU 84.83 vs 84.67 (n=600 paired, ns) |
| v4 燃料は BFCL を壊していた | base 84/100 → tail05 53/100 (**−31pt**, paired McNemar p=1.2e-07) |
| 壊した機構 = 関数名の複写忠実度の喪失 | mock_* テンプレ名への過適合。k32/ダイヤル/patch 方式は無罪 (B2 88/100 で確認) |
| 能力の崖は T3→T4 の間 | seed 採用率 T1 97% / T2 87% / T3 90% / **T4 26-28%** (3 run 再現) |
| probe のバグ修正で mock名 T4 は復活 | intent_r1 T4 **14/30=47%** (extend prior, n=30, best_of=4, arm C) |
| diverse名 seeds は T4 自己生成が全滅 | desc2 T4 **0/30** (stage1 で 26/30 撃沈, 同条件) |
| **description は無罪 (2026-07-15 新規)** | desc無し 0/30 (stage_fail={1:21,0:9}) vs desc有り 0/30 ({1:22,0:8}) — 同型で壊滅 |
| **壊滅の主モード = stage0 の再出力 (新規)** | 解剖 3/3 seed で stage1 以降に stage0 の calls をそのまま再出力。副モード = 同型 arg 間の値入替 (greedy, teacher-forced 解剖) |

## 3. 設計の緊張と決着 (2026-07-15 A/B 11 本 + 多角診断で確定)

```
mock名   = base が自己生成できる (43-47%、few-shot 併用 87%) が、訓練すると BFCL を壊す (−31pt)
diverse名 = 訓練は守れる見込みだが、base が自己生成できない (全 variant 0-3%)
```

**A/B 消去法の全記録** (各 n=30, best_of=4, extend prior):

| desc2 への変更 | T4 | 学び |
|---|---|---|
| description on/off | 0/30 both | desc 無罪 |
| tool 名のみ mock 序数化 | 0/30 | 名前の形は単独無罪 |
| request に tool 名 anchor 注入 | 0/30 | anchor 単独無罪 |
| transcription (anchor+arg=値 全部入り) | 0/30 (stage0 は 29/30 に改善) | 様式単独では stage1 崩壊が残る |
| receipt 名の自己記述化 (±transcription) | 0/30 both | 単独無罪 |
| **continuation few-shot 注入** | 1/30、**r1 に足すと 13/30→26/30 (87%)** | **C(進行) の主犯は single-turn only few-shot** |
| few-shot+receipt 名 | 0/30 (stage1 突破 0→7/30) | 前進するが不足 |
| few-shot+全命名 r1-parity 化 | 0/30 (stage0 悪化 5→15) | **request 文の骨格が最後の因子** |

**機構の結論** (解剖 9/9 が stage0 の byte 同一再出力、cx 積モデル P(T4)≈C×B×R×L と整合):
失敗は単一犯でなく **C(stage 進行プロトコル) × B(値→arg binding) × R(receipt 識別) の積**。全項を同時に満たすのは現状 r1 完全型 (命名+request 骨格+few-shot) のみ。

**採択戦略 = 生成/訓練の surface 分離** (Hammer arXiv:2410.04587 の function masking と同向き):
- 生成時: seeds を **全単射リネーム**で r1-parity surface (mock 序数名 + r1 骨格 request) に変換して rollout → 高収率
- 訓練時: 完成 trace を**逆写像で diverse 名 + natural paraphrase に復元** → BFCL 保護
- 健全性の根拠: 採択ゲート = expected との exact-match なので、復元後の trace は gold と同一 = 機械検証済み。request 文は値のみでツール名を含まないため復元で整合が壊れない
- 留意: rejection sampling の意味は「base が (mock surface で) 解ける seed の能力フィルタ」に変わる。訓練 surface は生成 surface より難しい — gold 検証済みなので許容と判断 (代替案: gold 直接レンダ = GPU ゼロで 100% 収率だが self-gen 性が完全に失われる。ユーザ判断事項として保留)
- 外部知見 (Grok): 値の羅列 request は主要パイプライン (APIGen/ToolBench/ToolACE/Hammer/BFCL) に存在しない異形。explicit-binding と paraphrase の混合が業界既定

## 4. 燃料再製造の手順 (質最優先・フォールバック禁止)

1. ~~T4 壊滅原因の切り分け~~ → **完了 (§3)**
2. ~~go/no-go ゲート~~ → **PASS: T4 93% (28/30 ローカル) / 97% (29/30 gpu-host)** (n=30×2 独立, 2026-07-15)。
   注: リネーム系 probe には receipt 期待値の再計算漏れによる偽陰性バグがあった (DEVLOG 参照)。recompute_derived で修正済み
3. 全単射リネームツール `esft/selfgen_name_bijection.py` (mockize/restore + round-trip テスト) — Codex 実装中
4. **本走行** (全 5,000 seed の rejection sampling, gpu-host :8199): mockize 済み seeds で発射
   - 発射前に不良 generation_records 消去 + **env 3点セット必須** (`SELFGEN_NOTHINK=1 SELFGEN_STAGE_HINT=1 SELFGEN_MT_FEWSHOT=1`)
5. **restore** (逆写像で diverse 名 + natural paraphrase に復元) → train.jsonl
6. G2 プリフライト (render errors 0 + 汚染監査 + sha + vault 保存) → 搬入
7. **fresh 再訓練**: stock base + router 凍結 + α=0.5 (汚染 checkpoint 系譜は使わない) — gpu-host、rollout 完了後に切替
8. 境界測定 (3 軸 + BFCL paired、BFCL 非劣化 hard gate) → HF 公開 (識別子 sanitize + 敵対 sweep 後)

## 5. その先の分岐

- **傾きが出た** → 現路線増強 (intent selfgen 量産、ja v2.1、300-step ごと燃料更新)
- **傾きゼロ** → SFT で積む仮説を疑い、**RL へ重心移動** (GRPO 自前実装、verifiable reward = 機械検証の合否 − λ×思考長)。Molt は棚上げ (監査済: 自前が安い)
- ThinkingCap 路線: プローブ (陽性済) → INC-0 (データ 925 行確保済、焼きは後回し) → GRPO の三段ゲート
- budget forcing は訓練と独立の出荷ノブとして標準装備

## 6. 変わらない運用原則

- 訓練量最優先・炉の空白ゼロ。測定は区間境界で 1 回、eval はローカル固定
- 同一条件 paired (n 明記) 以外の比較は信じない。盛らない — 数字には必ず (n=?, 条件) を添える
- 汚染ゲート (逐語 + BFCL 8-gram/関数名照合) は全数維持。系譜ポリシー B (GPT/Grok 製テキスト可) でも緩めない
- 実験データ削除禁止 (失敗 weights は recipe 化してから削除可)
- 公開物は実機名禁止 (gpu-host/aux-host 表記)・noreply author・敵対 sweep 必須
