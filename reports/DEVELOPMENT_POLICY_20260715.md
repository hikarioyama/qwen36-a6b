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

## 3. いま直面している設計の緊張 (本丸)

```
mock名   = base が自己生成できる (47%) が、訓練すると BFCL を壊す (−31pt)
diverse名 = 訓練は守れる見込みだが、base が自己生成できない (0%)
```

**中間解を探すのが現在の主戦線。** 有力な機構仮説 (hypothesized):

- intent_r1 の mock名は `inspect/list/reserve` という**動詞 + 序数 (_1.._5)** を持ち、user_request もその動詞で各 step を指す → request→tool の対応付けが自明。
- diverse名版は request が値の羅列だけで anchor が無い → モデルは stage 進行を見失い、計画全体を再出力する。
- つまり真犯人候補は「名前の複雑さ」そのものより **request↔tool の対応付け (anchor) の喪失**。

検証中の A/B (extend probe, n=30, best_of=4, arm C, ローカル vLLM 0.25.0):

1. ~~description on/off~~ → **完了: 無罪** (desc無し 0/30 vs 有り 0/30)
2. ~~名前だけ mock風差替~~ (desc2 構造のまま、序数付き単純名) → **完了: 無罪** (0/30, stage_fail={1:23,2:1,0:6}) — 名前の形は原因でない
3. **request への anchor 注入** (desc2 のまま、user_request に step ごとのツール名を明示) → 走行中。回復すれば anchor 喪失が確定 = seed generator の `_intent_request` (「識別子を絶対に漏らさない」設計) を改める
4. natural vs transcription (値→arg 割当の曖昧さ) → anchor で決着すれば不要の見込み

補足: anchor 注入は訓練面でも筋が良い仮説 — request にツール名が出る訓練データは「名前を忠実に複写する」挙動を教えるので、BFCL −31pt の機構 (複写忠実度喪失) への対策と同じ向き。

## 4. 燃料再製造の手順 (質最優先・フォールバック禁止)

1. **T4 壊滅原因の切り分け** (§3 の A/B) ← いまここ
2. seed generator 修正 → 少数再生成 → extend probe で **T4 が mock 並 (~40%+) に戻るまで反復**
3. 本走行 (全 5,000 seed の rejection sampling, gpu-host :8199)。**T4 回復前に回さない** — desc2 のまま回すと T4 枠 3,250 seed (65%) が死ぬ
   - 発射前に不良 generation_records 消去 + **env 2点セット必須** (`SELFGEN_NOTHINK=1 SELFGEN_STAGE_HINT=1`)
4. G2 プリフライト (render errors 0 + 汚染監査 + sha + vault 保存) → 搬入
5. **fresh 再訓練**: stock base + router 凍結 + α=0.5 (汚染 checkpoint 系譜は使わない)
6. 境界測定 (3 軸 + BFCL paired) → HF 公開 (データセットカード執筆済み、識別子 sanitize + 敵対 sweep 後)

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
