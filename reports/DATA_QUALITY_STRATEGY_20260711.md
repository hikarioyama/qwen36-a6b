# 最高品質データセット製造戦略 (2026-07-11 確定版)

作成: 敵対的パネル 5 腕 (Opus) + Grok 文献調査 + 当日実測を統合。
ドラフト: scratchpad/distill_strategy_draft.md、パネル生ログ: workflows wf_fe5c6bb0-c48 journal。

## 0. 結論 (1 段落)

**二層に分ける。今区間 (deadline ~20h) は「確実に出荷できる A 群 de-scope 版 (v4)」を発射し、品質の本戦 (意図レベル selfgen + 蒸留 + 一貫性軸) は次々区間 (v5) に全力投入する。** ops 批判の裁定「フル漏斗は 24h に収まらないが、de-scope 版は余裕で出荷可」に従う。GPT/Grok 余剰の主戦場は「審判 (厳選・監査)」と「難タスク設計」で、テキスト直接混入 (C 群) は機構が要求する最小限に絞る。

## 1. 実測アンカー (今日の数字)

| 数字 | 条件 | 含意 |
|---|---|---|
| 単発合格率 93.3% | wave1 candidate_index 分布, stage 単位 n=7,411 | ZPD (中間難易度帯) が seed プールにほぼ無い。転写タスク設計が原因 (request に正解コールを全記載) |
| best-of-4 採用率 98.6% | wave1 n=5,000 | 同上。全滅 seed 72 件のみ → フロンティア蒸留の弾も枯渇 |
| replay (mixed_v2) 分布 | 385k 行, 1/10 サンプル | math 44% / coding 24% / toolcall 18% / agentic 10%、**general+knowledge ~2%** — 借金返済用の general 矯正信号が replay にほぼ無い |
| v3 分布 | 240k 行 (801MB), 1/5 サンプル | japanese 39% / code 37% / knowledge_en 24% (mmlu_aux, nemotron_stem, arc)。実効 general 比率 ≈ 17% (v3×0.7+replay×0.3) |
| base@k32 の借金 | −3.17pt vs base@k8, MMLU n=600 paired, p=0.013 | 機構仮説: rank 9-32 expert の未較正 + softmax 再正規化。返済には general 矯正勾配と router 健全性の観測が必要 |

## 2. 今区間 (v4, 出荷ライン) — de-scope して確実に発射

**構成 (全て存在する部品のみ):**
1. **v3 継承** + Δ: selfgen wave1+2 (~10k 行) + Toucan clean_v2 厳選サブセット
2. **selfgen の難易度層別 (追加計算ゼロ)**: candidate_index≥1 の seed (6.7%) は全量、idx=0 は減衰サンプリング — 無料の難易度ラベル (IRT 的層別、パネル hack #3)
3. **knowledge/general スライス増強**: replay の general ~2% は薄すぎる。v4 では knowledge_en 系 (nemotron/mmlu_aux 系統) の比率を上げ、rank 9-32 への矯正勾配を確保 (借金対策の主レバー)
4. **Toucan 厳選**: ヒューリスティックフィルタ (長さ/構造/ツール数) + Codex rubric は spot-check (全量厳選は次々区間 — レート的に間に合わないため)

**スキーマ凍結 (S5 リスク対応)**: trainer offsets 改修が「マージ + smoke test 済み + 12h マージン」を満たさなければ**現行形式で出荷**。offsets を使う場合は loss-mask byte-exact 監査ゲート (パネル hack #6) を必須で挟む。

**プリフライト (発射 2h 前ステージング)**: ①loader で全量ロード検証、②decontam 再監査 (random 5k, 0 hit 確認)、③ローカル 5-step dry train-load。

**今区間で切るもの (次々区間へ)**: loss フィルタ / 分類器外挿 / 修正蒸留 / フロンティア蒸留の全量運用 / 一貫性軸 / ja v2.1 待ちの日本語増産。

## 3. eval ゲートの是正 (盛り防止)

- **実の床は base@k8**。base@k32 に勝つだけでは「元より弱いモデル」で勝利宣言できてしまう。返済完了条件 = joint@k32 ≥ base@k8 (MMLU n=600 paired)。vs base@k32 と vs base@k8 の両方を毎回併記する。
- **router-health 診断を常設**: rank 9-32 の gate 質量 / routing entropy / 256 expert 使用率 / 実効 active expert 数。訓練前後で比較し、「collapse (k32 計算の丸損)」か「narrow 過剰特化 (MMLU 悪化)」かを機構レベルで観測する。200-step export (転送中) で baseline を取る。
- eval 独立性: データ選別 judge (GPT/Grok) と eval を分離。eval は機械 gold (choice-logprob / mock executor / pytest) を最優先、意味的判定が要る場合のみ別 family judge。

## 4. 次々区間 (v5, 品質の本戦) — GPT/Grok 総動員

### 4.1 意図レベル selfgen (スペック済み: spec_selfgen_toolcall_intent.md)
転写 request をやめ、期待 trace を隠し持ったまま request を自然文化 (GLM-5.2 paraphrase = A 群クリーン維持)。distractor ツール + tier 制 (T2 意図 / T3 distractor / T4 長 chain)。**一意可解性監査を GPT/Grok に委任** (テキスト非混入)。tier 別採用率で「35B の能力の崖」を実測し、ZPD 帯 (pass 20-80%) に生成を寄せる閉ループ。

### 4.2 counterfactual tool-result 注入 (パネル hack #2)
成功 trace の mock executor 返り値を機械汚染 (404 / rate-limit / 空 list / 型不一致) → 回復続行を生成 → 状態機械で回復を機械検証。**検証可能性を保ったまま「失敗の尾」を製造**。注入テーブルは実 API の error taxonomy に限定。「諦めが正解」ケースも用意。

### 4.3 蒸留の正しい形 (訓練科学批判で修正)
- **on-policy 本命は rejection-sampled self-rollout** (難 seed で 35B が稀に成功した trace)。「修正蒸留 = on-policy」は誤りだった (exposure bias / prefix は勾配ゼロ) — 撤回。
- 修正蒸留は「修正後 trace で fresh rollout の pass 率が上がるか」(転移) で採否較正。diff% ゲートは廃止。
- フロンティア蒸留 (C 群): クリーン teacher (GLM-5.2/DSv4) が同 seed で合格すれば A 群優先。C 群キャップは**行数でなく実効 (non-zero-loss) トークン寄与**で管理。
- expert-gate 標的選別 (パネル hack #1): 候補を 1 forward し「rank 9-32 が受ける gate 質量」で優先度付け → 借金の機構を直撃。**最安パイロット: 上位 20% vs random 20% (各 300 件) を訓練し MMLU paired、一晩で殺すか活かすか**。

### 4.4 一貫性軸 (未設計 → 機械検証化)
- **Commitment-Ledger**: turn1 の約束 (FactSlot/FormatConstraint/Prohibition/PersonaAttr/SequenceState) を typed slot で植え、probe turn で復唱を exact 照合 — 一貫性の 60-80% を programmatic hard-label 化 (hypothesized)
- **Tripwire 制約 long rollout**: 「以後数字は漢数字」等の全走査可能な制約 + 系列長カリキュラム (L=6→20)。teacher 不要 = A 群 on-policy
- 矛盾注入→検出・修正ペア (注入は機械的なので ground truth 既知)。副産物として DPO ペア (RL 段用)
- 圧力 turn の spec のみ teacher 設計 (役割 4)

### 4.5 日本語軸
- **cross-lingual round-trip アンカリング**: ja 応答の tool-call/code コアを en 側の機械検証済み参照と exact 照合 — 主観二審への依存を構造コアで置換
- LLM judge は fluency しか測れず GPT 文体 (敬語過多/translationese) を注入するリスク → plain_style を明示フィルタ項に、judge は補助に格下げ
- ja v2.1 (語彙拡張、Codex 実装中) 完成後に量産再開

### 4.6 汚染・独立性の INC-0 (ゼロコスト先行ゲート)
1. **mutated-eval 再生率**: eval 問題の答えだけ変えた変異版を teacher に通し、n-gram フィルタの実 recall を測る (semantic 汚染 = paraphrase 再構成は n-gram で recall≈0 の疑い)
2. frozen clean eval slice (どの teacher/judge も触れていない機械検証 n 数百) を凍結
3. routing-entropy baseline (200-step export で測定)
4. judge P/R を機械検証済み gold で較正 + Codex/Grok の判定相関を測る (相関する二審はバイアスを打ち消さない)

## 5. GPT/Grok 余剰の使用マップ (優先順)

| # | 用途 | 系譜影響 | 文献裏付け |
|---|---|---|---|
| 1 | rubric 厳選 + cross-judge (PoLL 型 panel) | 選抜バイアス経由の弱い影響 (GPT-selected フラグ付与) | PoLL (2404.18796), バイアス定量 (2410.02736) ✓実在 |
| 2 | 一意可解性監査 (意図レベル seed) | 同上 | — |
| 3 | 難タスク spec / taxonomy 設計 (テキストはローカル具現化) | spec 系譜を記録 | LAB (2403.01081), PersonaHub (2406.20094) ✓実在 |
| 4 | フロンティア蒸留 (難 seed 解答, C 群) | **直接混入 — 実効トークン寄与でキャップ** | Phi-4 (2412.08905) ✓実在 |
| 5 | 修正蒸留 (転移較正後, C 群) | 直接混入 (span 単位ラベル) | 未検証 (2026 論文群) |
| 6 | ja/一貫性の意味的残渣の二審 | 選抜影響 | 同上 |

系譜ラベルは「除去可能 (生成テキスト)」と「不可逆 (選別/spec 影響)」の 2 種に分けて記録する (完全性批判の採用)。

## 6. パネル批判の採否

**採用**: 借金返済機構の欠落 (S5) / eval 床の是正 (S5) / スキーマ凍結 + プリフライト (S5×2) / semantic 汚染の INC-0 (S5) / judge-eval 独立性 (S5) / on-policy 誤標識の訂正 (S4) / diff% ゲート廃止 (S4) / ZPD 閉ループ (S4) / C キャップの実効トークン化 (S4) / replay 増強 (S4) / 多様性 floor (S4) / de-scope 出荷 (ops 裁定) / hack #1,2,3,6 / 一貫性軸 5 提案。
**保留**: 「一貫性を今区間から完全に切る」は採用しつつ、Commitment-Ledger の設計は v5 準備として先行 (設計コストは炉と競合しない)。loss フィルタは Superfiltering (2402.00530 ✓) が弱 proxy で可能と示すため、v5 では**軽量 proxy** で復活検討。

## 7. 実装 backlog (Codex 発注順、スロット空き次第)

1. **意図レベル selfgen** (spec_selfgen_toolcall_intent.md) — v5 の主力
2. counterfactual 注入ラッパー + 回復検証 (小規模、既存 executor 拡張)
3. router-health 診断スクリプト (gate 質量 / entropy / expert 使用率) — 200-step export で baseline
4. Commitment-Ledger + tripwire 検証器 (一貫性軸 0→1)
5. mutated-eval 再生率ハーネス (INC-0)

走行中の 3 本 (offsets 改修 / ja v2.1 / 品質選別ハーネス) は v4/v5 両方の部品。
