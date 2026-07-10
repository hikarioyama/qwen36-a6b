# 4軸評価ハーネス候補 (Grok 調査 + Opus 2体で一次ソース敵対検証済み)

作成: 2026-07-10 夜。検証方法: Grok(grok-4.3, web有効)の調査結果を、独立 Opus subagent 2体が GitHub/arXiv/HF の一次ソースで CONFIRMED/WRONG 判定。以下は **CONFIRMED のみ**、Grok の誤りは是正済み。

## 軸1: ツールコール
1. **BFCL v4** (推奨) — https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard、Apache-2.0、V4=2025-07-17。AST/executable の決定的採点、per-item 正誤で paired 可。**注意: v4 の `web_search` カテゴリのみ SerpAPI 依存 → 非live サブセット限定で使う**。日本語非対応。
2. **ACEBench** — https://github.com/chenchen0103/ACEBench、MIT、arXiv:2501.12851。~4,538 APIs/8ドメイン、Normal/Special/Agent、EN+ZH、mock 中心でローカル完結。
3. τ²-bench — MIT、arXiv:2506.07982。ユーザーシミュレータに LLM が必要=非決定的なので paired 主軸には不採用(参考のみ)。

## 軸2: 日本語
1. **JMMLU** (推奨) — https://github.com/nlp-waseda/JMMLU、n=7,536/56科目4択、exact-match 決定的。**ライセンスは混在: 53科目7,097問=CC BY-SA 4.0、3科目439問(VIST)=CC BY-NC-ND 4.0、日本史・世界史(STEP)=非商用制限(研究・評価は許諾)**。Grok の「56-57科目」は56が正。
2. **llm-jp-eval v2.1.5** (2026-06-03, Apache-2.0) — https://github.com/llm-jp/llm-jp-eval。**JHumanEval(コード実行採点)と M-IFEval(rule-based)を含み、vLLM 対応**。決定的部分に厳選して使う。
3. ELYZA-tasks-100 — 公式標準は人手評価(LLM-as-judge は慣行)。決定的でないため paired 主軸には不採用。

## 軸3: 一貫性
1. **M-IFEval 日本語 (推奨)** — 独立repo https://github.com/lightblue-tech/M-IFEval (arXiv:2502.04688, NAACL 2025 Findings)、llm-jp-eval にも統合済み。日本語固有の verifiable instructions(単なる翻訳でない)。rule-based 決定的。**複数 seed (k=5-10) の pass 分散/一致率を一貫性指標にする**(IFEval 本家 arXiv:2311.07911, n=541)。
2. **Paraphrase 一致率** — Self-Consistency (arXiv:2203.11171, Wang et al., ICLR 2023) を基盤に、JMMLU/GSM8K の paraphrase 版で回答一致率。paraphrase 生成パイプラインの固定が必須。
3. 長系列自己矛盾検出 — established harness がなく実装コスト高。後回し。

## 統合構成 (提案、protocol 凍結はユーザーと)
- ツールコール = BFCL v4 非live + ACEBench Normal/Agent
- コーディング = 既存 HumanEval (+JHumanEval で日本語コーディングを兼務)
- 日本語 = JMMLU + JHumanEval
- 一貫性 = M-IFEval(日) k=5 seed 分散 + paraphrase 一致率
- 全軸 item-level binary を統一保存 → McNemar + paired bootstrap CI95(既存ハーネスと同じ流儀)
- 外部API 完全排除、n はサブセットで制御

## 未決事項 (ユーザー判断)
1. 一貫性軸の定義をこの2指標(seed分散+paraphrase一致)で凍結してよいか。
2. ツールコール軸に日本語評価が存在しない — 内製 JP tool-call セットを作るか、EN で妥協するか。
3. 各軸の n と非劣化/改善マージンの凍結値。
