# スペック: 意図レベル・ツールコール selfgen (難易度エスカレーション)

## 動機 (実測)
現行 selfgen_toolcall_v1 の user request は期待コールを文面で全指定する転写タスク
(`make_seed` L347-380: "call {tool} with {field}={value}" を直接埋め込む)。
採用率 98.6% (n=5000) はタスクが易しすぎる証明。ツール選択・計画・引数導出を訓練できていない。
全滅 seed 72/5000 ではフロンティア蒸留の弾も枯渇する。

## 設計原則
1. **検証可能性は生成器側で担保**: 期待 trace を先に構築 (現行と同じ) → request の表現だけを難しくする。
   mock executor / 検証器はそのまま使える。
2. **A 群クリーン性の維持**: request の自然文化はローカル GLM-5.2 で行う (paraphrase)。
   GPT/Grok は「一意可解性の監査」(審判) のみ — テキストは訓練データに入らない。
3. 難易度は tier で層別し、tier ごとに採用率を計測して「35B の能力の崖」を見つける。

## 難易度 tier
| tier | 内容 | 期待採用率 |
|---|---|---|
| T1 (現行) | 転写 + フォーマット遵守 | ~98% (実測) |
| T2 | 意図レベル request: コール構造を書かない。引数値は request 本文か前 stage 結果から導出可能に | ? |
| T3 | + distractor ツール 3-5 本 (類似名・類似スキーマ、正解ツールと紛らわしい) | ? |
| T4 | + 長 chain (3-4 stage)、parallel と sequential の混在、条件分岐つき error recovery | ? |

## パイプライン
1. 生成器: 期待 trace + 転写 request を構築 (現行ロジック流用、stage 数・distractor を拡張)
2. **自然文化**: GLM-5.2 (ローカル serve) が転写 request を「意図レベルの自然な依頼」に書き換え。
   制約: 全引数値 (または導出元) が本文に残ること — programmatic チェック (値の出現検査)
3. **一意可解性監査** (サンプル監査→全量は閾値運用): GPT (cx) / Grok に tools+request を渡し
   「期待 trace が一意の正解か? 別解や曖昧性はないか」を判定させる。不合格テンプレは修正
4. 35B best-of-4 生成 → mock executor 検証 → トリアージ:
   - 合格 → A 群 selfgen (on-policy、意図レベル)
   - 全滅 → フロンティア蒸留 queue (GPT/Grok に解かせ検証、C 群) + 修正蒸留 queue (惜しい失敗)
5. 全量: decontam → Codex rubric 厳選

## 実装ノート
- selfgen_toolcall_v1.py の make_seed / scaffold_prompt / select_candidate を拡張。
  seed 側に `tier`, `distractor_tools`, `natural_request` フィールド追加
- 検証は expected_stages との照合 (現行 select_candidate) をそのまま使う —
  distractor 呼び出しは自動で不合格になる
- GLM-5.2 paraphrase は障害点なので、値出現検査に落ちたら転写 request に fallback (tier を T1 扱いに降格)
- tier 配分 (初期案): T1 10% / T2 40% / T3 30% / T4 20% — 採用率実測後に「崖」へ寄せる
