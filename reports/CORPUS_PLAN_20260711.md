# Full-FFN 本走用コーパス計画 (検証済み candidates + 自己生成)

作成: 2026-07-11。方針 (ユーザー承認済み): **足場付き自己生成 + 機械選別を柱に、外部の検証済みデータを混合。生成=ローカル、訓練=gpu-host。** 外部候補は Grok 調査 → Opus 2体で一次ソース裏取り済み。以下は検証済み情報のみ (Grok の誤り2点は是正済み)。

## 外部データ (検証済み、推奨順)

### ツールコール
| データ | 規模 | ライセンス | 備考 |
|---|---|---|---|
| **Toucan-1.5M** (Agent-Ark, HF) | 1,646,546 rows | Apache-2.0 | 実 MCP 環境由来、495 servers/2000+ tools。最有力。filtering 推奨 |
| **ToolMind** (Nanbeige, HF) | 368,611 rows | Apache-2.0 | 160k 合成+200k augmented、multi-turn reasoning |
| ToolACE (Team-ACE, HF) | ~11.3k | Apache-2.0 | 26,507 API、質は高いが小規模 |
| APIGen-MT-5k | 5k | **CC-BY-NC-4.0 + OpenAI条項** | 品質参考用のみ。訓練混合は避ける (ライセンス二重縛り) |

### コーディング
| データ | 規模 | ライセンス | 備考 |
|---|---|---|---|
| Magicoder-OSS-Instruct-75K | 75,197 | MIT 表記 | ただし GPT-3.5 生成で OpenAI ToS が実質乗る。要判断 |
| swallow-code-v2 (tokyotech-llm) | 大 | Apache-2.0 | コード CPT 用 (JP 検証の副産物で発見) |
| **LiveCodeBench は訓練禁止** | — | — | eval 専用に隔離。逆に訓練コーパスから LCB 出典問題を除外フィルタする |

### 日本語
| データ | 規模 | ライセンス | 備考 |
|---|---|---|---|
| **llm-jp-corpus-v4** (GitLab NII) | ja ~0.69T tokens | サブセット毎に異なる | 高品質サブセット抽出して CPT 混合。単一ライセンスでない点に注意 |
| llm-jp-instructions | 1,000 (人手) | CC-BY-4.0 | 小さいが商用可・最高品質 |
| AnswerCarefully v2.2 | ~千規模 | 利用可 | 安全性 QA |
| ichikara-instruction | ~4.8k | **CC-BY-NC-SA 4.0 (商用不可)** | 使うなら非商用縛りを受容するか要判断 |
| ~~Swallow Corpus 本体~~ | — | — | **Grok の誤り: 本体は再配布されていない** (公開は swallow-code 等の派生のみ) |

### 一貫性/長対話
| データ | 規模 | ライセンス | 備考 |
|---|---|---|---|
| PersonaAtlas (yccm, HF) | 10,462 multi-turn 会話 | MIT | 2026-02 公開、persona 一貫性合成 (arXiv:2602.12394)。実在確認済み |
| (参考) arXiv:2511.00222 | — | — | persona 一貫性の multi-turn RL。日本語文脈ではない (Grok の文脈ミス是正) |

## 自己生成トラック (v1 実装中)
- ツールコール軸: 足場付き生成 (system+few-shot+best-of-4) → mock 実行検証 → 合格のみ採用。`esft/data/selfgen_toolcall_v1/`。500 サンプルで検分 → 量産。
- 続いて: 日本語 verifiable 指示 (M-IFEval 同型の機械検証指示を自動生成)、一貫性 (矛盾検出フィルタ付き長対話)。**verifiable 日本語指示追従の外部データはほぼ存在しない** (検証済み) ため、この軸は自己生成が主力になる。

## 全軸共通の規律
1. eval 汚染除去: MMLU/GSM8K/HumanEval/JMMLU/BFCL/M-IFEval/(将来使うなら LiveCodeBench) と n-gram 照合、除去ログを manifest に残す。
2. 機械検証済みしか入れない (実行成功/テスト通過/ルール合格)。
3. 混合比は 200-step 級 probe で振ってから本走 (70/30 決め打ちにしない)。
4. ライセンス: 商用可 (Apache/MIT/CC-BY) を A 群、非商用 (ichikara/APIGen-MT) を B 群に分離管理し、B 群は既定で混ぜない。

## 未決 (ユーザー判断)
- B 群 (非商用) データを使うか。
- Magicoder の「MIT 表記 + OpenAI ToS」をどう扱うか。
- llm-jp-corpus-v4 のサブセット選定と取得規模。
