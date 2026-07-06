# 依頼: MoE k-expansion モデル開発のための包括的文献リサーチ

## 大目的(全ての判断はこれを基準に)

Qwen3.6-35B-A3B(MoE、総パラメータ35B / アクティブ3B、各層256 experts、top-k=8)をベースに、**推論時 top-k を32に引き上げた「35B-A6B」を開発し、知能(知識・推論)・一貫性・コーディング/agentic の3軸で元モデル(k=8)を統計的有意に上回るモデルを完成させる**。北極星ベンチマークは Terminal-Bench。ローカルGPUでの高速推論(投機的デコード込み)までを含めて「完成」とする。

このリサーチの目的は、**開発パイプラインの各段を文献で裏取り・改善・必要なら差し替える**こと。現行の手法選択は仮説にすぎない(後述)。より良い経路を示す文献があれば、遠慮なくそちらを推してほしい。

## 実測済みの事実(前提として使ってよい)

- naive に k=32 化(重み変更なし、routing 数だけ変更)すると知識系が劣化する:
  MMLU 84.3→80.7、GSM8K 89.3→86.5(各 n=600、同一条件、choice-logprob / no-think)
- コーディングの劣化は軽微: HumanEval 86.6→84.2(n=164、McNemar で有意差なし)
- 劣化の作業仮説: 追加された24 experts は「上位8で混合される」前提で事前訓練されており、32混合が未較正のためノイズ化する
- 頻度上位 ~6% の expert FFN のみを agentic データで微調整(router 凍結、delta 方式)すると、知識系は部分回復、HumanEval は naive k32 に有意勝ち(90.2 vs 84.2、McNemar p=0.04)
- 計算資源: 96GB GPU ×2 のノード2台 + 96GB GPU ×8 のサーバー1台(いずれも Blackwell / SM120)。フル事前訓練は不可能、FFN 全体の SFT は可能な規模

## 現在の作業仮説(決定事項ではない。覆すのは歓迎)

1. 混合ドメイン SFT(agentic + coding + toolcall + math を選抜 expert FFN の delta 訓練、router 凍結)で k32 のズレを修復
2. rejection sampling FT(best-of-8 で自己生成 → 正解のみ教材化)で上積み
3. GRPO 系 RL(実行ベース報酬)で最終仕上げ
4. 完成後に量子化(FP8 / NVFP4)+ 投機的デコードで配備

## リサーチトピック

### A. MoE の top-k 変更・expert 拡張

- 推論時 top-k を訓練時と変えた場合の挙動を直接扱った研究はあるか(性能・較正・ドメイン別感度)
- expert upcycling / expansion 系(dense→MoE、MoE の k や expert 数を増やす)の手法と「healing」レシピ
- k 変更後の再較正: router 再訓練 vs expert FFN 再訓練 vs 両方 vs gate 温度・renorm 調整、の比較エビデンス
- そもそも k=32 が最適という保証はない。k=16、動的 k(トークン毎可変)、レイヤー毎に異なる k、などの文献

### B. MoE の微調整手法

- ESFT(Expert-Specialized Fine-Tuning、DeepSeek 2024)とその後継・比較・批判研究
- router を凍結すべきか訓練すべきか、のエビデンス(router 訓練の不安定性・崩壊の報告含む)
- expert への delta/LoRA 訓練 vs full FT の品質差
- catastrophic forgetting 対策の MoE 特化知見(replay 混合、KL 蒸留、正則化)

### C. 知識劣化の回復

- MMLU 的な知識想起の劣化を SFT で回復した事例。必要なデータの種類と量の相場
- 汎用テキスト replay の混合率の相場(何%混ぜると忘却が止まるか)
- self-distillation の適用: 元モデル(k=8)の出力・logits を教師にして k=32 側を較正する、という発想の先行研究(on-policy distillation、GKD 等)

### D. rejection-FT / self-improvement

- STaR → RFT → rejection sampling fine-tuning → best-of-n distillation の系譜と2024年以降の到達点
- コード/agentic ドメインでの成功・失敗事例、データ量と反復回数の相場
- 自己生成データの品質フィルタ(実行検証以外に何が効くか)

### E. RL(GRPO 系)

- GRPO とその改良(DAPO、Dr. GRPO、RLOO、GSPO、VinePPO 等)の比較。小規模計算(GPU 8枚以内)で安定なのはどれか
- コーディング/terminal/agentic タスクの報酬設計(unit test 通過、実行ベース報酬、format 報酬の要否)
- SWE 系 RL(SWE-RL 等)の再現報告と落とし穴
- MoE を RL 訓練する際の固有問題(router 崩壊、expert 不均衡、rollout と train の精度不一致)

### F. 投機的デコード(このモデル専用 draft をゼロデイで用意したい)

背景: 最終モデル公開と同時に高速推論を提供したい。draft の訓練は本体訓練に比べ軽いので、本体完成 → draft 数日で訓練 → 同時リリース、を狙う。

- 35B級 MoE がターゲットの場合の最良手法比較: EAGLE-3、MTP ヘッド、独立小型 draft モデル、Medusa 系、self-speculative(layer skip)。accept rate と実測スピードアップの報告値
- ターゲットが「k=32 に微調整済みの非標準モデル」である場合:
  - draft は最終重みで訓練必須か、訓練途中の checkpoint で訓練した draft がどこまで流用できるか(target 分布シフトへの感度の実証研究)
  - ベースモデル用に訓練済みの draft/EAGLE ヘッドを微調整後モデルに転用した事例
- draft 訓練のデータ量・GPU時間の相場(最小構成でどこまで行けるか)
- vLLM / SGLang における Qwen3系 MoE + 投機的デコードの対応状況、既知 issue、Blackwell(SM120)固有の問題
- FP8 / NVFP4 量子化ターゲットと投機的デコードの併用時、accept rate への影響の報告

## 出力形式(厳守)

1. トピック A〜F ごとに: 論文リスト(タイトル + arXiv ID + 年)、各1〜2行の要点、**我々の設定への適用可能性(高/中/低 + 理由)**
2. 論文の自己申告の数字と、第三者による再現・追試の有無を区別して書く
3. 我々の作業仮説と**矛盾する知見は目立つ形で**報告する(隠さない・丸めない)
4. 最後に「自分がこのプロジェクトを設計し直すなら」という代替パイプライン案を1〜3個、根拠論文つきで提案
5. 読むべき優先度 top 10 論文を、理由つきでランキング

## 制約

- 2024年以降を中心に。ただし基礎文献(MoE 原典、Switch Transformer、speculative decoding 原典等)は含めてよい
- コード公開ありの研究を優先
- 公式 tech report・ブログも可、ただし査読論文と明確に区別して表記
- 断定を避け、各知見にエビデンス強度(単一論文のみ / 複数再現あり / ベンダー主張のみ)を添える
