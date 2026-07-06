# 一貫性メトリクス v1 — 層2 SWE-RL rollout

生成: consistency_metrics.py（CPU-only, numpy-only, GPU 不使用）
再実行:
```
~/vllm-env/bin/python consistency_metrics.py \
  --files rollouts/l2_base_k32.jsonl rollouts/l2_patch_k32.jsonl \
  --base base_k32 --json-out reports/consistency_v1.json
```

## データ来歴（scp は不要だった）

aux-host には `~/esft/rollouts/` が存在しない。層2 rollout は**既にローカル生成済み**で
`rl/rollouts/` にある（`rollout_gen.py` が aux-host serve を叩いて書いた、mtime 07-06 02:xx）:

| arm | file | prompts | n/prompt | temp | 由来 |
|---|---|---|---|---|---|
| base@k32  | `rollouts/l2_base_k32.jsonl`  | 48 | 4 | 1.0 | base Qwen3.6-35B-A3B, k=32 experts |
| patch@k32 | `rollouts/l2_patch_k32.jsonl` | 48 | 4 | 1.0 | ESFT patch 適用, k=32 |

2アーム揃い・同一 48 instance_id で **paired**。`rollouts_aux-host/` は空のまま（aux-host に転送元なし）。
`k32` は ESFT の expert 数であって sample 数ではない。sample は **n=4/prompt**。

overall lenient reward mean は log と完全一致（base **-0.9559**, patch **-0.8722**）= データ健全。

## 最重要の前提2つ（数字を読む前に）

**(A) 全アームが reward floor に張り付いている。** reward_lenient ∈ {-1.0} ∪ [0,1] で、
-1.0 = format-fail（パース/apply 失敗）、[0,1] = oracle diff との類似度。有効(> -1)は
base **5/192 (2.6%)** / patch **15/164 (9.1%)**、成功(>0.5)は base **3/192** / patch **10/164**。
分布は -1 のスパイク + 極薄の裾。**→ 「reward 分散」は解の品質のばらつきではなく
「4本のうち floor を脱出できたか」を測っている。** この局面では分散が大きい＝良い（脱出が起きた）
で、通常の「低分散＝一貫＝良い」と符号が逆になる。混同しないこと。

**(B) patch アームに infra 汚染。** patch の 192 完了のうち **28 (14.6%) が serve の
Connection reset（finish=`error:...`, tokens=0）**。これはモデル出力ではない。分布は
**7 prompts に集中し各4サンプル全滅**（serve が7 prompt 連続区間で落ちた）。残り 41 prompts は
完全クリーン(各n=4)。**モデル挙動の指標は全て「clean」(エラー除外)で算出**。paired 比較は
共通クリーン 41 prompts で実施。この 28 を含めると（"all"版）patch の floor 質量が水増しされ
有効率が過小に見える。

## 指標一覧（clean subset, per-prompt 集計 → 分布, n=prompts, n_sample/prompt=4, temp=1.0）

| 指標 | base@k32 | patch@k32 | 意味 |
|---|---|---|---|
| **format-valid rate** (clean, pooled) | **2.6%** [0.0,6.3] (5/192) | **9.1%** [3.5,15.6] (15/164) | reward_lenient>-1 の割合。今の律速 |
| **success rate** (clean, pooled) | 1.6% [0.0,4.7] (3/192) | 6.1% [1.7,11.5] (10/164) | reward_lenient>0.5 |
| length 切れ率 (all) | **25.0%** (48/192) | 19.3% (37/192) / clean 22.6% | finish=length。floor の主要因 |
| infra error 率 | 0.0% | **14.6%** (28/192) | serve 断。要修正 |
| reward_std (M1, ddof=1) | mean 0.050, **med 0** | mean 0.175, **med 0** | 中央値0＝大半の prompt が全4本 floor |
| pass@k gap best-mean (M2) | mean 0.055, med 0 | mean 0.180, med 0 | 同上、裾のみ非零 |
| token CV (M4) | mean 0.359 [0.29,0.43] | mean 0.291 [0.24,0.34] | 長さのばらつき |
| trunc_frac (M5, per-prompt) | mean 0.250 | mean 0.226 | prompt内の切れ割合 |
| **valid_repro** (M6b) | **0.417** (n=3 prompts) | **0.469** (n=8 prompts) | 有効を出せた prompt 内での有効率 |
| success_repro (M3) | 0.750 (**n=1 prompt**) | 0.417 (n=6 prompts) | 成功を出せた prompt 内での成功率 |

CI は 95% percentile bootstrap で **instance を resample（cluster bootstrap）**。
naive な per-sample bootstrap は楽観的になるため不採用。

## paired 差分（patch − base, 共通クリーン 41 prompts, cluster bootstrap）

`*` = 95% CI が 0 を跨がない。

| 指標 | base | patch | diff | 95% CI | 判定 |
|---|---|---|---|---|---|
| reward_std (M1) | 0.058 | 0.175 | **+0.117** | [+0.001, +0.242] | `*` patch が脱出多く分散↑ |
| pass@k gap (M2) | 0.065 | 0.180 | +0.115 | [-0.020, +0.252] | n.s.（正傾向） |
| token CV (M4) | 0.389 | 0.291 | **-0.097** | [-0.187, -0.011] | `*` patch の方が長さ一貫 |
| valid_frac (M6) | 0.030 | 0.091 | +0.061 | [-0.006, +0.134] | n.s.（3倍だが CI が0含む） |
| trunc_frac (M5) | 0.250 | 0.226 | -0.024 | [-0.122, +0.079] | n.s. |
| M3/M6b | — | — | — | — | 有効 prompt が少なすぎて paired 不能 |

## 解釈（mechanism 分解）

- **patch は base より確実に floor を脱出している**が、それは「reward 分散↑ (`*`)」という
  裏返しの形で出る。有効率 2.6%→9.1%（3.5倍）、成功率 1.6%→6.1%。ただし valid_frac の paired
  CI は僅かに0を含む（n=41, 各4本, 有効イベントが希少なため検出力不足）。方向は明確に正。
- **token CV は patch が有意に低い (-0.097 `*`)**＝ patch の出力長は base より揃う。これは
  「一貫性」らしい唯一の順方向シグナル。ただし length 切れ(10000 cap)を実出力として含むため、
  一部は切れ率差を反映。
- **floor の主要因は format と length 切れ**。base の 25%・patch の 19-23% が 10000 token で切れ、
  切れた完了はほぼ確実に format-fail（`</solution>` 未達）。**reward-品質のばらつきを語る前に、
  この2つ（parse 成功と切れ）が律速**。
- **成功の再現性 (M3) は base では測定不能**（成功 prompt が n=1）。patch は n=6 で「成功できる
  prompt でも4本中 41.7% しか成功しない」＝ **まぐれ寄り、まだ不安定**。valid_repro も base 0.42
  (n=3) / patch 0.47 (n=8)＝有効を出せる prompt でも半分以下しか有効patch にならない。

## 「一貫性」指標としての推奨（今後の追跡）

現局面（format floor）では **reward 分散系は追跡指標に不適**（解の信頼性でなく脱出率を測る、
中央値0で情報量なし）。優先順位:

1. **format-valid rate (clean, cluster CI)** — 現在の律速。下流(分散・成功)は全てこれに gate される。
   まずここを上げる。今の最良追跡量。
2. **valid_repro（有効 prompt 内の有効率）** — 最もクリーンな「再現性/一貫性」量だが、有効 prompt が
   増えないと不安定（今 base 3 / patch 8 prompts）。format-valid rate が上がれば意味を持ち始める。
3. **length 切れ率 + token CV** — floor の mechanistic driver。切れを減らす（学習で早く閉じる or
   max_tokens/停止条件）だけで format-valid が跳ねる可能性。
4. **reward_std / pass@k gap** — **format-valid rate が 20% を大きく超えるまで最適化対象にしない**。
   それ以降、reward-品質の within-prompt 分散が初めて「信頼性」を意味する。

## 次に生成すべきもの（この n では測り切れない）

- **測定力の壁**: n=4・有効率2-9% → prompt あたり期待有効数 0.1-0.36本。within-prompt 再現性(M3/M6b)を
  安定に測るには「4本中≥2本が有効」な prompt が要る。今それは base 3 / patch 8 prompt しかない。
  → タスク#3 の **INC-0 rollout 384×8**（384 prompt × n=8）が来れば prompt 8倍・sample 2倍で
  M3/M6b/valid_frac の CI が締まる。**完成次第この script を --files で回し直す**（引数対応済み）。
- **patch serve の infra 修正**: 28 の Connection reset = 7 prompt 全滅は serve 不安定の兆候。
  384×8 生成前に serve 側を安定化しないと、同じ汚染が大規模でも出る。生成後 `error_rate` を
  最初に確認すること（この script が per-arm で出す）。
- **base の n=1 成功 prompt** は成功再現性が原理的に測れない。base の脱出率を上げるか、
  format 強制プローブ（タスク#2）で fmt_ok を底上げしてから再測。

## 成果物

- script: `~/projects/qwen36-a6b/esft/rl/consistency_metrics.py`（`--files` 再実行可）
- json: `~/projects/qwen36-a6b/esft/rl/reports/consistency_v1.json`
- 本レポート: `~/projects/qwen36-a6b/esft/rl/reports/consistency_v1.md`
