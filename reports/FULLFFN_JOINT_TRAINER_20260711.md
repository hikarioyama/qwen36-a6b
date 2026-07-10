# Full-FFN + router 可動 joint trainer — 2026-07-11

## 判定

`esft/deploy/train_fullffn_dcp.py`（gpu-host 本番 trainer の SHA
`8c7433fd…` 取得コピー）に、**二重 opt-in** の Full-FFN + router 可動
joint モードを追加した。ここでの検証は CPU 構造テストのみであり、GPU 訓練・gpu-host
配備・200-step probe は実行していない（配備と実行は Fable 担当）。

背景の判断は `reports/GOALS_AND_TODO.md` T1 と `DEVLOG.md` の
2026-07-08 (4): router 凍結下の B2 は CE が改善しても一般能力を守れず、次の検証は
full-FFN と router を同時に動かす経路である。

## 起動契約

`--method full-ffn --train-router` だけでは拒否する。joint は次の二つを明示した時だけ
有効になる。

```bash
--train-router --allow-router-joint-fullffn
```

`--allow-router-joint-fullffn` 単独、または full-FFN 以外との併用も拒否する。従来の
delta、maskhook、router 凍結 Full-FFN は変更しない。`--deterministic-fullffn` はこの
新 flag と直交しており、既存の決定論環境検査をそのまま通る。

## 実装

- `enable_router_training` と base router snapshot を利用する。anchor は router forward hook が
  返す top-k **前**の full 256-expert logits を使い、FSDP wrap 後に live gate weight を
  再参照しない（詳細は末尾の差し戻し修正）。
- FSDP wrap 後に Trainer が optimizer を生成する時点で、gate 名を再解決する。
  expert group は base LR / 指定 weight decay、router group は
  `router_lr_mult * learning_rate` / weight decay 0。pre-wrap parameter を optimizer に
  渡さない。
- DCP checkpoint の model と optimizer に gate parameter/state が通常どおり含まれる。
  marker は schema 3 にし、`router_joint`（enabled、router LR multiplier / effective LR、
  anchor weight / stride）を保存する。joint resume は同じ設定を要求し、旧 marker に
  `router_joint` が無い checkpoint から joint resume することは拒否する。旧 frozen
  checkpoint の frozen resume は互換のまま。
- `FULLFFN_PROBE=1` は joint 時に router parameter が少なくとも一つの FSDP rank で
  non-zero gradient を受けることを必須にした（rank 間 MAX union）。router 凍結時は従来どおり
  router / attention / embedding を、joint 時は attention / embedding を含む全 frozen
  parameter の `grad is None` を要求する。

## load-balancing auxiliary loss の結論

joint でも load-balancing auxiliary loss は**混入しない**。`--fused-ce` 経路は backbone
hidden state から FLCE を直接構成し、model が返し得る router auxiliary loss を加算しない。
joint で gate へ加わる補助目的は、明示指定した `--router-anchor-weight` の base-routing
anchor KL だけである。この性質を CLI help、実行ログ、コード、ここに明記した。

## CPU 検証

実行日: 2026-07-11 JST。GPU job / benchmark sample は **n=0**（paired verdict・truncation は
該当なし）。

- `python3 esft/tests/test_fullffn_joint_trainer.py` — **3/3 PASS**
  - router の独立 param group（LR `8e-7 = 0.08 * 1e-5`、no-decay、expert と非重複）
  - existing `RouterAnchor` の KL 値が手計算値と一致し、anchor gradient の一歩で KL が減る
  - frozen / joint の frozen-grad assertion と joint router non-zero grad predicate
- `python3 -m unittest discover -s esft/tests -p 'test_*.py'` — **33/33 PASS**
  （既存 eval-harness を含む）
- `CUDA_VISIBLE_DEVICES='' python3 esft/tests/test_smoke.py` — **22/22 PASS**
- `python3 -m py_compile esft/deploy/train_fullffn_dcp.py`、`git diff --check` — PASS。

## 残リスクと Fable 配備前の確認

CPU は FSDP/DCP の collective と実 35B gate parameter 名を再現しない。200-step probe 前に
Fable は本番環境で、(1) joint freeze audit の `expert_tensors=80` と router tensor 数、
(2) joint optimizer の router LR、(3) `FULLFFN_PROBE` の router union coverage、
(4) checkpoint save → resume の model/optimizer digest MATCH、を確認する必要がある。
特に router joint checkpoint の resume は、同じ router anchor 設定を使うことが必須である。
決定論環境は既存 arm A/B で GREEN、速度コストは 200-step 条件で +8.9%（n=2/arm）だが、
joint 経路の memory / speed / exact-resume は未測定である。

## 2026-07-11 差し戻し修正: FULL_SHARD anchor の FSDP 安全化

gpu-host の 8-rank FULL_SHARD 実機では、joint anchor が全 rank で停止した。正とする
観測エラーは `training_step` の anchor 計算中の
`F.linear(xf, gate.weight)` であり、`input (7168)`, `mat (7168x2048)`,
`vec (0)` の size mismatch である。これは `gate.weight` が gate 自身の FSDP unit の
forward 実行中だけ unshard され、forward 完了後に trainer 側から読むと local shard が
0 要素になるためである。delta で露呈しなかったのは router が wrap 外だったためで、
FULL_SHARD の正当化にはならない。

`train_fullffn_dcp.py` は以下へ置換した。

- 起動時（FSDP wrap 前）の base gate weight は CPU snapshot にのみ保持する。
- `Qwen3_5MoeTopKRouter` の forward hook が、unshard 中に返す tuple の第 1 要素、すなわち
  top-k 前の full 256-expert `router_logits` を直接捕獲する。anchor は後段で live
  `gate.weight` を再参照しない。
- 同じ hook 内で入力 hidden state と CPU snapshot から base logits を作る。従って base 側も
  anchor の後段で live parameter を参照せず、二重 teacher forward は増えない。
- `begin()` / `compute()` で outer forward の捕獲範囲を閉じ、gradient-checkpoint の backward
  再計算で再発火する hook は記録しない。これにより stale recompute activation が次の
  microbatch に混入しない。trainer は既存どおり non-reentrant checkpoint を明示する。

CPU 再検証（GPU job / benchmark sample は **n=0**、paired verdict・truncation は該当なし）:

- `python3 esft/tests/test_fullffn_joint_trainer.py` — **4/4 PASS**。既存の optimizer / freeze
  assertions に加え、pre-top-k logit の KL 手計算一致、CPU snapshot、空の公開 `gate.weight`
  を持つ FULL_SHARD 模擬 router、checkpoint 再計算 hook の無視を検証した。
- `python3 -m unittest discover -s esft/tests -p 'test_*.py'` — **34/34 PASS**。
- `CUDA_VISIBLE_DEVICES='' python3 esft/tests/test_smoke.py` — **22/22 PASS**。
- `python3 -m py_compile esft/deploy/train_fullffn_dcp.py`、`git diff --check` — PASS。

Fable の配備・実機再走では、最初の microbatch で router anchor が 40 layer 分を捕獲し、
`vec (0)` が再発しないこと、ならびに既存の joint gradient / checkpoint-resume probe を確認する。
