# Full-FFN 決定論 probe arm B — 結果: GREEN (検分済み)

更新: 2026-07-10 23時台 JST (Fable 検分)。本ファイルの旧版は「Codex サンドボックスから SSH 不可で BLOCKED」だったが、gpu-host を直接検分した結果、**arm B は本日 05:31–05:38 UTC (14:31–14:38 JST) に実行済み・GREEN 完了**と判明した。前セッションが実行し DEVLOG 反映前に引き継ぎ境界を跨いだもの。旧版の BLOCKED 記録 (Codex sandbox の `socket: Operation not permitted`) は「新規再実行の試行」が塞がれたという事実として正しいが、実体の arm B は既に完了していた。

## 証跡 (gpu-host 直接検分)

- run dir: `/mnt/docker-raid/models/esft/codex_runs/fullffn_resume_det_b_20260710/` (arm A: `..._det_a_20260710/`、05:23–05:30 UTC)
- 決定論設定は per-rank ログで確認: `[fullffn-deterministic] enabled=True cublas=:4096:8 nccl_algo=Ring nccl_proto=Simple torch=True` (8 rank 全て)
- frozen param への gradient 漏れなし (router/attn/embed .grad all None、8 rank)
- peak max_memory_allocated 63.64GiB
- **最終判定 (outer.log 末尾)**:
  ```
  ASSERT load_model: MATCH (8)
  ASSERT load_optimizer: MATCH (8)
  ASSERT rng: MATCH (8)
  ASSERT batch_loss: MATCH (32)
  ASSERT gradient: MATCH (8)
  ASSERT post_optimizer: MATCH (8)
  RESUME REPRO RESULT: GREEN
  ```

## 解釈

- 先の「clip後 gradient digest 全 rank 不一致」は、決定論設定 (CUBLAS workspace 固定 + NCCL Ring/Simple + torch deterministic core + TF32 off) で**消える** — 発生源は grouped-mm backward / NCCL reduction の再起動間非決定性で確定。実装バグや DCP save/load の問題ではない。
- **Full-FFN の exact-resume 決定論ゲートは GREEN**。200-step 本番を技術面で止める栓は外れた。
- ただし 200-step 本番・本走の開始は**ユーザー承認待ちのまま** (GOALS_AND_TODO.md T2 の取り決め)。

## 残課題
1. 200-step 本番の GO/NO-GO (ユーザー)。
2. 決定論 env の速度コスト計測 (本番前に 1 度、同一条件 A/B)。決定論を本番でも維持するか、resume 検証専用にするかの判断材料。
