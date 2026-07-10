# Full-FFN 決定論 env 速度コスト A/B (ABBA) — 結果

実施: 2026-07-11 未明 JST、gpu-host GPU 0-7。実行者: Fable 直接 (Codex sandbox は SSH 不可のため BLOCKED → `/mnt/docker-raid/models/esft/run_fullffn_det_speed_ab.sh` を新規配置して Fable が実行)。旧版の BLOCKED 記録を実測結果で置換。

## 条件 (same-condition)

- arm A/B 決定論 probe と同一構成: 真stock、full-ffn、checkpoint-5 から resume、GA4、per-device 1、seq 7168、adafactor、`--skip-final-checkpoint`。max-steps 15 (実 10 optimizer steps)。
- A = 決定論 env 有効 (`--deterministic-fullffn` + CUBLAS_WORKSPACE_CONFIG=:4096:8 + NCCL_ALGO=Ring + NCCL_PROTO=Simple) / B = 無効。
- 実行順 ABBA (a1→b1→b2→a2)、各腕 fresh cold process。run dir: `codex_runs/fullffn_det_speed_ab_20260711/{a1,b1,b2,a2}`。checkpoint 書き込みなし、既存資産変更なし。

## 結果 (n=2/腕、same-condition、ABBA)

| arm | det | s/it (tqdm 最終) | wall (start→end) |
|---|---|---:|---:|
| a1 | on | 62.52 | 1115s |
| b1 | off | 56.56 | 1014s |
| b2 | off | 57.17 | 1007s |
| a2 | on | 61.39 | 1088s |

- det on 平均 61.96 s/it、det off 平均 56.87 s/it → **決定論 overhead = +5.09 s/it ≈ +8.9%** (n=2, same-condition, ABBA)。
- 腕内ばらつき: on 側 ±0.9%、off 側 ±0.5%。順序効果は ABBA で相殺。

## 判断材料 (200-step / 本走に向けて)

- 200-step probe を決定論 on で回すコスト: 約 +17分 (200×5.09s)。**resume 検証可能性を買う保険としては安い — 200-step は決定論 on を推奨**。
- 数日規模の本走では +8.9% は無視できない (例: 72h → +6.4h)。選択肢: (a) 本走も on で完全再現性、(b) 本走は off にして checkpoint 密度で保険、(c) resume 直後の検証区間だけ on。ユーザー判断。
- mechanism: overhead の主因候補は NCCL Ring/Simple 固定 (帯域最適 algo の放棄) と deterministic grouped-mm。分離計測はしていない (必要なら CUBLAS のみ / NCCL のみで再 A/B)。
