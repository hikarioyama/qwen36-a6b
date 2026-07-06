# qwen36-a6b — Qwen3.6-35B-A3B → A6B (k=32) 強化キャンペーン

MoE の推論時 top-k を 8→32 に拡張し、ESFT(選抜 expert delta 訓練)+ 混合 SFT + rejection-FT + GRPO で「A6B」として base を上回るモデルを作るプロジェクトの記録。

- **開発日誌**: [DEVLOG.md](DEVLOG.md) — 意思決定・実測(n/CI 付き)・失敗と教訓の時系列記録
- **計画**: [esft/PLAN.md](esft/PLAN.md)(master)
- **RL 設計**: [esft/rl/RL_DESIGN.md](esft/rl/RL_DESIGN.md)
- **実装**: `esft/`(delta 方式 ESFT trainer、eval harness、汚染ゲート、SWE-RL 報酬)
- **測定結果**: `esft/reports/eval/`(per-item JSON 込み、paired McNemar で判定)

訓練データ・重み・rollout は容量の都合で repo 外(.gitignore 参照)。データの出自と処理は DEVLOG と PLAN に記録。
