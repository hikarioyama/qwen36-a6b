#!/bin/bash
# 外部コーパス intake v1 全量 run: scan (全量実測) → decontam (8-gram 除去)。CPU/IO only。
# 起動は detached (nohup)。完了マーカー INTAKE_V1_DONE。
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (this script lives in esft/)
RUN_ROOT=${RUN_ROOT:-/mnt/vault/corpora/derived/qwen36-a6b-intake-20260711-v1}
mkdir -p "$RUN_ROOT"

echo "=== PHASE scan start $(date -u +%s)"
python3 esft/corpus_decontam_v1.py scan \
  --input /mnt/vault/corpora/toucan-1.5m \
  --input /mnt/vault/corpora/toolmind \
  --batch-size 512 --limit-per-file 0 \
  --output-json "$RUN_ROOT/scan.json" \
  --output-md "$RUN_ROOT/scan.md"

echo "=== PHASE decontam start $(date -u +%s)"
python3 esft/corpus_decontam_v1.py decontam \
  --input /mnt/vault/corpora/toucan-1.5m \
  --input /mnt/vault/corpora/toolmind \
  --batch-size 512 \
  --output-dir "$RUN_ROOT/clean"

echo "INTAKE_V1_DONE"
