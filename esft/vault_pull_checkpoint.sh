#!/bin/bash
# gpu-host の DCP checkpoint をローカル HDD (vault) へ回収する。ファイル単位 scp、
# サイズ一致ならスキップ (中断・再開可能)。完了後に全ファイル SHA256 照合し
# .vault_pulled マーカーを置く。実測帯域 28MB/s (2026-07-11, direct)。
# usage: vault_pull_checkpoint.sh <remote_run_rel> <checkpoint_name> <vault_dest_dir>
# 例: vault_pull_checkpoint.sh codex_runs/fullffn_joint_200step_20260711 checkpoint-100 \
#       /mnt/vault/checkpoints/fullffn_joint_probe200_20260711
set -euo pipefail
REMOTE_BASE=/mnt/docker-raid/models/esft
RUN_REL=$1; CKPT=$2; DEST_DIR=$3
SRC="$REMOTE_BASE/$RUN_REL/$CKPT"
DEST="$DEST_DIR/$CKPT"
SSH=(ssh -F "$HOME/.ssh/config" gpu-host)

mkdir -p "$DEST"
# リモートのファイル一覧 (相対パス + サイズ)
mapfile -t entries < <("${SSH[@]}" "cd '$SRC' && find . -type f -printf '%s %p\n'")
total=${#entries[@]}
echo "[vault-pull] $CKPT: $total files"
i=0
for entry in "${entries[@]}"; do
  size=${entry%% *}; rel=${entry#* }; rel=${rel#./}
  i=$((i+1))
  local_path="$DEST/$rel"
  if [ -f "$local_path" ] && [ "$(stat -c %s "$local_path")" = "$size" ]; then
    continue
  fi
  mkdir -p "$(dirname "$local_path")"
  echo "[vault-pull] ($i/$total) $rel ($size bytes)"
  scp -q -F "$HOME/.ssh/config" "gpu-host:$SRC/$rel" "$local_path"
done
echo "[vault-pull] transfer done; verifying sha256"
# 注意: リモートとローカルで locale の照合順序が違うと、同一内容でも sort 順が
# ズレて文字列比較が偽陰性になる (2026-07-12 実害)。比較直前に LC_ALL=C で
# 順序を正規化する。
remote_sha=$("${SSH[@]}" "cd '$SRC' && find . -type f -print0 | sort -z | xargs -0 sha256sum" | LC_ALL=C sort)
local_sha=$(cd "$DEST" && find . -type f ! -name .vault_pulled -print0 | sort -z | xargs -0 sha256sum | LC_ALL=C sort)
if [ "$remote_sha" = "$local_sha" ]; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "$DEST/.vault_pulled"
  echo "[vault-pull] VERIFIED_OK $CKPT"
else
  echo "[vault-pull] SHA_MISMATCH $CKPT"; exit 1
fi
