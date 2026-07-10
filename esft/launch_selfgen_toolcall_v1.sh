#!/usr/bin/env bash
set -euo pipefail

# The 45-second delay in run_selfgen_toolcall_v1.sh lets this launcher bind the
# exact PID to jobEvents before either GPU is touched.
RUN_ID="${1:?usage: $0 RUN_ID}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT}/esft/data/selfgen_toolcall_v1/${RUN_ID}"
NOHUP_LOG="${RUN_DIR}/nohup.log"

test -f "${RUN_DIR}/manifest.json"
/usr/bin/python3 "${ROOT}/esft/selfgen_toolcall_v1.py" preflight --run-id "${RUN_ID}"
nohup "${ROOT}/esft/run_selfgen_toolcall_v1.sh" "${RUN_ID}" >"${NOHUP_LOG}" 2>&1 &
PID=$!
printf '%s\n' "${PID}" >"${RUN_DIR}/nohup.pid"
if ! /usr/bin/python3 "${ROOT}/esft/arm_selfgen_toolcall_v1_job.py" --run-id "${RUN_ID}" --pid "${PID}"; then
  kill "${PID}" 2>/dev/null || true
  wait "${PID}" 2>/dev/null || true
  exit 1
fi
printf 'PID=%s\nlog=%s\n' "${PID}" "${NOHUP_LOG}"
