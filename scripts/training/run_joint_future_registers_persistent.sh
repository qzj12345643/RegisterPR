#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent launcher for the joint future-register training job.
# The caller should detach this script with: nohup setsid ... &

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_ROOT="${PERSISTENT_RUN_ROOT:-${REPO_ROOT}/exp/persistent_training/joint_future_registers}"
STAMP="${PERSISTENT_RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_PATH="${PERSISTENT_LOG_PATH:-${RUN_ROOT}/train-${STAMP}.log}"
PID_PATH="${PERSISTENT_PID_PATH:-${RUN_ROOT}/train-${STAMP}.pid}"

mkdir -p "${RUN_ROOT}"
printf '%s\n' "$$" > "${PID_PATH}"
exec >>"${LOG_PATH}" 2>&1

log() {
  printf '[persistent-launcher] %s %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

on_signal() {
  local signal="$1"
  log "received ${signal}; stopping training process group"
  case "${signal}" in
    INT) exit 130 ;;
    HUP) exit 129 ;;
    TERM) exit 143 ;;
  esac
}
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
trap 'on_signal TERM' TERM

log "started pid=$$ parent=$PPID gpu_ids=${GPU_IDS}"
log "repo=${REPO_ROOT}"
log "log_path=${LOG_PATH}"
log "pid_path=${PID_PATH}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || true
fi

set +e
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_PROJECT="${WANDB_PROJECT:-worlddrive}" \
  bash "${REPO_ROOT}/scripts/training/run_joint_future_registers.sh" "$@"
status=$?
set -e

log "training launcher exited with status=${status}"
if [[ "${status}" -gt 128 ]]; then
  log "process was terminated by signal=$((status - 128))"
fi
exit "${status}"
