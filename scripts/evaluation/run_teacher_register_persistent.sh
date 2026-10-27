#!/usr/bin/env bash
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/cfs/qiuzijun/DriveWorld-VLA/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/exp}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/cfs/qiuzijun/DriveWorld-VLA/exp/metric_cache}"
export FEATURE_CACHE="${FEATURE_CACHE:-${REPO_ROOT}/exp/worlddrive_navtest_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export GPUS="${GPUS:-2}"
export MASTER_PORT="${MASTER_PORT:-29639}"
export TEACHER_EVAL_CACHE_MODE=true
export PYTHONPATH="$REPO_ROOT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV LD_PRELOAD
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export MPLCONFIGDIR="${REPO_ROOT}/exp/matplotlib"
export NCCL_DEBUG=WARN
EVAL_RUN_DIR="${EVAL_RUN_DIR:-${NAVSIM_EXP_ROOT}/teacher_register_navtest_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$EVAL_RUN_DIR" "$MPLCONFIGDIR"
exec >>"${EVAL_RUN_DIR}/eval.log" 2>&1
printf '%s\n' "$$" > "${EVAL_RUN_DIR}/supervisor.pid"
on_exit() {
  local code=$?
  printf '%s\n' "$code" > "${EVAL_RUN_DIR}/exit_code"
  printf '[supervisor] %s exit_code=%s\n' "$(date -u +%FT%TZ)" "$code"
}
trap on_exit EXIT
trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; exit 143' TERM
trap 'kill -INT "$child" 2>/dev/null; wait "$child"; exit 130' INT
printf '[supervisor] %s pid=%s GPUs=%s output=%s\n' "$(date -u +%FT%TZ)" "$$" "$CUDA_VISIBLE_DEVICES" "$EVAL_RUN_DIR"
bash scripts/evaluation/run_teacher_register_pdm_score.sh "output_dir=${EVAL_RUN_DIR}" "$@" &
child=$!
printf '%s\n' "$child" > "${EVAL_RUN_DIR}/torchrun.pid"
wait "$child"
code=$?
exit "$code"
