#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python}"
: "${STUDENT_SCORER_CHECKPOINT:?Set this to a checkpoint produced by student scorer training}"
[[ -f "$STUDENT_SCORER_CHECKPOINT" ]] || { echo 'Student scorer checkpoint not found' >&2; exit 2; }
unset LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT PYTHONHOME PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO_ROOT" NAVSIM_DEVKIT_ROOT="$REPO_ROOT"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/cfs/qiuzijun/DriveWorld-VLA/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/exp}"
export FEATURE_CACHE="${FEATURE_CACHE:-${NAVSIM_EXP_ROOT}/worlddrive_navtest_cache}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/cfs/qiuzijun/DriveWorld-VLA/exp/metric_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${NAVSIM_EXP_ROOT}/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="${GPUS:-2}" --master_port="${MASTER_PORT:-29643}" \
  navsim/planning/script/run_pdm_score_worlddrive.py \
  --config-name default_run_pdm_score_student_scorer \
  navsim_log_path="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}" \
  sensor_blobs_path="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}" "$@"
