#!/usr/bin/env bash
set -euo pipefail

# Platform images may inject libraries from a different Python environment.
unset LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT PYTHONHOME PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Edit this absolute path here to change environments. Ignore platform overrides.
PYTHON_BIN="/mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python executable does not exist: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
PYTHON_ENV="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd -P)"
export PATH="${PYTHON_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CONDA_PREFIX="${PYTHON_ENV}"
export CONDA_DEFAULT_ENV="${PYTHON_ENV##*/}"
export LD_LIBRARY_PATH="${PYTHON_ENV}/lib"
for library_dir in \
  "${PYTHON_ENV}"/lib/python*/site-packages/torch/lib \
  "${PYTHON_ENV}"/lib/python*/site-packages/nvidia/*/lib \
  /usr/local/nvidia/lib /usr/local/nvidia/lib64; do
  if [[ -d "${library_dir}" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${library_dir}"
  fi
done
# Only explicitly configured driver paths are appended, never inherited ones.
if [[ -n "${WORLDDRIVE_DRIVER_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${WORLDDRIVE_DRIVER_LIBRARY_PATH}"
fi

# Edit this path to select the config when running the script without arguments.
CONFIG_PATH="/mnt/cfs/qiuzijun/WorldDrive/navsim/planning/script/config/training/future_registers_4gpu.yaml"
CONFIG_NAME="${1:-${CONFIG_PATH}}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Use the current checkout's experiments/cache and the shared trainval data.
# Every value remains overridable; the shared maps link to DriveWorld-VLA.
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/cfs/datasets/navsim-v1}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_DEVKIT_ROOT="${REPO_ROOT}"
export WORLDDRIVE_CONFIG_ROOT="${WORLDDRIVE_CONFIG_ROOT:-${REPO_ROOT}/navsim/planning/script/config/common}"
export PYTHONPATH="${REPO_ROOT}"
# Keep multiprocessing scratch on the experiment volume; /tmp may be full.
export TMPDIR="${TMPDIR:-${NAVSIM_EXP_ROOT}/tmp}"
mkdir -p "${TMPDIR}"

CONFIG_ARGS=()
if [[ "${CONFIG_NAME}" == /* ]]; then
  if [[ ! -f "${CONFIG_NAME}" ]]; then
    printf 'Config file does not exist: %s\n' "${CONFIG_NAME}" >&2
    exit 2
  fi
  # Load the selected file first; retain repository defaults for external YAMLs.
  CONFIG_ARGS+=(--config-path "$(dirname -- "${CONFIG_NAME}")")
  CONFIG_ARGS+=(--config-dir "${REPO_ROOT}/navsim/planning/script/config/training")
  CONFIG_NAME="$(basename -- "${CONFIG_NAME}")"
fi

if [[ "${CONFIG_NAME}" == *.yaml ]]; then
  CONFIG_NAME="${CONFIG_NAME%.yaml}"
fi
CONFIG_ARGS+=(--config-name "${CONFIG_NAME}")

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
RUNTIME_LOG_DIR="${NAVSIM_EXP_ROOT}/runtime_logs"
mkdir -p "${RUNTIME_LOG_DIR}"
RUNTIME_LOG="${RUNTIME_LOG_DIR}/launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
printf '[WorldDrive] Python: %s\n' "${PYTHON_BIN}"
printf '[WorldDrive] Runtime check: %s\n' "${RUNTIME_LOG}"
if "${PYTHON_BIN}" -u -X faulthandler scripts/check_future_registers_runtime.py \
  --expected-env "${PYTHON_ENV}" 2>&1 | tee "${RUNTIME_LOG}"; then
  if [[ "${WORLDDRIVE_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    exit 0
  fi
else
  status=$?
  printf '[WorldDrive] Runtime check failed (exit %s); see %s\n' "${status}" "${RUNTIME_LOG}" >&2
  exit "${status}"
fi

exec "${PYTHON_BIN}" -u -X faulthandler navsim/planning/script/run_training_worlddrive.py \
  "${CONFIG_ARGS[@]}" \
  "${@:2}"
