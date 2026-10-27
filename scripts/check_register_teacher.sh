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

export PYTHONPATH="${REPO_ROOT}"
export MPLCONFIGDIR="${REPO_ROOT}/exp/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
export OMP_NUM_THREADS=2
export OPENSCENE_DATA_ROOT="/mnt/cfs/datasets/navsim-v1"
export NUPLAN_MAPS_ROOT="${OPENSCENE_DATA_ROOT}/maps"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -u scripts/check_register_teacher.py "$@"
