#!/usr/bin/env bash
set -euo pipefail
# Weight-only warm start: TEACHER_REGISTER_CHECKPOINT and optional STUDENT_INIT_CHECKPOINT.
# Resume THIS stage (not the preceding joint run): DISTILL_RESUME_CHECKPOINT.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python"
unset LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT PYTHONHOME PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/cfs/datasets/navsim-v1}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_DEVKIT_ROOT="${REPO_ROOT}"
export WORLDDRIVE_CONFIG_ROOT="${WORLDDRIVE_CONFIG_ROOT:-${REPO_ROOT}/navsim/planning/script/config/common}"
export FEATURE_CACHE="${FEATURE_CACHE:-${NAVSIM_EXP_ROOT}/worlddrive_train_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${NAVSIM_EXP_ROOT}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

# W&B credentials are kept outside the repository.  An explicitly exported
# key takes precedence; otherwise read only the value from the shared file.
WANDB_CREDENTIALS_FILE="${WANDB_CREDENTIALS_FILE:-/mnt/cfs/qiuzijun/wandb.md}"
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_CREDENTIALS_FILE}" ]]; then
  export WANDB_API_KEY="$(awk -F': *' '$1 == "WANDB_API_KEY" {print $2; exit}' "${WANDB_CREDENTIALS_FILE}")"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-worlddrive}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${NAVSIM_EXP_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  printf 'WANDB_API_KEY is missing; expected an exported key or %s\n' "${WANDB_CREDENTIALS_FILE}" >&2
  exit 2
fi
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -u navsim/planning/script/run_training_worlddrive.py \
  --config-name register_distill_4gpu \
  "hydra.searchpath=[file://${WORLDDRIVE_CONFIG_ROOT}]" "$@"
