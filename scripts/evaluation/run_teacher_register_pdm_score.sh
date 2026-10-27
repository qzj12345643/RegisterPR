#!/usr/bin/env bash
set -euo pipefail

# Evaluate the trained future-register teacher through the official NAVSIM PDM
# scorer.  The teacher ranks the planner candidate pool; NAVSIM computes the
# final PDM score for the selected trajectory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/cfs/qiuzijun/miniconda3/envs/driveworld-vla/bin/python}"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the OpenScene/Navsim data root}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan maps root}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the experiment/data root}"

export NAVSIM_DEVKIT_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}"
export FEATURE_CACHE="${FEATURE_CACHE:-${NAVSIM_EXP_ROOT}/worlddrive_navtest_cache}"
export TEACHER_REGISTER_CHECKPOINT="${TEACHER_REGISTER_CHECKPOINT:-${REPO_ROOT}/exp/joint_future_registers_top16_pdm6_4gpu/2026.09.20.12.12.28/checkpoints/joint-registers-epoch=05-step=3552.ckpt}"
export PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${REPO_ROOT}/ckpts/worlddrive_stage1_train_87p7_pdms.ckpt}"
export WORLD_MODEL_CHECKPOINT="${WORLD_MODEL_CHECKPOINT:-${REPO_ROOT}/ckpts/worldtraj_stage1_512_tadwm.pkl}"
export TRAJECTORY_VOCAB="${TRAJECTORY_VOCAB:-${REPO_ROOT}/ckpts/trajectory_anchors_256.npy}"
export TEACHER_SCHEDULER_PATH="${TEACHER_SCHEDULER_PATH:-${REPO_ROOT}/ckpts/CogVideoX-2b}"
export PRETRAINED_VAE_PATH="${PRETRAINED_VAE_PATH:-${REPO_ROOT}/ckpts/CogVideoX-2b}"

GPUS="${GPUS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPUS}}"
MASTER_PORT="${MASTER_PORT:-29571}"
export MASTER_PORT

if [[ ! -f "${TEACHER_REGISTER_CHECKPOINT}" ]]; then
  echo "Teacher checkpoint not found: ${TEACHER_REGISTER_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -d "${METRIC_CACHE_PATH}" ]]; then
  echo "NAVSIM metric cache directory not found: ${METRIC_CACHE_PATH}" >&2
  exit 2
fi
if [[ ! -d "${FEATURE_CACHE}" ]]; then
  echo "Feature cache is missing: ${FEATURE_CACHE}. Set TEACHER_EVAL_CACHE_MODE=true to build it from raw sensors." >&2
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/navsim/planning/script/run_pdm_score_worlddrive.py" \
  --config-name default_run_pdm_score_teacher_register \
  agent.cache_mode="${TEACHER_EVAL_CACHE_MODE:-false}" \
  agent.cache_vae_token=true \
  train_test_split=navtest \
  navsim_log_path="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}" \
  sensor_blobs_path="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}" \
  "$@"
