#!/usr/bin/env bash
set -euo pipefail
: "${PLANNER_CHECKPOINT:?Set PLANNER_CHECKPOINT to the trained planner checkpoint}"
: "${WORLD_MODEL_CHECKPOINT:?Set WORLD_MODEL_CHECKPOINT to the TA-DWM checkpoint}"
: "${FEATURE_CACHE:?Set FEATURE_CACHE to the cached current/history VAE features}"
: "${METRIC_CACHE:?Set METRIC_CACHE to the NAVSIM metric cache}"
: "${TRAJECTORY_VOCAB:?Set TRAJECTORY_VOCAB to the 256-anchor npy}"
python_bin=${PYTHON_BIN:-python}
"$python_bin" navsim/planning/script/run_training_worlddrive.py \
  agent=future_register_agent \
  agent.checkpoint_path="$PLANNER_CHECKPOINT" \
  agent.worldmodel_checkpoint_path="$WORLD_MODEL_CHECKPOINT" \
  agent.metric_cache_path="$METRIC_CACHE" \
  agent.vocab_path="$TRAJECTORY_VOCAB" \
  agent.lr=0.0002 \
  cache_path="$FEATURE_CACHE" \
  use_cache_without_dataset=true force_cache_computation=false \
  experiment_name=future_registers \
  trainer.params.max_epochs=10 \
  dataloader.params.batch_size=1 \
  dataloader.params.num_workers=4 \
  "$@"
