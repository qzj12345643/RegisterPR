#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$REPO_ROOT/outputs/.matplotlib"
exec "$REPO_ROOT/.venv/bin/python" -u scripts/inference_worlddrive.py "$@"
