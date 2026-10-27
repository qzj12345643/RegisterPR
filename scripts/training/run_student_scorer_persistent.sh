#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
: "${SCORER_RUN_DIR:?Set an absolute directory for this persistent run}"
mkdir -p "$SCORER_RUN_DIR"
exec >>"$SCORER_RUN_DIR/train.log" 2>&1
printf '%s\n' "$$" > "$SCORER_RUN_DIR/supervisor.pid"
log() { printf '[supervisor] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
on_exit() {
  local status=$?
  printf '%s\n' "$status" > "$SCORER_RUN_DIR/exit_code"
  log "exit_code=$status"
}
on_signal() {
  local signal="$1" status="$2"
  log "received $signal; forwarding to training process group ${child:-unset}"
  if [[ -n "${child:-}" ]]; then
    kill -s "$signal" -- "-$child" 2>/dev/null || true
    wait "$child" || true
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP
export SCORER_CONFIG_NAME="${SCORER_CONFIG_NAME:-student_scorer_2gpu}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"
log "started pid=$$ GPUs=${CUDA_VISIBLE_DEVICES:-all} config=$SCORER_CONFIG_NAME"
# A distinct child session lets the supervisor forward signals to every DDP rank.
setsid bash scripts/training/run_student_scorer.sh "output_dir=$SCORER_RUN_DIR" "$@" &
child=$!
printf '%s\n' "$child" > "$SCORER_RUN_DIR/training.pid"
if wait "$child"; then status=0; else status=$?; fi
log "training process exited status=$status"
exit "$status"
