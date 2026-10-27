#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
: "${EVAL_RUN_DIR:?Set an absolute output directory}"
: "${STUDENT_SCORER_CHECKPOINT:?Set the trained student scoring checkpoint}"
mkdir -p "$EVAL_RUN_DIR"
exec >>"$EVAL_RUN_DIR/eval.log" 2>&1
printf '%s\n' "$$" > "$EVAL_RUN_DIR/supervisor.pid"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG=WARN
log() { printf '[supervisor] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
on_exit() {
  local status=$?
  printf '%s\n' "$status" > "$EVAL_RUN_DIR/exit_code"
  log "exit_code=$status"
}
on_signal() {
  local signal="$1" status="$2"
  log "received $signal"
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
log "started pid=$$ GPUs=${CUDA_VISIBLE_DEVICES:-all} checkpoint=$STUDENT_SCORER_CHECKPOINT"
setsid bash scripts/evaluation/run_student_scorer_pdm_score.sh "output_dir=$EVAL_RUN_DIR" "$@" &
child=$!
printf '%s\n' "$child" > "$EVAL_RUN_DIR/torchrun.pid"
if wait "$child"; then status=0; else status=$?; fi
exit "$status"
