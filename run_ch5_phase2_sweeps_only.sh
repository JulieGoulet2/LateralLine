#!/usr/bin/env bash
# Sweep-only re-launch for Phase 2 (training already complete; bug fixed in
# run_distance_sweep_extract.sh — now passes --n-mon through).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

if [[ "${1:-}" != "--_bg" ]]; then
  _LOG="Logs/ch5_phase2_sweeps_safe.log"
  touch "$_LOG"
  echo "=== $(date) phase2 sweeps only starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi
shift
LOG="Logs/ch5_phase2_sweeps_safe.log"

for NMON in 400 800 1600; do
  RUNNAME="llmon_nmon${NMON}_seeds123_125"
  LABEL="nmon${NMON}"
  echo "=== $(date) sweep MON=${NMON} ===" | tee -a "$LOG"
  bash run_distance_sweep_extract.sh --_bg \
    --topo 0.20 --src-run "$RUNNAME" \
    --seeds 123,124,125 --label "$LABEL" \
    --noise-hz 0.0 --nmon "$NMON" 2>&1 | tee -a "$LOG"
  echo "=== $(date) MON=${NMON} sweep done ===" | tee -a "$LOG"
done

echo "=== $(date) ALL PHASE 2 SWEEPS COMPLETE ===" | tee -a "$LOG"
osascript -e "display notification \"Phase 2 sweeps done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
