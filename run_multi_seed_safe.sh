#!/usr/bin/env bash
# Safe multi-seed runner for ll_stdp_brian2.py.
#
# Problem this solves: running --multi-seed N > 2 in one process causes OOM
# because Brian2 accumulates memory across seeds. The OS kills the process
# silently, losing all results after the last completed seed.
#
# Solution: run each seed as a completely separate Python process. Memory is
# fully released between seeds. If one seed is OOM-killed, the others are
# unaffected and their results are already saved.
#
# Usage (identical to run_ll_long.sh):
#   ./run_multi_seed_safe.sh --mode ll_thesis --run-name myrun \
#     --seed-start 127 --multi-seed 6 [all other ll_stdp_brian2 flags...]
#
# Requires: --run-name and --seed-start and --multi-seed
# Each seed writes its own per-seed log:  Logs/<run-name>_seed_NNN.log
# A summary log is written to:           Logs/<run-name>_safe.log

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

# ------------------------------------------------------------------
# Self-relaunch into background with nohup (same pattern as run_ll_long.sh).
# The sentinel flag --_bg tells us we're already in the background.
# ------------------------------------------------------------------
if [[ "${1:-}" != "--_bg" ]]; then
  # Parse --run-name before relaunching, so the log has a good name.
  _RUN_NAME="safe_$(date +%Y%m%d_%H%M%S)"
  _ARGS=("$@")
  for ((i=0; i<${#_ARGS[@]}; i++)); do
    if [[ "${_ARGS[$i]}" == "--run-name" && $((i+1)) -lt ${#_ARGS[@]} ]]; then
      _RUN_NAME="${_ARGS[$((i+1))]}"
      break
    fi
  done
  _LOG="Logs/${_RUN_NAME}_safe.log"
  touch "$_LOG"
  echo "=== $(date) run_multi_seed_safe starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg "$@" >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi

# We're running in the background. Remove the sentinel and parse args.
shift  # remove --_bg

# ------------------------------------------------------------------
# Parse --seed-start, --multi-seed, --run-name; rebuild pass-through args.
# ------------------------------------------------------------------
SEED_START=123
MULTI_SEED=1
RUN_NAME=""
PASS_ARGS=()

args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
  case "${args[$i]}" in
    --seed-start)
      SEED_START="${args[$((i+1))]}"
      i=$((i+2))
      ;;
    --multi-seed)
      MULTI_SEED="${args[$((i+1))]}"
      i=$((i+2))
      ;;
    --run-name)
      RUN_NAME="${args[$((i+1))]}"
      PASS_ARGS+=("--run-name" "$RUN_NAME")
      i=$((i+2))
      ;;
    *)
      PASS_ARGS+=("${args[$i]}")
      i=$((i+1))
      ;;
  esac
done

SUMMARY_LOG="Logs/${RUN_NAME}_safe.log"
echo "Seeds: ${SEED_START} to $((SEED_START + MULTI_SEED - 1))" | tee -a "$SUMMARY_LOG"
echo "Each seed runs as a separate process (OOM-safe)." | tee -a "$SUMMARY_LOG"

FAILED_SEEDS=()

for ((k=0; k<MULTI_SEED; k++)); do
  SEED=$((SEED_START + k))
  SEED_LOG="Logs/${RUN_NAME}_seed_${SEED}.log"
  echo "--- $(date) Seed ${SEED} starting ---" | tee -a "$SUMMARY_LOG"
  touch "$SEED_LOG"

  PYTHON="${PYTHON:-$(command -v python || command -v python3 || echo /Users/juliegoulet/anaconda3/bin/python)}"
  if env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -u ll_stdp_brian2.py \
      "${PASS_ARGS[@]}" --seed-start "$SEED" --multi-seed 1 \
      >> "$SEED_LOG" 2>&1; then
    echo "--- $(date) Seed ${SEED} DONE ---" | tee -a "$SUMMARY_LOG"
    # Print the key metrics line into the summary log for quick scanning.
    grep "Seed ${SEED} results:" "$SEED_LOG" >> "$SUMMARY_LOG" 2>/dev/null || \
      grep "PV map quality:" "$SEED_LOG" | tail -1 >> "$SUMMARY_LOG" 2>/dev/null || true
    osascript -e "display notification \"Seed ${SEED} done\" with title \"LateralLine\"" 2>/dev/null || true
  else
    EXIT=$?
    echo "--- $(date) Seed ${SEED} FAILED (exit ${EXIT}) ---" | tee -a "$SUMMARY_LOG"
    FAILED_SEEDS+=("$SEED")
    osascript -e "display notification \"Seed ${SEED} FAILED\" with title \"LateralLine\" sound name \"Basso\"" 2>/dev/null || true
  fi
done

echo "=== $(date) All ${MULTI_SEED} seeds attempted ===" | tee -a "$SUMMARY_LOG"
if [[ ${#FAILED_SEEDS[@]} -gt 0 ]]; then
  echo "FAILED: ${FAILED_SEEDS[*]}" | tee -a "$SUMMARY_LOG"
  osascript -e "display notification \"Seeds failed: ${FAILED_SEEDS[*]}\" with title \"LateralLine\" sound name \"Basso\"" 2>/dev/null || true
else
  echo "All ${MULTI_SEED} seeds completed successfully." | tee -a "$SUMMARY_LOG"
  osascript -e "display notification \"All ${MULTI_SEED} seeds done.\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
fi
