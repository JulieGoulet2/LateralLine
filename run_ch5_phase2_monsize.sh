#!/usr/bin/env bash
# run_ch5_phase2_monsize.sh
#
# Phase 2 for chapter 5 figures: train at different MON sizes to enable Figs 5.1b and 5.3.
#
# Trains at MON = {400, 800, 1600} with topo=0.20 baseline recipe, 3 seeds each.
# After training, runs distance sweep on each new run.
# Existing topo=0.20 with MON=3200 (llmon_topo020_seeds127_132) provides the n=3200 point.
#
# Estimated total time: ~8-10h (training) + ~45min (distance sweep) = ~10h.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

if [[ "${1:-}" != "--_bg" ]]; then
  _LOG="Logs/ch5_phase2_safe.log"
  touch "$_LOG"
  echo "=== $(date) ch5 phase2 starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi
shift

LOG="Logs/ch5_phase2_safe.log"

# ------------------------------------------------------------------
# Shared baseline parameters (topo=0.20, same as gradient run).
# ------------------------------------------------------------------
BASE_TRAIN=(
  --mode ll_thesis
  --use-ll-mon-stdp
  --ll-mon-in-degree 10
  --ll-mon-w-jitter-stdp-mv 8.0
  --ll-mon-w-init-mv 10.0
  --ll-mon-apre 0.010 --ll-mon-apost -0.0105
  --ll-mon-wmax-mv 20.0
  --ll-mon-homeo-eta 0.005
  --mon-ts-homeo-eta 0.001
  --mon-ts-gain-mv 220
  --ts-local-inh-peak-mv 1.5
  --bg-rate-mon-hz 18 --mon-global-inh-mv 1.8
  --n-training-trials 10000
  --training-distance-min-cm 0.8 --training-distance-max-cm 0.8
  --ll-mon-topo 0.20 --mon-ts-topo 0.20
)

# Train each MON size sequentially. run_multi_seed_safe.sh handles OOM-safety.
train_one_size() {
  local NMON="$1"
  local RUNNAME="llmon_nmon${NMON}_seeds123_125"
  echo "=== $(date) PHASE 2 TRAINING: MON=${NMON} -> ${RUNNAME} ===" | tee -a "$LOG"

  # Skip if already trained.
  if [[ -f "Runs/${RUNNAME}/artifacts/latest_seed_125.npz" ]]; then
    echo "    SKIP (already trained)" | tee -a "$LOG"
    return 0
  fi

  # run_multi_seed_safe.sh in --_bg mode (inline, not backgrounded).
  bash run_multi_seed_safe.sh --_bg \
    "${BASE_TRAIN[@]}" \
    --n-mon "$NMON" \
    --run-name "$RUNNAME" \
    --seed-start 123 --multi-seed 3 2>&1 | tee -a "$LOG"

  echo "=== $(date) MON=${NMON} training done ===" | tee -a "$LOG"
}

# Distance sweep on a trained MON size.
sweep_one_size() {
  local NMON="$1"
  local RUNNAME="llmon_nmon${NMON}_seeds123_125"
  local LABEL="nmon${NMON}"
  echo "=== $(date) DISTANCE SWEEP: MON=${NMON} ===" | tee -a "$LOG"

  bash run_distance_sweep_extract.sh --_bg \
    --topo 0.20 \
    --src-run "$RUNNAME" \
    --seeds 123,124,125 \
    --label "$LABEL" \
    --noise-hz 0.0 \
    --nmon "$NMON" 2>&1 | tee -a "$LOG"

  echo "=== $(date) MON=${NMON} sweep done ===" | tee -a "$LOG"
}

# ------------------------------------------------------------------
# Run sequentially: train each size, then sweep all sizes at the end.
# ------------------------------------------------------------------
for NMON in 400 800 1600; do
  train_one_size "$NMON"
done

echo "=== $(date) All training done. Starting distance sweeps. ===" | tee -a "$LOG"

for NMON in 400 800 1600; do
  sweep_one_size "$NMON"
done

# Note: MON=3200 already has distance sweep data from Phase 1 (distswp_topo020).
# We don't need to redo it.

echo "=== $(date) ALL CH5 PHASE 2 COMPLETE ===" | tee -a "$LOG"
osascript -e "display notification \"Phase 2 done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
