#!/usr/bin/env bash
# run_ch5_phase2_full_recovery.sh
#
# Three-step Phase 2 recovery:
#   STEP 1: Re-run distance sweep on existing (UNSCALED gain=220) trainings.
#           These will mostly fail (TS silent) but we need the data for the comparison plot.
#   STEP 2: Retrain at scaled gain (gain = 220 * 3200/n_mon).
#   STEP 3: Run distance sweep on scaled trainings.
#
# Gain scaling: each TS cell receives ~N_mon × 16 / 300 inputs.
# At MON=3200 with gain=220 mV the recipe works. To keep mean drive constant
# when MON shrinks, scale gain by 3200 / N_mon:
#   MON=400  → gain=1760
#   MON=800  → gain=880
#   MON=1600 → gain=440
#
# Existing unscaled runs (gain=220) are preserved in llmon_nmon{N}_seeds123_125/.
# New scaled runs go to       llmon_nmon{N}_scaled_seeds123_125/.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

if [[ "${1:-}" != "--_bg" ]]; then
  _LOG="Logs/ch5_phase2_recovery_safe.log"
  touch "$_LOG"
  echo "=== $(date) phase2 recovery starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi
shift
LOG="Logs/ch5_phase2_recovery_safe.log"

# ------------------------------------------------------------------
# Common training recipe.
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
  --ts-local-inh-peak-mv 1.5
  --bg-rate-mon-hz 18 --mon-global-inh-mv 1.8
  --n-training-trials 10000
  --training-distance-min-cm 0.8 --training-distance-max-cm 0.8
  --ll-mon-topo 0.20 --mon-ts-topo 0.20
)

# Scaled gain per MON size (gain = 220 * 3200 / N_mon). Bash 3 compat.
scaled_gain_for() {
  case "$1" in
    400)  echo 1760 ;;
    800)  echo 880  ;;
    1600) echo 440  ;;
    *)    echo 220  ;;
  esac
}

# ==================================================================
# STEP 1 — Distance sweep on existing UNSCALED trainings (gain=220).
# ==================================================================
echo "=== $(date) STEP 1: sweep UNSCALED trainings ===" | tee -a "$LOG"
for NMON in 400 800 1600; do
  RUNNAME="llmon_nmon${NMON}_seeds123_125"
  LABEL="nmon${NMON}"
  if [[ ! -f "Runs/${RUNNAME}/artifacts/latest_seed_125.npz" ]]; then
    echo "    SKIP MON=${NMON} (no training)" | tee -a "$LOG"
    continue
  fi
  echo "--- $(date) sweep UNSCALED MON=${NMON} ---" | tee -a "$LOG"
  bash run_distance_sweep_extract.sh --_bg \
    --topo 0.20 --src-run "$RUNNAME" \
    --seeds 123,124,125 --label "$LABEL" \
    --noise-hz 0.0 --nmon "$NMON" 2>&1 | tee -a "$LOG"
done

# ==================================================================
# STEP 2 — Retrain with SCALED gain.
# ==================================================================
echo "=== $(date) STEP 2: train SCALED-gain trainings ===" | tee -a "$LOG"
for NMON in 400 800 1600; do
  RUNNAME="llmon_nmon${NMON}_scaled_seeds123_125"
  GAIN="$(scaled_gain_for "$NMON")"
  if [[ -f "Runs/${RUNNAME}/artifacts/latest_seed_125.npz" ]]; then
    echo "    SKIP MON=${NMON} (already trained)" | tee -a "$LOG"
    continue
  fi
  echo "--- $(date) train SCALED MON=${NMON} gain=${GAIN} ---" | tee -a "$LOG"
  bash run_multi_seed_safe.sh --_bg \
    "${BASE_TRAIN[@]}" \
    --n-mon "$NMON" \
    --mon-ts-gain-mv "$GAIN" \
    --run-name "$RUNNAME" \
    --seed-start 123 --multi-seed 3 2>&1 | tee -a "$LOG"
done

# ==================================================================
# STEP 3 — Distance sweep on SCALED trainings.
# ==================================================================
echo "=== $(date) STEP 3: sweep SCALED-gain trainings ===" | tee -a "$LOG"
for NMON in 400 800 1600; do
  RUNNAME="llmon_nmon${NMON}_scaled_seeds123_125"
  LABEL="nmon${NMON}_scaled"
  if [[ ! -f "Runs/${RUNNAME}/artifacts/latest_seed_125.npz" ]]; then
    echo "    SKIP MON=${NMON} (no training)" | tee -a "$LOG"
    continue
  fi
  echo "--- $(date) sweep SCALED MON=${NMON} ---" | tee -a "$LOG"
  bash run_distance_sweep_extract.sh --_bg \
    --topo 0.20 --src-run "$RUNNAME" \
    --seeds 123,124,125 --label "$LABEL" \
    --noise-hz 0.0 --nmon "$NMON" 2>&1 | tee -a "$LOG"
done

echo "=== $(date) ALL PHASE 2 RECOVERY COMPLETE ===" | tee -a "$LOG"
osascript -e "display notification \"Phase 2 recovery done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
