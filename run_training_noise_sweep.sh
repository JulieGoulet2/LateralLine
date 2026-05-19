#!/usr/bin/env bash
# run_training_noise_sweep.sh
#
# Training-phase LL-noise robustness sweep.
#   topo = 0.20, MON = 3200, 10 000 trials, current recipe (post-B1).
#   --training-noise-early = --training-noise-late = scale.
#   5 scales × 2 seeds (123, 124) = 10 runs, ~ 20 h sequential.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

# Self-background.
if [[ "${1:-}" != "--_bg" ]]; then
  _LOG="Logs/training_noise_sweep_safe.log"
  touch "$_LOG"
  echo "=== $(date) training-noise sweep starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi
shift
LOG="Logs/training_noise_sweep_safe.log"

# Common training recipe (topo=0.20 baseline).
BASE_TRAIN=(
  --mode ll_thesis
  --use-ll-mon-stdp
  --ll-mon-in-degree 10
  --ll-mon-w-jitter-stdp-mv 8.0 --ll-mon-w-init-mv 10.0
  --ll-mon-apre 0.010 --ll-mon-apost -0.0105
  --ll-mon-wmax-mv 20.0
  --ll-mon-homeo-eta 0.005 --mon-ts-homeo-eta 0.001
  --mon-ts-gain-mv 220
  --ts-local-inh-peak-mv 1.5
  --bg-rate-mon-hz 18 --mon-global-inh-mv 1.8
  --n-training-trials 10000
  --training-distance-min-cm 0.8 --training-distance-max-cm 0.8
  --ll-mon-topo 0.20 --mon-ts-topo 0.20
)

# Run one (scale, seed) combination. Skips if seed result file already exists.
run_one() {
  local SCALE="$1"
  local TAG="$2"
  local RUNNAME="llmon_trainnoise_${TAG}_seeds123_124"

  if [[ -f "Runs/${RUNNAME}/artifacts/seed_124_results.json" ]]; then
    echo "--- $(date) SKIP $RUNNAME (already done) ---" | tee -a "$LOG"
    return 0
  fi

  echo "=== $(date) TRAIN noise=${SCALE} (tag=${TAG}) ===" | tee -a "$LOG"

  bash run_multi_seed_safe.sh --_bg \
    "${BASE_TRAIN[@]}" \
    --training-noise-early "$SCALE" --training-noise-late "$SCALE" \
    --run-name "$RUNNAME" \
    --seed-start 123 --multi-seed 2 2>&1 | tee -a "$LOG"

  echo "=== $(date) DONE noise=${SCALE} ===" | tee -a "$LOG"
}

# Five noise levels, 2 seeds each. Order from least to most demanding so that
# if the user kills it partway we already have the most-informative early
# datapoints.
run_one 0.0 noise00
run_one 0.3 noise03
run_one 0.5 noise05
run_one 0.8 noise08
run_one 1.0 noise10

echo "=== $(date) ALL TRAINING-NOISE SWEEP COMPLETE ===" | tee -a "$LOG"
osascript -e "display notification \"Training-noise sweep done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
