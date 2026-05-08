#!/usr/bin/env bash
# run_extract_evaluation.sh
#
# Runs test-only (extract-mode) evaluation for all seeds at topo=0.20 and
# topo=0.15, using the saved final weights from each training run.
#
# Each seed takes ~5-10 minutes (no training, just test phase).
# Results are written to per-seed JSON files and the summary log.
#
# Usage: ./run_extract_evaluation.sh
# (no arguments — seeds and sources are hard-coded below)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

# ------------------------------------------------------------------
# Self-relaunch into background with nohup.
# ------------------------------------------------------------------
if [[ "${1:-}" != "--_bg" ]]; then
  _LOG="Logs/extract_evaluation.log"
  touch "$_LOG"
  echo "=== $(date) run_extract_evaluation starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi
shift  # remove --_bg

LOG="Logs/extract_evaluation.log"

# ------------------------------------------------------------------
# Shared baseline parameters (no topo flags — added per group below).
# ------------------------------------------------------------------
BASE_ARGS=(
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
)

# ------------------------------------------------------------------
# Helper: run one extract-mode seed.
# Args: topo seed src_run_name topo_label
# ------------------------------------------------------------------
run_one() {
  local TOPO="$1"
  local SEED="$2"
  local SRC_RUN="$3"
  local LABEL="$4"   # e.g. "topo020" or "topo015"

  local DEST_RUN="extract_${LABEL}_seed_${SEED}"
  local SEED_LOG="Logs/${DEST_RUN}.log"

  echo "--- $(date) extract ${LABEL} seed ${SEED} starting ---" | tee -a "$LOG"
  touch "$SEED_LOG"

  # Create the minimal checkpoint from saved weights.
  python make_extract_checkpoint.py "$SEED" "$SRC_RUN" "$DEST_RUN" >> "$SEED_LOG" 2>&1

  # Run test-only simulation.
  if env PYTHONUNBUFFERED=1 python -u ll_stdp_brian2.py \
      "${BASE_ARGS[@]}" \
      --ll-mon-topo "$TOPO" --mon-ts-topo "$TOPO" \
      --run-name "$DEST_RUN" \
      --seed-start "$SEED" --multi-seed 1 \
      --resume-from "Runs/${DEST_RUN}/" \
      >> "$SEED_LOG" 2>&1; then
    # Extract result line from seed log.
    RESULT=$(grep "PV map quality:" "$SEED_LOG" | tail -1 || echo "no result line found")
    echo "--- $(date) extract ${LABEL} seed ${SEED} DONE: $RESULT ---" | tee -a "$LOG"
    osascript -e "display notification \"extract ${LABEL} seed ${SEED} done\" with title \"LateralLine\"" 2>/dev/null || true
  else
    EXIT=$?
    echo "--- $(date) extract ${LABEL} seed ${SEED} FAILED (exit ${EXIT}) ---" | tee -a "$LOG"
  fi
}

# ------------------------------------------------------------------
# topo=0.20 — 10 seeds
# ------------------------------------------------------------------
echo "=== topo=0.20 extract-mode (10 seeds) ===" | tee -a "$LOG"
run_one 0.20 123 llmon_U_llmonhomeo005_10k         topo020
run_one 0.20 124 llmon_X_U_multiseed3_10k          topo020
run_one 0.20 125 llmon_X_U_multiseed3_10k          topo020
run_one 0.20 126 llmon_X_U_multiseed3_10k          topo020
run_one 0.20 127 llmon_topo020_seeds127_132         topo020
run_one 0.20 128 llmon_topo020_seeds127_132         topo020
run_one 0.20 129 llmon_topo020_seeds127_132         topo020
run_one 0.20 130 llmon_topo020_seeds127_132         topo020
run_one 0.20 131 llmon_topo020_seeds127_132         topo020
run_one 0.20 132 llmon_topo020_seeds127_132         topo020

# ------------------------------------------------------------------
# topo=0.15 — 10 seeds
# ------------------------------------------------------------------
echo "=== topo=0.15 extract-mode (10 seeds) ===" | tee -a "$LOG"
run_one 0.15 123 llmon_Y2_topo015_seed123_10k      topo015
run_one 0.15 124 llmon_Y2_topo015_seed124_10k      topo015
run_one 0.15 125 llmon_Y2_topo015_multiseed3_10k   topo015
run_one 0.15 126 llmon_Y2_topo015_multiseed3_10k   topo015
run_one 0.15 127 llmon_topo015_seeds127_132         topo015
run_one 0.15 128 llmon_topo015_seeds127_132         topo015
run_one 0.15 129 llmon_topo015_seeds127_132         topo015
run_one 0.15 130 llmon_topo015_seeds127_132         topo015
run_one 0.15 131 llmon_topo015_seeds127_132         topo015
run_one 0.15 132 llmon_topo015_seeds127_132         topo015

echo "=== $(date) All extract evaluations done ===" | tee -a "$LOG"
osascript -e "display notification \"All extract evaluations done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
