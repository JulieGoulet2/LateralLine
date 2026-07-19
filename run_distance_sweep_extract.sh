#!/usr/bin/env bash
# run_distance_sweep_extract.sh
#
# Run test-only (extract-mode) evaluation at multiple stimulus distances D
# for a pre-trained network. Uses saved weights from a training run.
#
# For each seed × each distance D:
#   - Creates a minimal extract checkpoint from saved weights (once per seed)
#   - Runs ll_stdp_brian2.py test phase only with --distance-cm D
#   - Writes results to Runs/distswp_<label>/seed_NNN_dXXX_results.json
#
# Usage:
#   ./run_distance_sweep_extract.sh \
#       --topo 0.20 \
#       --src-run llmon_topo020_seeds127_132 \
#       --seeds 127,128,129 \
#       --label topo020
#
# Optional:
#   --distances "0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0,2.5,3.0"  (default)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

# Python interpreter: default to the anaconda base env (has Brian2). Bare `python`
# only works when conda base is on PATH; override with LL_PYTHON if you migrate envs.
PYTHON="${LL_PYTHON:-/Users/juliegoulet/anaconda3/bin/python}"

# ------------------------------------------------------------------
# Self-relaunch into background with nohup + caffeinate.
# ------------------------------------------------------------------
if [[ "${1:-}" != "--_bg" ]]; then
  _ARGS=("$@")
  # Derive a log name from the label argument if present.
  _LABEL="distswp"
  for i in "${!_ARGS[@]}"; do
    if [[ "${_ARGS[$i]}" == "--label" ]]; then
      _LABEL="distswp_${_ARGS[$((i+1))]}"
    fi
  done
  _LOG="Logs/${_LABEL}_safe.log"
  touch "$_LOG"
  echo "=== $(date) run_distance_sweep_extract starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg "$@" >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi
shift  # remove --_bg

# ------------------------------------------------------------------
# Parse arguments.
# ------------------------------------------------------------------
TOPO=""
SRC_RUN=""
SEEDS_STR=""
LABEL=""
DISTANCES_STR="0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0,2.5,3.0"
NOISE_HZ="0.0"
NMON="3200"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topo)       TOPO="$2";       shift 2 ;;
    --src-run)    SRC_RUN="$2";    shift 2 ;;
    --seeds)      SEEDS_STR="$2";  shift 2 ;;
    --label)      LABEL="$2";      shift 2 ;;
    --distances)  DISTANCES_STR="$2"; shift 2 ;;
    --noise-hz)   NOISE_HZ="$2";   shift 2 ;;
    --nmon)       NMON="$2";       shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "$TOPO" ]]     && { echo "ERROR: --topo required"; exit 1; }
[[ -z "$SRC_RUN" ]]  && { echo "ERROR: --src-run required"; exit 1; }
[[ -z "$SEEDS_STR" ]] && { echo "ERROR: --seeds required"; exit 1; }
[[ -z "$LABEL" ]]    && LABEL="$SRC_RUN"

LOG="Logs/distswp_${LABEL}_safe.log"

IFS=',' read -ra SEEDS     <<< "$SEEDS_STR"
IFS=',' read -ra DISTANCES <<< "$DISTANCES_STR"

echo "=== $(date) distance sweep: label=$LABEL topo=$TOPO src=$SRC_RUN ===" | tee -a "$LOG"
echo "    Seeds:     ${SEEDS[*]}"     | tee -a "$LOG"
echo "    Distances: ${DISTANCES[*]}" | tee -a "$LOG"

# ------------------------------------------------------------------
# Shared baseline parameters (same recipe as all topo gradient runs).
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
  --ll-mon-topo "$TOPO" --mon-ts-topo "$TOPO"
  --n-mon "$NMON"
)

# ------------------------------------------------------------------
# Main loop: for each seed, for each distance.
# ------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
  # Create extract checkpoint once per seed.
  EXTRACT_RUN="distswp_${LABEL}_seed${SEED}_extract"
  SEED_LOG="Logs/${EXTRACT_RUN}.log"
  touch "$SEED_LOG"

  echo "--- $(date) Creating extract checkpoint for seed ${SEED} ---" | tee -a "$LOG"
  "$PYTHON" make_extract_checkpoint.py "$SEED" "$SRC_RUN" "$EXTRACT_RUN" >> "$SEED_LOG" 2>&1

  for D in "${DISTANCES[@]}"; do
    # Format distance as integer hundredths for filenames: 0.2 -> d020, 1.5 -> d150
    D_STR=$("$PYTHON" -c "print(f'd{round(float(\"$D\")*100):03d}')")
    RUN_NAME="distswp_${LABEL}_seed${SEED}_${D_STR}"
    DIST_LOG="Logs/${RUN_NAME}.log"
    RESULT_DIR="Runs/${RUN_NAME}/artifacts"

    # Skip if result already exists.
    if [[ -f "${RESULT_DIR}/seed_${SEED}_results.json" ]]; then
      echo "--- $(date) seed ${SEED} D=${D} SKIP (result exists) ---" | tee -a "$LOG"
      continue
    fi

    # Copy the extract checkpoint into the per-distance run directory.
    mkdir -p "$RESULT_DIR"
    cp "Runs/${EXTRACT_RUN}/artifacts/mid_checkpoint.npz" "${RESULT_DIR}/mid_checkpoint.npz"

    touch "$DIST_LOG"
    echo "--- $(date) seed ${SEED} D=${D} cm starting ---" | tee -a "$LOG"

    if env PYTHONUNBUFFERED=1 "$PYTHON" -u ll_stdp_brian2.py \
        "${BASE_ARGS[@]}" \
        --distance-cm "$D" \
        --test-ll-noise-hz "$NOISE_HZ" \
        --run-name "$RUN_NAME" \
        --seed-start "$SEED" --multi-seed 1 \
        --resume-from "Runs/${RUN_NAME}/" \
        >> "$DIST_LOG" 2>&1; then
      # Extract key metrics from result JSON.
      RESULT=$("$PYTHON" -c "
import json, sys
try:
    d = json.load(open('${RESULT_DIR}/seed_${SEED}_results.json'))
    print(f\"sigma={d.get('sigma_theta_rad',float('nan')):.3f} valid={d.get('valid_fraction',float('nan')):.3f} sw_ll={d.get('sigma_w_ll_cm',float('nan')):.3f} sw_ts={d.get('sigma_w_ts_cm',float('nan')):.3f}\")
except Exception as e:
    print(f'parse error: {e}')
" 2>/dev/null || echo "no result")
      echo "--- $(date) seed ${SEED} D=${D} DONE: $RESULT ---" | tee -a "$LOG"
    else
      EXIT=$?
      echo "--- $(date) seed ${SEED} D=${D} FAILED (exit ${EXIT}) ---" | tee -a "$LOG"
    fi
  done
done

echo "=== $(date) distance sweep ${LABEL} complete ===" | tee -a "$LOG"
osascript -e "display notification \"Distance sweep ${LABEL} done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
