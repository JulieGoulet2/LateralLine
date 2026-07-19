#!/usr/bin/env bash
# run_stimvar_extract.sh
#
# Step 2 (Q4): test-only (extract-mode) evaluation of a PRE-TRAINED network under
# varied TEST stimuli — direction, speed, and object size. NO retraining: the saved
# weights are frozen; only the test sweep stimulus changes.
#
# For each seed × each stimulus condition:
#   - Creates a minimal extract checkpoint from saved weights (once per seed)
#   - Runs ll_stdp_brian2.py test phase only with the condition's extra flag(s)
#   - Writes results to Runs/stimvar_<label>_seed<NNN>_<cond>/artifacts/seed_NNN_results.json
#
# Usage:
#   ./run_stimvar_extract.sh \
#       --topo 0.20 \
#       --src-run llmon_topo020_seeds127_132 \
#       --seeds 127,128,129,130,131,132 \
#       --label topo020
#
# Conditions swept (baseline = forward, 5 cm/s, sphere r=0.5 cm):
#   base        : (baseline, all defaults)
#   dir_back    : --direction -1
#   speed_slow  : --speed-cm-s 2.5
#   speed_fast  : --speed-cm-s 10
#   size_small  : --sphere-radius-cm 0.3
#   size_big    : --sphere-radius-cm 0.7

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

# Python interpreter: default to the anaconda base env (has Brian2). Bare `python`
# only works when conda base is on PATH; override with LL_PYTHON if you migrate envs.
PYTHON="${LL_PYTHON:-/Users/juliegoulet/anaconda3/bin/python}"

# Self-relaunch into background with nohup + caffeinate.
if [[ "${1:-}" != "--_bg" ]]; then
  _ARGS=("$@")
  _LABEL="stimvar"
  for i in "${!_ARGS[@]}"; do
    if [[ "${_ARGS[$i]}" == "--label" ]]; then _LABEL="stimvar_${_ARGS[$((i+1))]}"; fi
  done
  _LOG="Logs/${_LABEL}_safe.log"
  touch "$_LOG"
  echo "=== $(date) run_stimvar_extract starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg "$@" >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi
shift  # remove --_bg

TOPO=""; SRC_RUN=""; SEEDS_STR=""; LABEL=""; NOISE_HZ="0.0"; NMON="3200"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --topo)     TOPO="$2";      shift 2 ;;
    --src-run)  SRC_RUN="$2";   shift 2 ;;
    --seeds)    SEEDS_STR="$2"; shift 2 ;;
    --label)    LABEL="$2";     shift 2 ;;
    --noise-hz) NOISE_HZ="$2";  shift 2 ;;
    --nmon)     NMON="$2";      shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[[ -z "$TOPO" ]]      && { echo "ERROR: --topo required"; exit 1; }
[[ -z "$SRC_RUN" ]]   && { echo "ERROR: --src-run required"; exit 1; }
[[ -z "$SEEDS_STR" ]] && { echo "ERROR: --seeds required"; exit 1; }
[[ -z "$LABEL" ]]     && LABEL="$SRC_RUN"

LOG="Logs/stimvar_${LABEL}_safe.log"
IFS=',' read -ra SEEDS <<< "$SEEDS_STR"

# Stimulus conditions: "cond_label|extra flags".
CONDITIONS=(
  "base|"
  "dir_back|--direction -1"
  "speed_slow|--speed-cm-s 2.5"
  "speed_fast|--speed-cm-s 10"
  "size_small|--sphere-radius-cm 0.3"
  "size_big|--sphere-radius-cm 0.7"
)

echo "=== $(date) stimvar sweep: label=$LABEL topo=$TOPO src=$SRC_RUN ===" | tee -a "$LOG"
echo "    Seeds:      ${SEEDS[*]}" | tee -a "$LOG"
echo "    Conditions: ${CONDITIONS[*]}" | tee -a "$LOG"

# Baseline recipe (same as all topo gradient runs). Distance fixed at the trained 0.8 cm.
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
  --distance-cm 0.8
)

for SEED in "${SEEDS[@]}"; do
  EXTRACT_RUN="stimvar_${LABEL}_seed${SEED}_extract"
  SEED_LOG="Logs/${EXTRACT_RUN}.log"; touch "$SEED_LOG"
  echo "--- $(date) Creating extract checkpoint for seed ${SEED} ---" | tee -a "$LOG"
  "$PYTHON" make_extract_checkpoint.py "$SEED" "$SRC_RUN" "$EXTRACT_RUN" >> "$SEED_LOG" 2>&1

  for COND in "${CONDITIONS[@]}"; do
    CLABEL="${COND%%|*}"
    CFLAGS_STR="${COND#*|}"
    read -ra CFLAGS <<< "$CFLAGS_STR"
    RUN_NAME="stimvar_${LABEL}_seed${SEED}_${CLABEL}"
    COND_LOG="Logs/${RUN_NAME}.log"
    RESULT_DIR="Runs/${RUN_NAME}/artifacts"

    if [[ -f "${RESULT_DIR}/seed_${SEED}_results.json" ]]; then
      echo "--- $(date) seed ${SEED} ${CLABEL} SKIP (result exists) ---" | tee -a "$LOG"
      continue
    fi
    mkdir -p "$RESULT_DIR"
    cp "Runs/${EXTRACT_RUN}/artifacts/mid_checkpoint.npz" "${RESULT_DIR}/mid_checkpoint.npz"
    touch "$COND_LOG"
    echo "--- $(date) seed ${SEED} ${CLABEL} [${CFLAGS_STR}] starting ---" | tee -a "$LOG"

    if env PYTHONUNBUFFERED=1 "$PYTHON" -u ll_stdp_brian2.py \
        "${BASE_ARGS[@]}" ${CFLAGS[@]+"${CFLAGS[@]}"} \
        --test-ll-noise-hz "$NOISE_HZ" \
        --run-name "$RUN_NAME" \
        --seed-start "$SEED" --multi-seed 1 \
        --resume-from "Runs/${RUN_NAME}/" \
        >> "$COND_LOG" 2>&1; then
      RESULT=$("$PYTHON" -c "
import json
try:
    d = json.load(open('${RESULT_DIR}/seed_${SEED}_results.json'))
    print(f\"sigma={d.get('sigma_theta_rad',float('nan')):.3f} valid={d.get('valid_fraction',float('nan')):.3f}\")
except Exception as e:
    print(f'parse error: {e}')
" 2>/dev/null || echo "no result")
      echo "--- $(date) seed ${SEED} ${CLABEL} DONE: $RESULT ---" | tee -a "$LOG"
    else
      EXIT=$?
      echo "--- $(date) seed ${SEED} ${CLABEL} FAILED (exit ${EXIT}) ---" | tee -a "$LOG"
    fi
  done
done

echo "=== $(date) stimvar sweep ${LABEL} complete ===" | tee -a "$LOG"
osascript -e "display notification \"Stimvar sweep ${LABEL} done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
