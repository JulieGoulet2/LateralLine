#!/usr/bin/env bash
# run_extract_topo010.sh
#
# Extract-mode (test-only) evaluation for all available topo=0.10 seeds.
# Uses saved final weights from each training run.
#
# Coverage:
#   seed 123      -> Runs/topo010_10k/                  (only seed with saved weights from old Y4 run)
#   seed 126-132  -> Runs/llmon_topo010_seeds126_132/   (new safe-multiseed run)
#   seeds 124,125 -> NOT AVAILABLE (no saved weights from old Y4 run)
#
# Usage: ./run_extract_topo010.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

LOG="Logs/extract_topo010.log"
echo "=== $(date) extract topo=0.10 starting ===" | tee -a "$LOG"

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

run_one() {
  local SEED="$1"
  local SRC_RUN="$2"
  local DEST_RUN="extract_topo010_seed_${SEED}"
  local SEED_LOG="Logs/${DEST_RUN}.log"

  echo "--- $(date) extract topo010 seed ${SEED} starting ---" | tee -a "$LOG"
  touch "$SEED_LOG"

  python make_extract_checkpoint.py "$SEED" "$SRC_RUN" "$DEST_RUN" >> "$SEED_LOG" 2>&1

  if env PYTHONUNBUFFERED=1 python -u ll_stdp_brian2.py \
      "${BASE_ARGS[@]}" \
      --ll-mon-topo 0.10 --mon-ts-topo 0.10 \
      --run-name "$DEST_RUN" \
      --seed-start "$SEED" --multi-seed 1 \
      --resume-from "Runs/${DEST_RUN}/" \
      >> "$SEED_LOG" 2>&1; then
    RESULT=$(grep "PV map quality:" "$SEED_LOG" | tail -1 || echo "no result line found")
    echo "--- $(date) extract topo010 seed ${SEED} DONE: $RESULT ---" | tee -a "$LOG"
  else
    EXIT=$?
    echo "--- $(date) extract topo010 seed ${SEED} FAILED (exit ${EXIT}) ---" | tee -a "$LOG"
  fi
}

run_one 123 topo010_10k
run_one 126 llmon_topo010_seeds126_132
run_one 127 llmon_topo010_seeds126_132
run_one 128 llmon_topo010_seeds126_132
run_one 129 llmon_topo010_seeds126_132
run_one 130 llmon_topo010_seeds126_132
run_one 131 llmon_topo010_seeds126_132
run_one 132 llmon_topo010_seeds126_132

echo "=== $(date) All topo=0.10 extract evaluations done ===" | tee -a "$LOG"
osascript -e "display notification \"topo=0.10 extract done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
