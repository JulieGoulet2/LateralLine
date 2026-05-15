#!/usr/bin/env bash
# run_ch5_sweeps_master.sh
#
# Run all distance sweeps needed for chapter 5 figures in sequence:
#   1) topo=0.10, seeds 126,127,128, σ_noise=0      (Fig 5.1a)
#   2) topo=0.40, seeds 123,124,125, σ_noise=0      (Fig 5.1a)
#   3) topo=0.80, seeds 123,124,125, σ_noise=0      (Fig 5.1a)
#   4) topo=0.20, seeds 127,128,129, σ_noise=2 Hz   (Fig 5.4, 5.5)
#   5) topo=0.20, seeds 127,128,129, σ_noise=5 Hz   (Fig 5.4, 5.5)
#
# topo=0.20 σ_noise=0 already done in distswp_topo020_safe.log
#
# Each test run: ~30s. 5 sweeps × ~30 runs each = ~75 min total.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p Logs

# Self-background with nohup + caffeinate.
if [[ "${1:-}" != "--_bg" ]]; then
  _LOG="Logs/ch5_master_safe.log"
  touch "$_LOG"
  echo "=== $(date) ch5 sweeps master starting ===" >> "$_LOG"
  nohup caffeinate -dis bash "$0" --_bg >> "$_LOG" 2>&1 &
  PID=$!
  echo "Started PID $PID"
  echo "Log:   $ROOT/$_LOG"
  echo "Watch: tail -f \"$ROOT/$_LOG\""
  echo "Stop:  kill $PID"
  exit 0
fi
shift

LOG="Logs/ch5_master_safe.log"

# Helper to call run_distance_sweep_extract.sh INLINE (not backgrounded).
# We strip the --_bg self-relaunch by calling its inner logic directly via bash.
run_sweep_inline() {
  local LABEL="$1"
  shift
  echo "=== $(date) starting sweep: $LABEL ===" | tee -a "$LOG"
  # Call the sweep script with --_bg flag set so it skips self-relaunch and runs inline.
  bash run_distance_sweep_extract.sh --_bg "$@" 2>&1 | tee -a "$LOG"
  echo "=== $(date) sweep $LABEL done ===" | tee -a "$LOG"
}

run_sweep_inline "topo010_n0" \
  --topo 0.10 --src-run llmon_topo010_seeds126_132 \
  --seeds 126,127,128 --label topo010 --noise-hz 0.0

run_sweep_inline "topo040_n0" \
  --topo 0.40 --src-run llmon_topo040_seeds123_132 \
  --seeds 123,124,125 --label topo040 --noise-hz 0.0

run_sweep_inline "topo080_n0" \
  --topo 0.80 --src-run llmon_topo080_seeds123_132 \
  --seeds 123,124,125 --label topo080 --noise-hz 0.0

run_sweep_inline "topo020_n2" \
  --topo 0.20 --src-run llmon_topo020_seeds127_132 \
  --seeds 127,128,129 --label topo020n2 --noise-hz 2.0

run_sweep_inline "topo020_n5" \
  --topo 0.20 --src-run llmon_topo020_seeds127_132 \
  --seeds 127,128,129 --label topo020n5 --noise-hz 5.0

echo "=== $(date) ALL CH5 SWEEPS COMPLETE ===" | tee -a "$LOG"
osascript -e "display notification \"All ch5 sweeps done\" with title \"LateralLine\" sound name \"Glass\"" 2>/dev/null || true
