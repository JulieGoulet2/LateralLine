#!/usr/bin/env bash
# Long-run helper for ll_stdp_brian2.py:
#   - PYTHONUNBUFFERED=1 + python -u  → log lines appear immediately
#   - nohup  → survives closing the terminal (SIGHUP)
#   - caffeinate -dis  → prevents idle sleep AND lid-close sleep (requires AC power)
#
# Usage (all arguments go to ll_stdp_brian2.py):
#   ./run_ll_long.sh --mode ll_thesis --n-training-trials 10000 ...
#
# Custom log path:
#   LL_RUN_LOG=run_center_llmon_moderate.log ./run_ll_long.sh --mode ll_thesis ...
#
# Default log: Logs/run_long_YYYYMMDD_HHMMSS.log

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p Logs
LOG="${LL_RUN_LOG:-Logs/run_long_$(date +%Y%m%d_%H%M%S).log}"
touch "$LOG"
echo "=== $(date) starting ll_stdp_brian2 pid will be below ===" >>"$LOG"

# Extract --run-name from arguments so the resume script knows the checkpoint path.
RUN_NAME=""
args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--run-name" && $((i+1)) -lt ${#args[@]} ]]; then
    RUN_NAME="${args[$((i+1))]}"
    break
  fi
done

# Write a RESUME.sh next to the log file before launching.
RESUME_SCRIPT="${LOG%.log}_RESUME.sh"
{
  echo "#!/usr/bin/env bash"
  echo "# Auto-generated resume script — run from: $ROOT"
  echo "# Generated: $(date)"
  echo "cd \"$ROOT\" || exit 1"
  if [[ -n "$RUN_NAME" ]]; then
    RESUME_LOG="${LOG%.log}_resume.log"
    printf 'LL_RUN_LOG=%s ./run_ll_long.sh' "$RESUME_LOG"
    for arg in "$@"; do printf ' %q' "$arg"; done
    printf ' \\\n  --resume-from Runs/%s/\n' "$RUN_NAME"
  else
    printf 'LL_RUN_LOG=%s ./run_ll_long.sh' "${LOG%.log}_resume.log"
    for arg in "$@"; do printf ' %q' "$arg"; done
    printf '\n'
    echo "# NOTE: --run-name was not passed; add --resume-from Runs/<run_name>/ manually."
  fi
} > "$RESUME_SCRIPT"
chmod +x "$RESUME_SCRIPT"
echo "Resume script: $ROOT/$RESUME_SCRIPT"

nohup env PYTHONUNBUFFERED=1 caffeinate -dis python -u ll_stdp_brian2.py "$@" >>"$LOG" 2>&1 &
PID=$!

# Watcher: waits for the simulation to finish, then sends a macOS notification.
(
  wait "$PID" 2>/dev/null
  EXIT_CODE=$?
  if [ "$EXIT_CODE" -eq 0 ]; then
    MSG="Simulation finished successfully."
  else
    MSG="Simulation ended with error (exit $EXIT_CODE). Check log."
  fi
  echo "=== $(date) $MSG ===" >>"$LOG"
  osascript -e "display notification \"$MSG\" with title \"LateralLine\" sound name \"Glass\""
) &

echo "Started PID $PID"
echo "Log file: $ROOT/$LOG"
echo "Watch:    tail -f \"$ROOT/$LOG\""
echo "Stop:     kill $PID"
echo ""
echo "Note: If the job still dies with \"killed\", check Memory in Activity Monitor"
echo "      (OOM). This script does not reduce RAM use."
