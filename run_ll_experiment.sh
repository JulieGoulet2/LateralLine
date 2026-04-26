#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_ll_experiment.sh EXPERIMENT_NAME [extra ll_stdp_brian2.py args...]
#
# Example:
#   ./run_ll_experiment.sh clean_y1p2 \
#       --mode ll_thesis \
#       --ll-rate-mode modulation \
#       --mon-ts-gain-mv 90

NAME="$1"
shift || true

# Where to put things
WEIGHT_DIR="SavedModels/${NAME}"
FIG_DIR="Picture/${NAME}"

mkdir -p "${WEIGHT_DIR}" "${FIG_DIR}"

python ll_stdp_brian2.py \
  --save-tag "${NAME}" \
  --save-weights-dir "${WEIGHT_DIR}" \
  "$@"

# Move just the newly created figures with this tag/seed pattern
# (adjust patterns if you change naming later)
for f in Picture/brian2_*${NAME}* Picture/LL_THESIS_BASELINE_ACTIVE_latest.png; do
  if [ -f "$f" ]; then
    mv "$f" "${FIG_DIR}/"
  fi
done

echo "Weights/params in: ${WEIGHT_DIR}"
echo "Figures in:        ${FIG_DIR}"