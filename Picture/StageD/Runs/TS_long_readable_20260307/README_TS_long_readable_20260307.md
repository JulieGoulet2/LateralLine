# TS Long Readable Run (2026-03-07)

This run uses the updated TS plotting scale:
- robust percentile scaling
- hard cap at 50 Hz for TS colormaps

Command:
python3 stage_d_torus_map.py \
  --trials 1000 \
  --tag TS_long_readable \
  --output-dir Picture/StageD/Runs/TS_long_readable_20260307 \
  --ll-mon-topo 0.22 \
  --mon-ts-topo 0.12 \
  --mon-ts-gain-mv 88 \
  --mon-global-inh-mv 1.6 \
  --ts-lat-peak-mv 1.8 \
  --ts-lat-radius 22 \
  --ts-feedback-drive-mv 0.25 \
  --ts-feedback-inh-mv 0.50 \
  --distances-cm 0.8,1.2,1.6 \
  --noise-levels 0.0
