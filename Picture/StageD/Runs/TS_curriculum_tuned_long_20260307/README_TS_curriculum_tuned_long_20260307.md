# TS curriculum tuned long run

- curriculum enabled
- temporary topographic prior with decay
- homeostatic incoming normalization
- TS readout colormap capped at 50 Hz

Command:
python3 stage_d_torus_map.py \
  --trials 1000 \
  --tag TS_curriculum_tuned_long \
  --output-dir Picture/StageD/Runs/TS_curriculum_tuned_long_20260307 \
  --use-curriculum \
  --curriculum-phase1-frac 0.45 \
  --curriculum-phase2-frac 0.55 \
  --curriculum-fixed-distance-cm 1.2 \
  --curriculum-final-noise-scale 0.0 \
  --prior-boost-strength 0.30 \
  --prior-decay-fraction 0.45 \
  --homeo-eta 0.08 \
  --homeo-every-trials 5 \
  --ll-mon-topo 0.22 \
  --mon-ts-topo 0.16 \
  --mon-ts-gain-mv 70 \
  --mon-global-inh-mv 1.6 \
  --ts-lat-peak-mv 2.4 \
  --ts-lat-radius 22 \
  --ts-feedback-drive-mv 0.25 \
  --ts-feedback-inh-mv 0.90 \
  --distances-cm 0.8,1.2,1.6 \
  --noise-levels 0.0
