# TS Long Run: TS_long_L22F

## Purpose
Long Stage-D TS learning run with weak MON->TS topography and lateral inhibition/feedback enabled.

## Command
`python3 stage_d_torus_map.py --trials 1000 --tag TS_long_L22F --output-dir Picture/StageD/Runs/TS_long_L22F_20260305 --ll-mon-topo 0.22 --mon-ts-topo 0.12 --mon-ts-gain-mv 88 --mon-global-inh-mv 1.6 --ts-lat-peak-mv 1.8 --ts-lat-radius 22 --ts-feedback-drive-mv 0.25 --ts-feedback-inh-mv 0.50 --distances-cm 0.8,1.2,1.6 --noise-levels 0.0`

## Key Parameters
- train trials: 1000
- LL->MON topography: 0.22
- MON->TS topography: 0.12
- MON->TS gain (mV): 88
- MON global inhibition (mV): 1.6
- TS lateral inhibition peak (mV): 1.8
- TS lateral inhibition radius: 22
- TS feedback drive (mV): 0.25
- TS feedback inhibition (mV): 0.50
- test distances (cm): 0.8, 1.2, 1.6
- test noise levels: 0.0

## Expected Outputs (written by script at end)
- `stageD_torus_map_no_noise_before_TS_long_L22F.png`
- `stageD_torus_map_no_noise_after_TS_long_L22F.png`
- `stageD_torus_metrics_TS_long_L22F.csv`
- `stageD_weight_stabilization_TS_long_L22F.png`
- `stageD_weight_stabilization_TS_long_L22F.csv`
- `stageD_before_after_time_noise_0.00_TS_long_L22F.png`
- `stageD_dynamic_time_activity_TS_long_L22F.png`
- `RUN_SUMMARY_TS_long_L22F.md`

## Status
- Started: 2026-03-05
- Runtime: in progress (long run)
