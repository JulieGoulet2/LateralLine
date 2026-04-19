# LateralLine STDP — Confirmed Baseline (2026-04-19)

## Result

| Metric | Value | Goal |
|--------|-------|------|
| `valid_fraction` | **0.594** | ≥ 0.5 |
| `sigma_theta` | **0.381 rad** | < 0.5 |

Map is real. TS layer forms a somatotopic map. Some boundary effects at edges — acceptable.

---

## Run folder

```
Runs/baseline_repro/
```

Figures are in `Runs/baseline_repro/figures/`.

---

## Exact command to reproduce

```bash
python ll_stdp_brian2.py --mode ll_thesis \
  --run-name baseline_repro \
  --n-training-trials 10000 \
  --ll-mon-topo 0.8 \
  --mon-ts-topo 0.2 \
  --mon-ts-gain-mv 100 \
  --eval-x-min-cm 0.5 \
  --eval-x-max-cm 3.5 \
  > Logs/run_baseline_repro.log 2>&1 &
```

Duration: ~130 min on MacBook Air (M-series).  
Keep Mac awake: run `caffeinate -di &` before starting.

---

## Key parameters (ll_thesis preset — do not change these)

| Parameter | Value | Why it matters |
|-----------|-------|----------------|
| `mon_to_ts_sigma` | **10.0** | NEVER change. 140 destroys the map. |
| `distance_cm` | **0.8** | Test distance. Too large = no TS spikes. |
| `training_distance_max_cm` | **0.8** | Training at near field = strong signal. |
| `mon_ts_apre` | 0.01 | STDP ratio must stay ~-0.6 (net potentiation). |
| `mon_ts_apost` | -0.006 | |
| `ts_local_inh_peak_mV` | 0.9 | Lateral inhibition for map sharpening. |
| `mon_ts_homeo_eta` | 0.02 | Homeostatic weight normalization. |

---

## Next scientific goal

Reduce MON somatotopy: lower `--ll-mon-topo` from 0.8 toward 0.0.

Suggested next run:
```bash
python ll_stdp_brian2.py --mode ll_thesis \
  --run-name topo_llmon_06 \
  --n-training-trials 10000 \
  --ll-mon-topo 0.6 \
  --mon-ts-topo 0.2 \
  --mon-ts-gain-mv 100 \
  --eval-x-min-cm 0.5 \
  --eval-x-max-cm 3.5
```

---

## What was broken and fixed (2026-04-18/19)

Four bugs found in the `ll_thesis` preset after code updates:

1. `mon_to_ts_sigma`: changed to 140 → **reverted to 10.0**
2. `training_distance_max_cm`: changed to 1.8 → **reverted to 0.8**
3. `distance_cm`: default was 1.5 → **set to 0.8 in preset**
4. `mon_ts_homeo_eta`: was accidentally disabled → **re-enabled at 0.02**
