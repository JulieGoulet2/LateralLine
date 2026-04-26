# LateralLine STDP — Confirmed Baseline (2026-04-21)

## Result

| Metric | Value | Goal |
|--------|-------|------|
| `valid_fraction` | **0.571** | ≥ 0.5 |
| `sigma_theta` | **0.376 rad** | < 0.5 |
| `TS spikes (test)` | **3162** | — |

Map is real. LL→MON STDP enabled — network learns LL→MON connectivity from activity.
This is the new baseline for reducing MON somatotopy.

---

## Run folder

```
Runs/monton02_llmon_stdp_gain120_full/
```

Figures are in `Runs/monton02_llmon_stdp_gain120_full/figures/`.

---

## Exact command to reproduce

```bash
python ll_stdp_brian2.py --mode ll_thesis \
  --run-name monton02_llmon_stdp_gain120_full \
  --n-training-trials 10000 \
  --ll-mon-topo 0.8 \
  --mon-ts-topo 0.2 \
  --mon-ts-gain-mv 120 \
  --eval-x-min-cm 0.5 \
  --eval-x-max-cm 3.5 \
  --use-ll-mon-stdp \
  --ll-mon-apre 0.005 \
  --ll-mon-apost -0.004 \
  > Logs/run_monton02_llmon_stdp_gain120_full.log 2>&1 &
```

Duration: ~130 min on MacBook Air (M-series).
Keep Mac awake: run `caffeinate -di &` before starting.

---

## Key parameters

| Parameter | Value | Why it matters |
|-----------|-------|----------------|
| `mon_to_ts_sigma` | **10.0** | NEVER change. 140 destroys the map. |
| `distance_cm` | **0.8** | Test distance. |
| `training_distance_max_cm` | **0.8** | Near-field training. |
| `mon_ts_apre` | 0.01 | MON→TS STDP (net potentiation). |
| `mon_ts_apost` | -0.006 | |
| `ts_local_inh_peak_mV` | 0.9 | Lateral inhibition for map sharpening. |
| `mon_ts_homeo_eta` | 0.02 | Homeostatic weight normalization. |
| `mon_ts_gain_mV` | **120** | Gain tuned for good TS activity with STDP. |
| `ll_mon_apre` | **0.005** | LL→MON STDP learning rate. |
| `ll_mon_apost` | **-0.004** | Net-potentiating (ratio = 0.80). |

---

## Previous baseline (2026-04-19)

| Metric | Value |
|--------|-------|
| Run | `Runs/baseline_repro/` |
| `valid_fraction` | 0.594 |
| `sigma_theta` | 0.381 rad |
| Command | `--ll-mon-topo 0.8 --mon-ts-topo 0.2 --mon-ts-gain-mv 100` (no STDP) |

---

## Experiments run from this baseline (2026-04-26)

Tried reducing LL→MON somatotopy to 0.2 with LL→MON STDP (standard and 2× stronger):

| Run | ll_mon_topo | sigma_theta | valid_fraction | Verdict |
|-----|-------------|-------------|----------------|---------|
| stdp_topo02_with_v2 | 0.2 | 0.841 | 0.697 | **worse** |
| llmon_stdp_strong_10k (2× STDP) | 0.2 | 0.673 | 0.675 | **worse** |

Reducing LL→MON topography degrades map quality even with strong STDP. LL→MON STDP alone cannot compensate for lost anatomical somatotopy.

## Next scientific goal

Improve map quality from the confirmed baseline. Options to try:
- Increase TS lateral inhibition (`ts_local_inh_peak_mV` > 0.9) to sharpen tuning
- Tune MON→TS STDP rates (currently apre=0.01, apost=-0.006)
- Increase training trials beyond 10k
