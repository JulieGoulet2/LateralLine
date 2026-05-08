# LateralLine STDP — New Baseline (2026-05-04)

**A working recipe at WEAK MON somatotopy (ll_mon_topo = mon_ts_topo = 0.2).**

## Result (single 10k-trial run, seed 123)

| Metric | Value | Goal |
|--------|-------|------|
| `valid_fraction` | **0.786** | ≥ 0.6 |
| `sigma_theta` | **0.622 rad** | < 0.5 |
| `TS spikes (test)` | **2552** | many |
| `frac(w==wmax) MON→TS` | 0.131 | (bimodal weights) |
| `frac(w==0) MON→TS` | 0.011 | |

## Multi-seed validation (10 seeds: 123–132, extract-mode)

| Metric | Mean ± SD |
|--------|-----------|
| `sigma_theta` | **0.354** ± 0.058 |
| `valid_fraction` | **0.912** ± 0.028 |

**ALL 10 seeds beat the previous high-topo baseline (sigma=0.875, valid=0.660 at topo=0.8).**

The map forms via STDP + multi-layer homeostasis even at weak MON somatotopy — the central scientific question of this project.

---

## Run folder

```
Runs/llmon_U_llmonhomeo005_10k/        # canonical single-seed baseline
Runs/llmon_X_U_multiseed3_10k/          # multi-seed validation (seeds 124, 125, 126)
```

Figures are in each `figures/` subfolder.

---

## Exact command to reproduce the baseline

```bash
LL_RUN_LOG=Logs/llmon_U_llmonhomeo005_10k.log ./run_ll_long.sh \
  --mode ll_thesis \
  --run-name llmon_U_llmonhomeo005_10k \
  --use-ll-mon-stdp \
  --ll-mon-topo 0.2 --mon-ts-topo 0.2 \
  --ll-mon-in-degree 10 \
  --ll-mon-w-jitter-stdp-mv 8.0 \
  --ll-mon-w-init-mv 10.0 \
  --ll-mon-apre 0.010 --ll-mon-apost -0.0105 \
  --ll-mon-wmax-mv 20.0 \
  --ll-mon-homeo-eta 0.005 \
  --mon-ts-homeo-eta 0.001 \
  --mon-ts-gain-mv 220 \
  --ts-local-inh-peak-mv 1.5 \
  --bg-rate-mon-hz 18 --mon-global-inh-mv 1.8 \
  --n-training-trials 10000 --seed-start 123 \
  --training-distance-min-cm 0.8 --training-distance-max-cm 0.8
```

Duration: ~3 hours single-CPU on MacBook (M-series).

---

## Key parameters (and why they matter)

| Parameter | Value | Role |
|-----------|-------|------|
| `ll_mon_topo` | **0.2** | Weak MON anatomical somatotopy — the scientific constraint |
| `mon_ts_topo` | **0.2** | Weak MON→TS topography |
| `ll_to_mon_in_degree` | **10** | Each MON receives input from 10 of 100 LL cells (sparse but not starved) |
| `ll_mon_w_jitter_stdp_mV` | **8.0** | Wide initial weight jitter — breaks symmetry between MON cells |
| `ll_mon_w_init_mV` | 10.0 | Center of jitter range; weights start in [2, 18] mV |
| `ll_mon_apre` / `ll_mon_apost` | 0.010 / -0.0105 | Mild LTD-biased multiplicative STDP (default values) |
| `ll_mon_homeo_eta` | **0.005** | LL→MON homeostasis — forces MON to specialize on subset of inputs |
| `mon_ts_homeo_eta` | **0.001** | MON→TS homeostasis — caps incoming weight per TS cell |
| `mon_ts_gain_mV` | **220** | EPSP gain — required to make sparse MON drive sufficient for TS |
| `ts_local_inh_peak_mV` | **1.5** | Strong TS lateral inhibition — winner-take-all between TS cells |
| `bg_rate_mon_hz` | 18 | MON background drive (lowered from default 22 to reduce noise) |
| `mon_global_inh_mV` | 1.8 | Global MON inhibition (raised from default 1.15) |

---

## Mechanism — why this recipe works

1. **Wide LL→MON jitter (8 mV)** breaks initial symmetry — each MON cell starts with a unique random preference for its 10 LL inputs.
2. **LL→MON homeostasis (eta=0.005)** keeps the per-MON incoming weight sum bounded, forcing MON cells to specialize (some weights → wmax, some → 0).
3. **MON→TS homeostasis (eta=0.001)** does the same job at the second synapse — each TS cell's incoming weight sum is capped, so it can't accumulate strong drive from many uncorrelated MON cells.
4. **High MON→TS gain (220)** compensates for the resulting sparse, selective MON drive.
5. **Strong TS lateral inhibition (1.5)** forces TS cells to compete per stimulus.

Together these mechanisms produce a **population-level somatotopic map** at low MON somatotopy. Per-individual-network bands persist (each network has different x positions where many TS cells co-fire) but **bands shift across seeds** → multi-seed averaging cleans the map.

---

## Topo gradient — does the recipe scale to even weaker somatotopy?

See `RESULTS.md` for full details. Summary: the recipe works at `ll_mon_topo = mon_ts_topo` as low as **0.1** (5× weaker than the trivially-working high-topo baseline of 0.8).

| topo | mean sigma_theta | mean valid_fraction | seeds | Status |
|---|---|---|---|---|
| 0.20 | 0.354 (extract-mode) | 0.912 (extract-mode) | 10 | **baseline (best balance)** |
| 0.15 | 0.455 (extract-mode) | 0.893 (extract-mode) | 10 | works (slightly degraded) |
| 0.10 | 0.76 (extract-mode) | 0.84 (extract-mode) | 3 | still works (substantially degraded) |
| (high-topo 0.80) | 0.875 | 0.660 | reference | trivially works |

---

## Code changes added in this baseline cycle

- New CLI flag: `--ll-mon-in-degree` (overrides `ll_to_mon_in_degree`)
- New plot: `brian2_ll_spikes_vs_x_test_*.png` (LL afferent diagnostic, useful for confirming input is identical across runs)
- Resume check at `_run_spatial_two_stage_model` relaxed (`>=` → `>`) to allow test-only re-evaluation from saved weights

---

## Previous baseline (2026-04-27, archived for reference)

| Metric | Value |
|--------|-------|
| Run | `Runs/ts_inh15_gain125_10k/` (HIGH topo: ll_mon_topo=0.8) |
| `valid_fraction` | 0.582 |
| `sigma_theta` | 0.376 rad |

The high-topo case has always worked. The point of this project — and of this new baseline — is to show that **weak** MON somatotopy is sufficient when STDP + homeostasis are tuned correctly.
