# Simulations Index — LateralLine STDP

Quick reference for finding simulation runs.
Last updated: **2026-05-15**.

For the scientific narrative and figure captions see **`RESULTS.md`**.
For the current best recipe (commands you can copy-paste) see **`BASELINE.md`**.

---

## 1. Active result-producing runs (USE THESE)

All use the same baseline recipe (topo=0.20, MON=3200, 10000 trials, fixed D=0.8 cm),
varying only `--ll-mon-topo`/`--mon-ts-topo`, or `--n-mon`, or the test-phase noise.

### 1.1 Topo gradient — main result (`RESULTS.md` Fig 1)

10 seeds each, training at fixed D=0.8 cm.

| topo level | training run | sweep label | N seeds | mean σ_θ |
|------------|--------------|-------------|---------|---------|
| 0.10 | `Runs/llmon_topo010_seeds126_132/` (+ extract_topo010_seed_123)        | `topo010` (sweep) / `extract_topo010_*` (single-D) | 8 | 0.919 |
| 0.15 | `Runs/llmon_topo015_seeds127_132/` (+ Y2 seeds 123-126)                | `extract_topo015_*` | 10 | 0.455 |
| 0.20 | `Runs/llmon_topo020_seeds127_132/` (+ extract_topo020 seeds 123-126)   | `topo020` (sweep) / `extract_topo020_*` (single-D) | 10 | 0.353 |
| 0.40 | `Runs/llmon_topo040_seeds123_132/` | `topo040` (sweep) | 10 | 0.330 |
| 0.60 | `Runs/llmon_topo060_seeds123_132/` | (no sweep)         | 10 | 0.286 |
| 0.80 | `Runs/llmon_topo080_seeds123_132/` | `topo080` (sweep) | 10 | 0.278 |

Plot: `Picture/topo_gradient_summary.png` · Code: `plots/topo_gradient_summary.py`.

### 1.2 Chapter-5 distance sweeps — different topo (Fig 5.1a)

Uses extract-mode evaluation on the topo-gradient checkpoints above.
Test paths from `Runs/distswp_topoXXX_seedNNN_dDDD/` (XXX = topo×100, DDD = D×100 cm).

| label | source training | seeds | distances | σ_noise |
|-------|-----------------|-------|-----------|---------|
| `topo010` | `llmon_topo010_seeds126_132` | 126, 127, 128 | 0.2–3.0 cm | 0 |
| `topo020` | `llmon_topo020_seeds127_132` | 127, 128, 129 | 0.2–3.0 cm | 0 |
| `topo040` | `llmon_topo040_seeds123_132` | 123, 124, 125 | 0.2–3.0 cm | 0 |
| `topo080` | `llmon_topo080_seeds123_132` | 123, 124, 125 | 0.2–3.0 cm | 0 |

### 1.3 Noise sweep — topo=0.20 (Fig 5.4, 5.5)

Same source training as `topo020`. Adds Iris-Hydi-style Gaussian noise to LL rates during test.

| label | source training | seeds | σ_noise |
|-------|-----------------|-------|---------|
| `topo020`   | `llmon_topo020_seeds127_132` | 127, 128, 129 | 0 Hz (baseline) |
| `topo020n2` | `llmon_topo020_seeds127_132` | 127, 128, 129 | 2 Hz |
| `topo020n5` | `llmon_topo020_seeds127_132` | 127, 128, 129 | 5 Hz |

Noise param: `--test-ll-noise-hz` (added in `ll_stdp_brian2.py` 2026-05-13).

### 1.4 MON-size sweep — Phase 2 (Fig 5.1b, 5.3, 5.1b')

Topo=0.20 fixed, varying MON neuron count and gain. Three seeds each (123, 124, 125).

| Run | N_MON | gain (mV) | trained at | mean σ_θ at D=0.8 |
|-----|-------|-----------|------------|---------------------|
| `llmon_nmon400_seeds123_125`         | 400  | 220 (UNSCALED) | 2026-05-14 morning | π (mostly silent) |
| `llmon_nmon800_seeds123_125`         | 800  | 220 (UNSCALED) | 2026-05-14 morning | 1.29 (broken) |
| `llmon_nmon1600_seeds123_125`        | 1600 | 220 (UNSCALED) | 2026-05-14 morning | 0.67 |
| `llmon_nmon400_scaled_seeds123_125`  | 400  | 1760 (SCALED)  | 2026-05-14 10:28-15:19 | 1.77 |
| `llmon_nmon800_scaled_seeds123_125`  | 800  | 880 (SCALED)   | 2026-05-14 15:19-20:08 | 1.04 |
| `llmon_nmon1600_scaled_seeds123_125` | 1600 | 440 (SCALED)   | 2026-05-14 20:08-01:46 | 0.52 |
| `llmon_topo020_seeds127_132`         | 3200 | 220 (baseline) | (prior)            | 0.30 |

Sweep labels: `nmon{400,800,1600}` (unscaled) and `nmon{400,800,1600}_scaled` (scaled).

Gain-scaling rule: `gain = 220 × 3200 / N_MON` (keeps mean drive to TS roughly constant).

### 1.5 Multi-distance pilot — generalization test

Trained at d ∈ [0.6, 1.2] cm (uniform per trial) instead of fixed 0.8 cm.

| Run | seeds | distances trained | sweep label |
|-----|-------|-------------------|-------------|
| `multidist_pilot_3seed/` | 123, 124, 125 | uniform [0.6, 1.2] cm | `multidist` (run 2026-05-15) |
| `multidist_pilot_smoke_2k/` | 123 | smoke test, 2000 trials | — |

---

## 2. Figures (Picture/)

| File | Source data | Plot code |
|------|-------------|-----------|
| `topo_gradient_summary.png` | §1.1 | `plots/topo_gradient_summary.py` |
| `ch5_fig51a_sigma_vs_dist_topo.png` | §1.2 | `plots/chapter5_figures.py::fig51a` |
| `ch5_fig51b_sigma_vs_dist_nmon.png` | §1.4 scaled | `plots/chapter5_figures.py::fig51b` |
| `ch5_fig51b_comparison_unscaled_vs_scaled.png` | §1.4 both | `plots/chapter5_figures.py::fig51b_comparison` |
| `ch5_fig53_sigma_vs_nmon.png` | §1.4 both at D=0.8 | `plots/chapter5_figures.py::fig53` |
| `ch5_fig54_sharpening_vs_dist.png` | §1.3 | `plots/chapter5_figures.py::fig54` |
| `ch5_fig55_variability_vs_dist.png` | §1.3 | `plots/chapter5_figures.py::fig55` |

Regenerate all chapter-5 figures (≈ 5 s):
```bash
/Users/juliegoulet/anaconda3/bin/python plots/chapter5_figures.py
```

---

## 3. Orchestration scripts (project root)

| Script | Purpose |
|--------|---------|
| `run_multi_seed_safe.sh` | OOM-safe training: one Python process per seed |
| `run_distance_sweep_extract.sh` | Distance sweep on a trained run (extract-mode test) |
| `run_ch5_sweeps_master.sh` | Master for chapter-5 sweeps (topo + noise sweeps) |
| `run_ch5_phase2_monsize.sh` | Phase-2 MON-size trainings + sweeps |
| `run_ch5_phase2_full_recovery.sh` | Full recovery: unscaled sweeps + scaled retraining + scaled sweeps |
| `run_extract_evaluation.sh` | Extract-mode batch (topo=0.15, 0.20) |
| `run_extract_topo010.sh` | Extract-mode batch (topo=0.10) |
| `run_ll_long.sh` | Generic single-run launcher (older runs) |
| `make_extract_checkpoint.py` | Create minimal checkpoint from `latest_seed_NNN.npz` for extract-mode test |

All scripts self-background via `nohup caffeinate` when invoked without `--_bg`.
Logs go to `Logs/<scriptname>.log`.

---

## 4. Naming conventions

### Training runs
- `llmon_topo{XXX}_seeds{NNN_MMM}` — main topo-gradient runs.
- `llmon_nmon{N}_seeds{NNN_MMM}` — MON-size unscaled.
- `llmon_nmon{N}_scaled_seeds{NNN_MMM}` — MON-size scaled gain.
- `multidist_*` — multi-distance training.

### Distance-sweep runs
- `distswp_{label}_seed{NNN}_d{DDD}` — one (seed, distance) test, `DDD = D×100` cm.
- `distswp_{label}_seed{NNN}_extract` — temp dir used for extract checkpoint.

### Extract evaluation runs
- `extract_topo{XXX}_seed_{NNN}` — extract-mode test at training distance only.

### Logs
- `Logs/<run-name>.log` — Python stdout/stderr.
- `Logs/<script>_safe.log` — wrapper script log.
- `Logs/<run-name>_RESUME.sh` — auto-generated resume script.

---

## 5. Useful one-liners

### Aggregate sweep results for a label
```bash
ls Runs/distswp_<label>_seed*_d*/artifacts/seed_*_results.json | wc -l
```

### Check what's currently running
```bash
ps aux | grep ll_stdp_brian2 | grep -v grep
```

### Tail master log
```bash
tail -20 Logs/ch5_phase2_recovery_safe.log
```

### Watch a long-running training
```bash
tail -f Logs/llmon_<runname>_safe.log
```

---

## 6. Historical / exploratory runs (do not use for current results)

These were part of recipe development from March–April 2026 and are kept for traceability.
**They use different parameter sets** from the current baseline and are not directly comparable.

- `20260326_*` … `20260424_*` — dated exploration runs.
- `baseline_repro` — pre-STDP baseline (2026-04-19).
- `monton02_llmon_stdp_*` — early LL→MON STDP tuning.
- `llmontopo_*` — topography reduction series.
- `llmon_indeg*`, `llmon_jit*`, `llmon_homeo*` — recipe component sweeps.
- `llmon_X*`, `llmon_Y*`, `llmon_U*`, `llmon_T*` — multi-seed pilots.
- `ts_inh15_gain125_10k` — old HIGH-topo "baseline" (topo=0.80, gain=125) — superseded.
- `topo010_10k`, `topo015_10k`, `topo_*` — distance/topography pilots.
- `nohomeo_*`, `stdp_influence_*`, `stdp_topo02_*` — STDP ablation studies.

For why each step was taken, see project memory at:
`~/.claude/projects/-Users-juliegoulet-Documents-LateralLine2026-Code-LateralLine/memory/`

<!-- AUTOGEN:START -->

## 7. Auto-generated run inventory

_Regenerated by `tools/update_simulations_index.py` on **2026-05-15 08:54:53**._
_Total run dirs: **641**  ·  Training: 23  ·  Sweep runs: 392  ·  Extract: 28  ·  Historical: 63_

### Training runs (latest 25 by mtime)

| run-name | seeds done | mean σ_θ | mean valid | mtime |
|----------|-----------|----------|-----------|-------|
| `llmon_nmon1600_scaled_seeds123_125` | 3 (123,124,125) | 0.601 | 0.806 | 2026-05-15 01:05 |
| `llmon_nmon800_scaled_seeds123_125` | 3 (123,124,125) | 0.757 | 0.836 | 2026-05-14 20:08 |
| `llmon_nmon400_scaled_seeds123_125` | 3 (123,124,125) | 0.820 | 0.890 | 2026-05-14 15:20 |
| `llmon_nmon1600_seeds123_125` | 3 (123,124,125) | 1.130 | 0.391 | 2026-05-14 05:00 |
| `llmon_nmon800_seeds123_125` | 3 (123,124,125) | 0.740 | 0.092 | 2026-05-14 00:31 |
| `llmon_nmon400_seeds123_125` | 3 (123,124,125) | 3.142 | 0.000 | 2026-05-13 20:23 |
| `llmon_topo080_seeds123_132` | 10 (123,124,125,126,127,128,129,130,131,132) | 0.278 | 0.983 | 2026-05-13 10:26 |
| `llmon_topo060_seeds123_132` | 10 (123,124,125,126,127,128,129,130,131,132) | 0.286 | 0.971 | 2026-05-12 15:12 |
| `llmon_topo040_seeds123_132` | 10 (123,124,125,126,127,128,129,130,131,132) | 0.330 | 0.906 | 2026-05-11 05:39 |
| `llmon_topo010_seeds124_125` | 2 (124,125) | 0.971 | 0.748 | 2026-05-10 12:52 |
| `multidist_pilot_3seed` | 3 (123,124,125) | 0.601 | 0.813 | 2026-05-09 16:55 |
| `multidist_pilot_smoke_2k` | 1 (123) | 0.562 | 0.807 | 2026-05-09 11:14 |
| `llmon_topo010_seeds126_132` | 7 (126,127,128,129,130,131,132) | 1.004 | 0.733 | 2026-05-08 22:51 |
| `llmon_topo015_seeds127_132` | 6 (127,128,129,130,131,132) | 0.597 | 0.809 | 2026-05-07 21:07 |
| `llmon_topo020_seeds127_132` | 6 (127,128,129,130,131,132) | 0.512 | 0.808 | 2026-05-06 22:57 |
| `llmon_topo020_seeds127_128` | 0 (—) | — | — | 2026-05-06 10:51 |
| `llmon_extract_seed125_topo01` | 1 (125) | — | — | 2026-05-04 15:22 |
| `llmon_extract_seed124_topo01` | 1 (124) | — | — | 2026-05-04 08:59 |
| `llmon_extract_seed123_topo01` | 1 (123) | — | — | 2026-05-04 08:59 |
| `llmon_extract_seed124_topo015` | 1 (124) | — | — | 2026-05-03 20:54 |
| `llmon_extract_seed123_topo015` | 1 (123) | — | — | 2026-05-03 20:41 |
| `llmon_extract_seed126_topo015` | 1 (126) | — | — | 2026-05-03 20:36 |
| `llmon_extract_seed125_topo015` | 1 (125) | — | — | 2026-05-03 20:33 |

### Distance sweeps (by label)

| label | runs done | distinct seeds | distinct distances | latest mtime |
|-------|-----------|----------------|--------------------|--------------|
| `multidist` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-15 08:42 |
| `nmon1600` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-14 10:28 |
| `nmon1600_scaled` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-15 01:46 |
| `nmon400` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-14 09:56 |
| `nmon400_scaled` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-15 01:18 |
| `nmon800` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-14 10:12 |
| `nmon800_scaled` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-15 01:32 |
| `noisecheck` | 1 | 1 (127) | 1 (0.80) | 2026-05-13 13:38 |
| `sanity` | 1 | 1 (127) | 1 (0.80) | 2026-05-13 11:27 |
| `topo010` | 30 | 3 (126,127,128) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-13 13:56 |
| `topo020` | 30 | 3 (127,128,129) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-13 11:45 |
| `topo020n2` | 30 | 3 (127,128,129) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-13 14:44 |
| `topo020n5` | 30 | 3 (127,128,129) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-13 15:00 |
| `topo040` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-13 14:11 |
| `topo080` | 30 | 3 (123,124,125) | 10 (0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00) | 2026-05-13 14:28 |

### Extract single-distance runs: 28 dirs (see `extract_topoXXX_seed_NNN/`).

### Historical / exploratory: 159 dirs (see §6 of this index for context).

<!-- AUTOGEN:END -->
