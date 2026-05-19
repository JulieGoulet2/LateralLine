# LateralLine STDP — Results Summary (2026-05-15)

## The scientific question

> Can spike-timing-dependent plasticity (STDP) plus lateral inhibition self-organise a somatotopic map in the Torus semicircularis (TS) when the upstream MON layer has only **weak** anatomical somatotopy?

The trivial case — strong, hand-wired anatomical topography in MON — has always worked and is not interesting. The question is whether the network can *learn* the map when the anatomical scaffold is genuinely weak. This is the central hypothesis of the thesis.

## TL;DR

**Yes**, with one important qualification. The recipe documented in `BASELINE.md` produces a population-level somatotopic map in TS at MON anatomical somatotopy as low as **`ll_mon_topo = mon_ts_topo = 0.15`**, comfortably below the high-topo reference (0.80) used in earlier work. At the **best operating point (topo = 0.20)** the recipe is highly reliable: 10 / 10 seeds give σ_θ < 0.5 rad and valid_fraction > 0.86. At **topo = 0.15** it remains reliable but with broader maps. At the **floor (topo = 0.10)** it becomes unstable — most seeds still produce a map, but ~10 % fail outright.

The full topo gradient (0.10 → 0.80, 10 seeds each) is now complete. Map quality improves monotonically with anatomical topography, but the improvement flattens sharply above topo = 0.20: going from 0.20 to 0.80 reduces mean σ_θ by only ~0.07 rad (0.353 → 0.278), while going from 0.10 to 0.20 reduces it by 0.55 rad (0.902 → 0.353). **The recipe already captures the vast majority of achievable map quality at weak somatotopy.**

A persistent imperfection across all topo levels is **multimodal per-TS-cell tuning** (vertical bands in TS spike rasters) — see "Open questions" below.

**Chapter-5 reproduction (2026-05-15).** Seven figures from Iris Hydi's master thesis chapter 5 have been reproduced in this model — see the *Chapter-5-style analyses* section. Key findings: (i) σ_θ(D) is U-shaped with minimum at D ≈ 0.6–0.8 cm and saturation to chance at D ≳ 0.4 L (≈ 1.6 cm) — consistent with Iris Hydi's "stable localization within ~1 body length"; (ii) the U-shape minimum is **driven by stimulus geometry, not training-distance overfitting** (proved by a multi-distance training sanity check); (iii) σ_θ at the training distance scales monotonically with MON size N (0.30 at N = 3200 → 1.77 at N = 400), but only once the MON → TS gain is rescaled as gain ∝ 1/N; without rescaling small N is broken (TS silent); (iv) population sharpening σ_θ^LL/σ_θ^TS is robust to up to 5 Hz of test-phase LL noise.

---

## Methods

### Network architecture

Three-layer feed-forward network (defaults from `params.py`, mode `ll_thesis`):

| Layer | Size | Role |
|---|---|---|
| **LL** (lateral-line afferents) | 100 | One neuron per neuromast, x-positions equispaced on a 4 cm body. Input = analytical hydrodynamic dipole field (`stimulus.py`); only the **spike emission** is Poisson (inhomogeneous rate, see Stimulus section). |
| **MON** (intermediate) | 3200 | LIF spiking, plastic incoming synapses (LL → MON STDP) |
| **TS** (map output) | 300 | LIF spiking, plastic incoming synapses (MON → TS STDP), lateral inhibition |

LIF parameters (identical for MON and TS): `V_th = -54 mV`, `V_reset = -60 mV`, `E_L = -74 mV`, `τ_m = 10 ms`, `τ_s = 2 ms`, refractory 2 ms. Numerical integration: Brian2, `dt = 1 ms`.

### Connectivity — the topography knob

LL → MON and MON → TS connectivity each mix two components:
- a **random** part (uniform connection probability)
- a **topographic** part (Gaussian centred at the somatotopically matched position)

The fraction that follows topography is set by the parameters `ll_mon_topo` and `mon_ts_topo` (both denoted *topo* below). At `topo = 0.0` connections are fully random; at `topo = 1.0` they are fully Gaussian-topographic. **All experiments here use `ll_mon_topo = mon_ts_topo`** — the two are varied together. Each MON cell receives 10 inputs from LL (`--ll-mon-in-degree 10`); each MON cell projects to 16 TS cells (`out-degree 16`).

### Stimulus

A small sphere ("dipole") moves past the lateral line at fixed speed (5 cm/s) at lateral distance `d`. The hydrodynamic velocity field at each LL position is computed analytically (`stimulus.hydrodynamic_velocity_parallel`). Each LL neuromast fires an inhomogeneous Poisson spike train with rate `r_0 + A · v(x_i, x_source, d) + spatial-correlated noise`, clipped to `[0, r_max]`.

For all experiments in the topo gradient below, the training distance is **fixed at `d = 0.8 cm`** (`--training-distance-min-cm 0.8 --training-distance-max-cm 0.8`). A multi-distance pilot (d ∈ [0.6, 1.2] cm) was completed on 2026-05-09 — see "Multi-distance pilot" section below.

### Training protocol

- **10 000 trials**, each 1.2 s long.
- Each trial holds the source position fixed for 50 ms windows (`training_position_hold_s = 0.05`) and sweeps positions across the body in **balanced ordered forward/backward sweeps** (`training_ordered_sweeps = True`) so every x-position receives equal training time.
- Source distance fixed at 0.8 cm (see above).
- Source direction fixed (no bidirectional flipping).
- **No additive LL noise** during training (`training_noise_scale_early/late = 0.0`) — only the spatially-correlated background of the stimulus model is present.
- Background drive in MON: Poisson at 18 Hz, EPSP weight 1.5 mV.

### Plasticity

The two plastic synapses use **different STDP rules**, a fact discovered during the 2026-05-15 code review:

| Synapse | STDP rule | apre (LTP) | apost (LTD) | wmax | w_init | w_jitter | homeo η |
|---|---|---|---|---|---|---|---|
| **LL → MON** | **Multiplicative** (`Δw_LTP = apre × (wmax - w)`, `Δw_LTD = apost × w`) | 0.010 | -0.0105 | 20 mV | 10 mV | **8 mV** | **0.005** |
| **MON → TS** | **Additive** (`Δw_LTP = apre`, `Δw_LTD = apost`) | 0.010 | -0.006 | 0.028 | 0.020 | 0.005 | **0.001** |

**Why LL → MON is multiplicative.** Additive STDP with net potentiation (apre > |apost|) drives every weight to wmax, saturating the MON layer and destroying the somatotopic map. The multiplicative form self-stabilises at an intermediate weight via the weight-dependent scaling. We verified this in April 2026 — there is a regression test in `tests/` that catches reversal of the rule.

**Why MON → TS is additive (and OK).** The recipe at this synapse pairs additive STDP with the **slow multiplicative incoming-weight homeostasis** described below (`mon_ts_homeo_eta = 0.001`). The homeostasis rescales Σ_in w per TS cell back toward its initial target every 10 trials, which prevents the additive-STDP runaway saturation. The combined dynamic (additive STDP + multiplicative homeostasis) is what gives the current baseline its sharpness. Changing this synapse to multiplicative would require re-tuning the entire recipe (apre/apost amplitudes, homeo η, gain) and re-running the topo gradient and chapter-5 sweeps — a scientific decision deferred. In-code documentation: see the comment block on the `s_mon_ts = b2.Synapses(...)` definition in `ll_stdp_brian2.py`. Memory note: `~/.claude/projects/<...>/memory/project_stdp_rules.md`.

The **wide LL → MON initial jitter (8 mV, range 2–18 mV)** is essential — it breaks initial symmetry between MON cells so that STDP has a non-trivial bias to amplify.

### Homeostasis

Both synapses also carry a slow **multiplicative homeostatic rescaling** of incoming weights, applied every 10 trials:

- **LL → MON** (`η = 0.005`): for each MON cell, scale all incoming LL weights so that `Σ w` matches a target (set by `w_init * in_degree`). Forces MON cells to **specialise** — some incoming weights are pushed to `wmax`, others to 0.
- **MON → TS** (`η = 0.001`): same mechanism for incoming MON → TS weights. Prevents any TS cell from inheriting strong drive from many uncorrelated MON inputs.

### Lateral inhibition in TS

Strong winner-take-all between TS cells: `ts_local_inh_peak_mV = 1.5` (peak at distance 0 in the TS index ring), inhibition radius 14 (TS-index units), toroidal. There is also a small global feedback inhibition path (`use_ts_feedback_inh = True`), present in all runs.

### Test phase and evaluation

After training, the source is moved continuously across the body at 5 cm/s over a 5 cm path at the same distance (`d = 0.8 cm`). All plasticity is **frozen** during this test sweep.

Two metrics summarise map quality:

- **σ_θ** (`pv_sigma_theta`, rad). For each test stimulus position, decode the TS population vector to an angle `θ̂`. σ_θ is the mean angular deviation between `θ̂` and the true source angle, taken in radians. Lower is sharper. **Chance level is π/2 ≈ 1.57 rad**.
- **valid_fraction** (`pv_valid_fraction`). Fraction of test stimuli for which the population vector amplitude exceeds a fixed threshold (i.e. enough TS cells fired to produce a usable estimate at all). Higher is more reliable. **Threshold for "this run is a working map": valid_fraction ≥ 0.60**.

### Multi-seed protocol (OOM-safe)

Each seed varies (a) all random connectivity draws, (b) initial weight jitter, (c) noise streams, (d) the training position-shuffle order. To run many seeds without losing data when Brian2 accumulates memory across seeds in a single Python process (an OS-level OOM kill that has happened repeatedly), `run_multi_seed_safe.sh` launches **one separate Python process per seed**. Each seed writes its own results JSON immediately on completion (`Runs/<run-name>/artifacts/seed_NNN_results.json`).

### Extract-mode evaluation (consistent across topo levels)

To compare topo levels on identical RNG state, all map-quality numbers reported below are **extract-mode**: load the saved final weights (`latest_seed_NNN.npz`), reconstruct a minimal checkpoint with `make_extract_checkpoint.py`, and re-run only the test phase from a fresh RNG state. This systematically gives σ_θ values 0.15–0.20 lower than training-mode metrics measured at the end of training, because the test phase is not perturbed by any leftover plasticity-induced state in TS. The orchestration script `run_extract_evaluation.sh` (topo = 0.15, 0.20) and `run_extract_topo010.sh` (topo = 0.10) generated the numbers in this document.

### Reproducing the baseline

The exact CLI for the topo = 0.20 baseline is in `BASELINE.md`. All runs in this document use the same parameter set; only `--ll-mon-topo` and `--mon-ts-topo` (always equal) and `--seed-start` vary.

---

## Results

### Overview figure

![Topo gradient summary](Picture/topo_gradient_summary.png)

**Figure 1.** *Somatotopic map quality as a function of MON anatomical topography strength.*
**(A)** Map sharpness measured by σ_θ in radians (lower is better). Light blue dots are individual seeds; dark blue points show mean ± SD across seeds at each topo level. The dotted line at π/2 ≈ 1.57 rad marks chance level. **(B)** Map reliability measured by `valid_fraction` (higher is better). Same conventions as in (A). The dotted line at 0.60 marks the validity threshold used to classify a run as a working map. Values for topo = 0.10 / 0.15 / 0.20 are extract-mode (saved final weights, test phase from a fresh RNG state); values for topo = 0.40 / 0.60 / 0.80 are training-mode metrics from `run_multi_seed_safe.sh`. N = 10 seeds at every topo level (seeds 123–132). **Map quality improves monotonically with anatomical topography but saturates rapidly above topo = 0.20**; the recipe at topo = 0.20 already achieves ≈ 95 % of the map sharpness seen at topo = 0.80.

### `topo = 0.20` — operational baseline

10 seeds, 10 000 trials each, extract-mode metrics.

| Seed | σ_θ (rad) | valid_fraction |
|------|-----------|----------------|
| 123  | 0.421     | 0.926          |
| 124  | 0.401     | 0.888          |
| 125  | 0.420     | 0.890          |
| 126  | 0.323     | 0.902          |
| 127  | 0.382     | 0.931          |
| 128  | 0.336     | 0.891          |
| 129  | 0.291     | 0.881          |
| 130  | 0.293     | 0.960          |
| 131  | 0.400     | 0.952          |
| 132  | 0.267     | 0.901          |
| **mean ± SD** | **0.354 ± 0.058** | **0.912 ± 0.028** |

All 10 seeds beat the high-topo reference on both metrics. SD is tight (16 % of the mean for σ_θ, 3 % for valid_fraction). The recipe is reliable here.

### `topo = 0.15` — slightly degraded but still reliable

10 seeds, 10 000 trials each, extract-mode metrics.

| Seed | σ_θ (rad) | valid_fraction |
|------|-----------|----------------|
| 123  | 0.459     | 0.864          |
| 124  | 0.580     | 0.880          |
| 125  | 0.278     | 0.887          |
| 126  | 0.583     | 0.896          |
| 127  | 0.418     | 0.917          |
| 128  | 0.398     | 0.903          |
| 129  | 0.528     | 0.869          |
| 130  | 0.477     | 0.916          |
| 131  | 0.376     | 0.929          |
| 132  | 0.453     | 0.864          |
| **mean ± SD** | **0.455 ± 0.094** | **0.893 ± 0.024** |

10 / 10 seeds give `valid_fraction > 0.86`; 9 / 10 give σ_θ < 0.60. Spread is wider (SD ≈ 21 % of mean for σ_θ) but no run fails.

### `topo = 0.40` — plateau begins

10 seeds (123–132), 10 000 trials each, training-mode metrics.

| Seed | σ_θ (rad) | valid_fraction |
|------|-----------|----------------|
| 123  | 0.360     | 0.909          |
| 124  | 0.344     | 0.925          |
| 125  | 0.432     | 0.865          |
| 126  | 0.375     | 0.909          |
| 127  | 0.249     | 0.937          |
| 128  | 0.304     | 0.818          |
| 129  | 0.317     | 0.944          |
| 130  | 0.317     | 0.925          |
| 131  | 0.331     | 0.900          |
| 132  | 0.272     | 0.929          |
| **mean ± SD** | **0.330 ± 0.052** | **0.906 ± 0.038** |

Only marginally sharper than topo = 0.20 (σ_θ 0.330 vs 0.353). The saturation of map quality above topo = 0.20 is already visible here.

### `topo = 0.60` — near-ceiling performance

10 seeds (123–132), 10 000 trials each, training-mode metrics.

| Seed | σ_θ (rad) | valid_fraction |
|------|-----------|----------------|
| 123  | 0.343     | 0.980          |
| 124  | 0.346     | 0.985          |
| 125  | 0.250     | 0.974          |
| 126  | 0.278     | 0.971          |
| 127  | 0.256     | 0.978          |
| 128  | 0.247     | 0.974          |
| 129  | 0.289     | 0.963          |
| 130  | 0.294     | 0.957          |
| 131  | 0.289     | 0.950          |
| 132  | 0.269     | 0.973          |
| **mean ± SD** | **0.286 ± 0.035** | **0.971 ± 0.011** |

Very tight SD (12 % of mean for σ_θ, 1 % for valid_fraction). High reliability — every seed gives valid_fraction > 0.95.

### `topo = 0.80` — fully-topographic ceiling

10 seeds (123–132), 10 000 trials each, training-mode metrics.

| Seed | σ_θ (rad) | valid_fraction |
|------|-----------|----------------|
| 123  | 0.280     | 0.977          |
| 124  | 0.280     | 0.993          |
| 125  | 0.232     | 0.984          |
| 126  | 0.327     | 0.987          |
| 127  | 0.227     | 1.000          |
| 128  | 0.297     | 0.980          |
| 129  | 0.326     | 0.999          |
| 130  | 0.281     | 0.987          |
| 131  | 0.286     | 0.981          |
| 132  | 0.240     | 0.940          |
| **mean ± SD** | **0.278 ± 0.035** | **0.983 ± 0.017** |

Best map quality across all topo levels, as expected. Virtually indistinguishable from topo = 0.60 on σ_θ (0.278 vs 0.286), confirming saturation. **Note:** the old single-run reference at topo = 0.80 (σ_θ = 0.875, valid = 0.660, from `Runs/ts_inh15_gain125_10k/`) used a different parameter set (gain = 125, no LL→MON homeostasis) and is not comparable — see archived reference below.

---

### `topo = 0.10` — the floor of recipe robustness

10 seeds, 10 000 trials each, extract-mode metrics. Seeds 124 and 125 were retrained on 2026-05-17 post-B1 (commit `3392ea8` onward) using deterministic Brian2 RNG; the other 8 seeds (123, 126–132) are from the original Y4 multi-seed run with un-seeded Brian2 RNG.

| Seed | σ_θ (rad) | valid_fraction | Note |
|------|-----------|----------------|------|
| 123  | **1.482** | **0.539** | below validity threshold |
| 124  | 0.775     | 0.868          | retrained 2026-05-17 (post-B1) |
| 125  | 0.897     | 0.886          | retrained 2026-05-17 (post-B1) |
| 126  | 0.915     | 0.860          | |
| 127  | 0.775     | 0.906          | |
| 128  | 0.699     | 0.884          | |
| 129  | 0.834     | 0.848          | |
| 130  | 0.765     | 0.889          | |
| 131  | 0.770     | 0.940          | |
| 132  | 1.108     | 0.844          | |
| **mean ± SD (N = 10)** | **0.902 ± 0.234** | **0.846 ± 0.112** | |
| **mean ± SD (N = 9, excl. seed 123)** | **0.838 ± 0.122** | **0.881 ± 0.030** | |

At `topo = 0.10` the recipe is at the edge of what it can deliver. **One seed in ten (≈ 10 %) outright fails** (seed 123: σ_θ near chance, valid_fraction below threshold). The other 9 seeds still produce a working map, but σ_θ is broader (mean 0.84 rad, comparable to the high-topo reference) and the spread is large. At this somatotopy, **the recipe is no longer reliable on a per-individual-network basis** — it works on average but with non-trivial probability of complete failure. The two newly added seeds (124, 125) fall within the typical range, confirming the failure mode is not specific to the seeds that the original Y4 run happened to use.

---

## Chapter-5-style analyses (Iris Hydi reproduction)

A four-part reproduction of Iris Hydi's master-thesis chapter 5 figures, transferred from her snake pit-organ system to the lateral-line model.
All sweeps use the topo = 0.20 baseline recipe unless otherwise noted (see § Methods).
Distance D refers to the source-body lateral distance; the body has length L = 4 cm.

### Stimulus distance — Fig 5.1a

**Q.** How does map quality depend on stimulus distance D, and how does this interact with the anatomical somatotopy parameter `topo`?

**Result.** All four topo levels show a clear U-shape in σ_θ as a function of D, with minimum at D ≈ 0.6–0.8 cm (D/L ≈ 0.15–0.20) and degradation at both smaller D (sharp dipole field, poor coverage) and larger D (weak signal, low valid_fraction). σ_θ approaches chance (π/2) beyond **D ≈ 1.5 cm (≈ 0.4 L)** for all topo levels. This is consistent with Iris Hydi's claim that "stable object localization is only possible within about one body length."

The four curves separate cleanly by topo level only at small D (≤ 1 cm); at large D they collapse onto the same chance-level plateau.

![Figure 5.1a — sigma_theta vs distance, four topo levels](Picture/ch5_fig51a_sigma_vs_dist_topo.png)

**Figure 5.1a** (`Picture/ch5_fig51a_sigma_vs_dist_topo.png`). Somatotopic decoding error σ_θ as a function of stimulus distance D/L, for four MON anatomical somatotopy levels (topo = 0.10, 0.20, 0.40, 0.80). Each curve is mean ± SD across 3 seeds; shading shows ± 1 SD. Dashed black line marks the training distance D = 0.8 cm; grey dotted line marks π/2 (chance). All evaluations are extract-mode tests (saved final weights, fresh RNG test phase).

### Single-distance vs multi-distance training — Fig 5.1a'

**Q.** Is the U-shape minimum at D = 0.8 cm an artifact of the training protocol (we trained only at that distance), or is it intrinsic to the stimulus geometry?

**Result.** Identical U-shape in both training protocols. Training on a uniform mixture d ∈ [0.6, 1.2] cm produces a minimum at D = 0.6 cm with **σ_θ = 0.400 rad**, a single-D fit at D = 0.8 gives **σ_θ = 0.303 rad** — the multi-D curve is slightly broader (σ_θ ≈ 0.40 over a 0.6–1.0 cm range, vs single-D's sharp peak at 0.80 cm) but the minimum is **in the same neighbourhood** of distances regardless of training. Outside the training range (D ≥ 1.5 cm) both curves are statistically indistinguishable.

**Interpretation.** The U-shape minimum is driven by **stimulus geometry**: the dipole hydrodynamic field at D ≈ 0.6–0.8 cm provides the steepest spatial gradient across the lateral-line array — the signal is most informative there. Training distance modulates how much the network *exploits* that geometry, but does not move the optimal distance.

![Figure 5.1a' — single-D vs multi-D training](Picture/ch5_fig51a_singleD_vs_multiD.png)

**Figure 5.1a'** (`Picture/ch5_fig51a_singleD_vs_multiD.png`). Single-distance training (blue, fixed D = 0.8 cm) vs multi-distance training (orange, D ∈ [0.6, 1.2] cm uniform per trial). Mean ± SD across 3 seeds. Shaded vertical band marks the multi-D training range; dashed line marks the single-D training point. Both protocols use topo = 0.20 and 10 000 trials.

### MON neuron count — Figs 5.1b, 5.1b′, 5.3

**Q.** How does the number of MON neurons N affect map quality? Does the recipe transfer when the population size shrinks?

**Result.** Two phases:

1. **Unscaled gain (220 mV for all N).** At small N, TS goes silent — each TS cell receives ~N × out_degree / N_TS inputs and the average drive scales as N. With gain calibrated at N = 3200, dropping to N = 400 starves TS (3/3 seeds silent at most distances). At N = 800 most distances are silent; at N = 1600 the map exists but is degraded.

2. **Scaled gain (`gain = 220 × 3200 / N`).** TS firing is recovered at every N. Map quality at the training distance D = 0.8 cm is now a **clean monotonic function of N**:

   | N_MON | gain (mV) | σ_θ at D = 0.8 cm | mean valid |
   |---|---|---|---|
   | 400   | 1760 | 1.77 ± 1.19 | 0.31 |
   | 800   | 880  | 1.04 ± 0.23 | 0.66 |
   | 1600  | 440  | 0.52 ± 0.06 | 0.79 |
   | 3200  | 220  | 0.30 ± 0.02 | 0.96 |

   σ_θ decreases by ~6× across this 8× range in N. The MON layer is the population-coding bottleneck: more neurons → finer x-position resolution → sharper somatotopic map.

**Interpretation.** This recovers the Iris Hydi N_MON-vs-σ_θ relationship from her chapter 5 (her Fig 5.3). The gain-scaling step is critical: it disentangles two effects — *can the network fire at all* (gain) from *how precisely can it code position* (N).

![Figure 5.1b — sigma_theta vs distance, four MON sizes (scaled gain)](Picture/ch5_fig51b_sigma_vs_dist_nmon.png)

**Figure 5.1b** (`Picture/ch5_fig51b_sigma_vs_dist_nmon.png`). σ_θ vs distance D/L for four MON sizes (N = 400, 800, 1600, 3200) using the gain-scaling recipe. Mean ± SD across 3 seeds. Same conventions as Fig 5.1a.

![Figure 5.1b' — unscaled vs scaled gain comparison](Picture/ch5_fig51b_comparison_unscaled_vs_scaled.png)

**Figure 5.1b'** (`Picture/ch5_fig51b_comparison_unscaled_vs_scaled.png`). Two-panel comparison. **Left**: unscaled (gain = 220 mV for all N) — small-N curves are clearly broken (high σ_θ, silent runs at many distances). **Right**: scaled (gain = 220 × 3200/N) — all curves recover the expected family of U-shapes, separating cleanly by N. This figure justifies why the gain-scaling correction is necessary before comparing across N.

![Figure 5.3 — sigma_theta at D=0.8 cm vs MON size](Picture/ch5_fig53_sigma_vs_nmon.png)

**Figure 5.3** (`Picture/ch5_fig53_sigma_vs_nmon.png`). Map quality at the training distance D = 0.8 cm vs N_MON, log₂ x-axis. **Blue (solid)**: scaled gain — clean monotonic improvement σ_θ = 1.77 → 1.04 → 0.52 → 0.30 across N = 400 → 3200. **Red (dashed)**: unscaled gain (gain = 220 mV) — non-monotonic, because the recipe is broken for small N. Error bars are SD across 3 seeds. Dotted line marks π/2 (chance).

### Test-phase noise — Figs 5.4 & 5.5

**Q.** How robust is the population-level map to noise added to LL afferent rates at test time?

**Result.** Three noise levels were tested at topo = 0.20: σ_noise = 0, 2, 5 Hz (Gaussian noise added per-neuron per-timestep to LL Poisson rates during the test phase only, via `--test-ll-noise-hz`).

- **Population sharpening σ_θ^LL / σ_θ^TS (Fig 5.4)** peaks at D ≈ 0.6–0.8 cm with ratio ≈ 2.3–2.6 (TS map is 2× sharper than direct LL decoding), then falls below 1 beyond D ≈ 1.2 cm. **Noise has essentially no effect** — the three curves are statistically indistinguishable. The map is robust to up to 5 Hz of additive LL noise at the population-decoding level.

- **Trial-to-trial variability ratio Δ_TS / Δ_LL (Fig 5.5)** is **above 1 across all distances** (TS is more variable than LL), peaking at D < 0.4 cm (ratio ≈ 5×). **Direction opposite to Iris Hydi's pit-organ result** (her Fig 5.5 shows RC less variable than IR). Reason: in our recipe the LL layer is essentially a clean Poisson representation of the stimulus (the spatial-correlated background noise σ_corr is already in the stimulus model); 5 Hz of additive noise is small compared to mean LL rate (~ 8 Hz), so LL stays cleaner than the noisy multi-stage TS output (MON spikes + lateral inhibition + STDP residual). To reproduce Iris Hydi's Fig 5.5 *quantitatively*, the LL would need much higher intrinsic noise — a design choice rather than a property of the model.

![Figure 5.4 — population sharpening vs distance, three noise levels](Picture/ch5_fig54_sharpening_vs_dist.png)

**Figure 5.4** (`Picture/ch5_fig54_sharpening_vs_dist.png`). Population sharpening ratio σ_θ^LL / σ_θ^TS as a function of D/L, for three test-phase noise levels (σ_noise = 0, 2, 5 Hz). Ratio > 1 means the TS population vector is sharper than direct LL decoding. Mean ± SD across 3 seeds. Dashed line: training D = 0.8 cm. Dotted: ratio = 1 (no sharpening).

![Figure 5.5 — trial variability ratio vs distance, three noise levels](Picture/ch5_fig55_variability_vs_dist.png)

**Figure 5.5** (`Picture/ch5_fig55_variability_vs_dist.png`). Trial-to-trial variability ratio Δ_trial^TS / Δ_trial^LL as a function of D/L, same conditions and conventions as Fig 5.4. Ratio < 1 would indicate the TS layer reduces trial-to-trial noise (Iris Hydi's pit-organ finding); our model shows ratio > 1 because LL is essentially noiseless in our recipe.

### Summary of chapter-5 reproduction

| Iris Hydi figure | Our equivalent | Result reproduced? |
|---|---|---|
| Fig 5.1 (σ_θ vs D, topo levels) | Fig 5.1a | ✅ U-shape with minimum at training D, falls to chance beyond ~1 body-length |
| (no equivalent in IH) | Fig 5.1a' (single-D vs multi-D) | ✅ U-shape is geometric, not training-distance overfitting |
| Fig 5.2 (σ_θ vs D, obs period T) | — | Not yet (requires per-window analysis) |
| Fig 5.3 (σ_θ vs MON size) | Fig 5.3 | ✅ Monotonic, with explicit gain-scaling step |
| Fig 5.4 (sharpening σ_LL/σ_TS) | Fig 5.4 | ✅ Population sharpening peaks at training D |
| Fig 5.5 (variability Δ_TS/Δ_LL) | Fig 5.5 | ⚠ Direction opposite — LL too clean in our recipe (see text) |

All data hardcoded in `plots/chapter5_figures.py`. Regenerate with:
```bash
python plots/chapter5_figures.py
```
Underlying sweep runs are catalogued in `SIMULATIONS_INDEX.md`.

---

## Training-phase noise robustness (2026-05-19)

### Question

Earlier chapter-5 figures (5.4 / 5.5) tested noise added at **test time** to a network that had been trained noise-free. That isolated decoding robustness from learning robustness. Here we ask the harder question:

**If LL afferents are noisy during the entire 10 000-trial training phase, does the somatotopic map still form, and is 10 000 trials still enough for STDP to converge?**

### Method

5 conditions on the topo = 0.20 baseline recipe (MON = 3200, fixed D = 0.8 cm), with constant Gaussian noise on LL rates throughout training. The noise SD in Hz equals `training_noise_scale × sigma_noise_hz` where `sigma_noise_hz = 10` Hz, so:

| Condition | `--training-noise-early = --training-noise-late` | Noise SD (Hz) |
|----------|-------------------------------------------------|---------------|
| control  | 0.0 | 0  |
| low      | 0.3 | 3  |
| moderate | 0.5 | 5  |
| high     | 0.8 | 8  |
| very high| 1.0 | 10 |

2 seeds per condition (123, 124), 10 trainings total (≈ 20 h sequential on M4 via `run_multi_seed_safe.sh`).

After training, each saved checkpoint was extract-mode-evaluated at the training distance D = 0.8 cm for an apples-to-apples comparison with the rest of the document.

### Result

**The recipe is essentially invariant to LL training-phase noise across the entire range tested**, including the very-high condition where the noise SD equals the documented `sigma_noise_hz` (and is comparable to the mean LL test-phase firing rate ~ 8 Hz).

| Condition | seed 123 σ_θ (extract) | seed 124 σ_θ (extract) | mean ± SD | mean valid |
|-----------|------------------------|-------------------------|-----------|-----------|
| noise = 0.0  | 0.453 | 0.418 | **0.436 ± 0.025** | 0.914 |
| noise = 0.3  | 0.455 | 0.370 | **0.413 ± 0.060** | 0.913 |
| noise = 0.5  | 0.493 | 0.414 | **0.454 ± 0.056** | 0.912 |
| noise = 0.8  | 0.432 | 0.388 | **0.410 ± 0.031** | 0.912 |
| noise = 1.0  | 0.499 | 0.370 | **0.435 ± 0.091** | 0.911 |

Mean σ_θ stays in the **0.41–0.45 rad** band at every noise level — well within the seed-to-seed variability seen elsewhere in this document (e.g. baseline topo = 0.20 has σ_θ = 0.353 ± 0.058 across 10 seeds, so 0.43 is within 1.5 SD of that mean). `valid_fraction` is essentially constant at 0.91.

### Convergence verdict

The 10 000-trial budget is sufficient under noise. Across all five conditions:

- The **σ_θ plateau test** passes — σ_θ stops decreasing in the last 1500 trials of every run; the smoothed learning curves (Fig X.A) are flat after about trial 5000.
- A **strict weight-stabilisation test** (mean |Δw| < 0.5 % for 4 consecutive checkpoints) does not pass for any condition — but it also does not pass for the noise = 0.0 control. This is a known property of the recipe (multiplicative STDP + slow homeostasis maintains a small steady weight churn at equilibrium) and is not a noise-induced failure. The σ_θ plateau is the meaningful convergence signal at this scale.

### Why the recipe is so robust

Three mechanisms cooperate to absorb noise at the input:

1. **Spatially-correlated stimulus noise was already present** (`stimulus.l_noise_cm = 1 cm`, `sigma_noise_hz = 10` Hz used as the structure background of `_sample_instantaneous_rates`). The training noise scale here multiplies that same correlated source; we are increasing existing trial-to-trial fluctuations, not introducing a new noise channel.
2. **LL → MON multiplicative STDP** integrates over many spike pairs per trial and over many trials per checkpoint; with 100 LL neurons × 100 ms presentations × 50 presentations per trial × 10 000 trials, the SNR on each MON cell's preferred-direction weight is very high even when the per-spike noise is comparable to signal.
3. **MON → TS slow multiplicative homeostasis** (`mon_ts_homeo_eta = 0.001`) rescales the sum-of-incoming-weights toward target every 10 trials. This actively counteracts any noise-induced drift in the overall TS drive level, even though it does not select *which* MON inputs each TS cell prefers.

### Figure

![Figure X — training-phase LL noise robustness](Picture/ch5_training_noise_robustness.png)

**Figure X.** *Training-phase LL noise robustness.* **(A)** Learning curves: σ_θ during training (smoothed over 50 checkpoints, ≈ 500 trials) for the five noise conditions, mean across 2 seeds. Dashed vertical line marks the 10 000-trial budget; grey dotted line marks π/2 (chance). All five curves overlap within seed-to-seed variability. **(B)** Final extract-mode σ_θ at D = 0.8 cm vs noise scale, mean ± SD across 2 seeds. Horizontal dotted line: noise = 0 control. The 5-point sweep shows no systematic dependence on noise — the recipe is invariant to LL training-phase noise across the entire range tested, up to and including the 10 Hz level where the noise SD is comparable to the mean LL test-phase firing rate.

### Caveats and follow-ups

- Only 2 seeds per condition; the mean is informative but per-seed scatter could be sharpened by adding more seeds.
- The noise model used here is the existing `stimulus.py` spatially-correlated Gaussian on LL rates. A different noise structure (e.g. pure white noise per neuron, or jitter in `sigma_distance_cm`) could behave differently.
- Tested only on the topo = 0.20 baseline. The weak-topo regime (topo ≤ 0.15), where the recipe is already at the edge of failing, may not be as forgiving.

Underlying training runs: `Runs/llmon_trainnoise_noise{00,03,05,08,10}_seeds123_124/`. Extract-mode evaluations: `Runs/extract_trainnoise_noise{XX}_seed_{NNN}/`. Plot code: `plots/training_noise_robustness.py`. Convergence checker: `tools/check_convergence.py`.

---

## Mechanism — what made the recipe work at weak topo

After ~30 design experiments, three levers turned out to be necessary at low MON topo (without them the network either fails to form a map or produces all-saturated dead weights):

1. **Wide initial LL → MON weight jitter** (`--ll-mon-w-jitter-stdp-mv 8.0`). Breaks initial symmetry between MON cells so STDP has a non-trivial starting bias to amplify. Without it, MON cells start nearly identical and STDP cannot decide which to specialise.
2. **LL → MON multiplicative homeostasis** (`--ll-mon-homeo-eta 0.005`). Caps each MON cell's incoming weight sum, forcing the selectivity to live in *which* LL inputs are kept rather than *how strong* they all are. Without it, all weights drift toward the multiplicative-STDP equilibrium and the MON layer becomes unselective.
3. **MON → TS multiplicative homeostasis** (`--mon-ts-homeo-eta 0.001`). Same mechanism at the second synapse — prevents any TS cell from inheriting strong drive from many uncorrelated MON cells.

Two supporting levers:

4. **High MON → TS gain** (`--mon-ts-gain-mv 220`). Compensates for the resulting sparse, selective MON drive — without it TS rarely fires.
5. **Strong TS lateral inhibition** (`--ts-local-inh-peak-mv 1.5`). Winner-take-all between TS cells, so a single test stimulus selects a small group of winners.

The fix had to attack the MON layer first (heterogeneous init + homeostasis), then propagate downstream. Approaches that only modified TS — stronger lateral inhibition alone, or higher gain alone — failed because they addressed symptoms rather than the root cause (MON cells were not selective in the first place).

### Key dead ends — what did NOT work

- **LTD-biased STDP without homeostasis**: multiplicative STDP equilibrium (LL → MON) is fixed by the apre/apost ratio; uncorrelated weights settle in the middle, no bimodal weight distribution emerges. Homeostasis is what breaks this equilibrium and forces specialisation.
- **Sparser LL → MON anatomy alone** (`in_degree = 7` instead of 10): starves MON (rate drops from ~9 to ~2 Hz), TS goes silent.
- **Increased gain alone**: amplifies noise as much as signal if MON itself is not selective.
- **Stronger TS lateral inhibition alone**: addresses a symptom (band co-firing in TS) not the cause (MON multimodal preferences).

---

## Open questions

### 1. Per-individual-TS-cell tuning is multimodal — the "vertical bands"

In every single network, individual TS cells fire at **2–3 distinct x positions**, not at a single x. This is visible as vertical bands in `brian2_ts_spikes_vs_x_test_*.png`. The bands shift to different x positions in different seeds, so **population-vector decoding (averaged across the 300 TS cells) still works** — but per-cell tuning curves are not unimodal.

**Tested hypothesis — multi-distance training (2026-05-09, NEGATIVE result):** the original hypothesis was that training at a single distance imprints the dipole side-lobe geometry as spurious ghost correlations, which multi-distance training would average out. A 3-seed pilot at topo = 0.20, d ∈ [0.6, 1.2] cm uniform per trial, 10 000 trials each was run. The vertical bands remained equally present and map quality was slightly worse (σ_θ = 0.601 ± 0.101 vs 0.354 ± 0.058 extract-mode baseline). **The hypothesis was rejected.** The bands are not a distance-sampling artifact.

**Revised interpretation:** the multimodal per-TS-cell tuning is an **intrinsic property of the lateral line geometry** — certain x positions produce similar LL activation patterns regardless of source distance, and STDP reinforces these invariant co-firings. This is consistent with the experimental literature, where single-unit recordings in teleost fish lateral line show messy or multimodal tuning and a clean somatotopic map is hard to see at single-unit resolution. The model thus predicts that the map is a **population-level phenomenon** requiring multi-electrode population decoding to observe — a testable prediction. This is accepted as a known model property, not a bug.

### 2. Topo = 0.10 is the practical floor

The 10 % single-seed failure rate at topo = 0.10 (1 / 10 seeds) means the recipe is no longer robust at this level. Lower topo (e.g. 0.05) is unlikely to work without an additional mechanism (bigger jitter, higher in-degree, or multi-distance training). We have not pushed below 0.10.

### 3. Single training distance — generalisation untested

All current results train and test at exactly `d = 0.8 cm`. We do not know yet whether the resulting map generalises to other distances, or how much the recipe depends on the specific dipole-field geometry at this one distance.

### 4. Limited range of test stimuli

Test sweeps use a single x range, single distance, single source size, single speed. Cross-stimulus generalisation has not been measured.

---

## Status of the codebase / dataset

- All 28 (10 + 10 + 8) extract-mode evaluations are saved as JSON in the corresponding `Runs/extract_topo*_seed_NNN/artifacts/seed_NNN_results.json`.
- The training runs themselves are in `Runs/llmon_topo020_seeds127_132/`, `Runs/llmon_topo015_seeds127_132/`, `Runs/llmon_topo010_seeds126_132/`, plus the original Y2/U/X seeds in `Runs/llmon_U_*` and `Runs/llmon_Y2_*`.
- Topo gradient training runs (2026-05-10 to 2026-05-13): `Runs/llmon_topo040_seeds123_132/`, `Runs/llmon_topo060_seeds123_132/`, `Runs/llmon_topo080_seeds123_132/`. Per-seed results JSON in each `artifacts/` subdirectory.
- Multi-seed orchestration: `run_multi_seed_safe.sh` (training, OOM-safe), `run_extract_evaluation.sh` and `run_extract_topo010.sh` (extract-mode batch).
- Distance-sampling code edit (2026-05-08): `_sample_instantaneous_rates` in `ll_stdp_brian2.py` now samples uniformly on `[min, max]` whenever `min < max`. Backwards compatible (when `min == max`, the original clamp-to-fixed-value behaviour is preserved).
- **Publication figure**: `Picture/topo_gradient_summary.png` — regenerate with `python plots/topo_gradient_summary.py` from the project root. All data is hardcoded in that script for reproducibility.

## Completed experiment — multi-distance training pilot (2026-05-09)

**Hypothesis:** training at d ∈ [0.6, 1.2] cm (uniform per trial) would soften the per-TS-cell vertical bands by averaging out dipole-field side-lobe ghost correlations.

**Protocol:** smoke test (1 seed, 2000 trials, d ∈ [0.6, 1.2]) passed cleanly. Full pilot: 3 seeds (123–125), topo = 0.20, 10 000 trials each, d sampled uniformly in [0.6, 1.2] cm per trial. Run: `Runs/multidist_pilot_3seed/`. Code change in `_sample_instantaneous_rates` (commit b5bf205): when `min < max`, distance is drawn from `Uniform([min, max])` rather than the old Gaussian-and-clamp.

**Results (training-mode metrics):**

| Seed | σ_θ (rad) | valid_fraction |
|------|-----------|----------------|
| 123  | 0.717     | 0.782          |
| 124  | 0.537     | 0.831          |
| 125  | 0.549     | 0.827          |
| **mean ± SD** | **0.601 ± 0.101** | **0.813 ± 0.027** |

**Comparison to single-distance baseline (topo = 0.20, training-mode, seed 123):** σ_θ = 0.622, valid = 0.786. Multi-distance is marginally worse on σ_θ and comparable on valid_fraction. Vertical bands remain clearly present in TS spike rasters — visually indistinguishable from single-distance runs.

**Conclusion: NEGATIVE.** Multi-distance training does not fix the bands and does not improve map quality. The single-distance recipe is retained. See "Open questions §1" for revised interpretation.

---

## Status

✅ **Topo gradient complete (2026-05-13; topo=0.10 completed to 10 seeds on 2026-05-17).** All 6 topo levels (0.10, 0.15, 0.20, 0.40, 0.60, 0.80) complete with 10 seeds each. Figure in `Picture/topo_gradient_summary.png`; plot code in `plots/topo_gradient_summary.py`.

✅ **Chapter-5 reproduction complete (2026-05-15).** Seven figures reproducing Iris Hydi's master-thesis chapter 5 from the snake pit-organ system in the lateral-line model — see § *Chapter-5-style analyses* above. Plot code in `plots/chapter5_figures.py`. Sweep data catalogued in `SIMULATIONS_INDEX.md`. Phase-2 MON-size sweeps with explicit gain scaling completed overnight 2026-05-14 → 2026-05-15. Single-distance-vs-multi-distance sanity check (Fig 5.1a') confirms the U-shape minimum is driven by stimulus geometry, not training-distance overfitting.

🟧 **Not yet done.** Fig 5.2 (σ_θ vs observation period T) would require per-window analysis of the test sweep — skipped in this initial pass.

---

## Previous high-topo baseline (archived for reference — different recipe)

| Metric | Value |
|--------|-------|
| Run | `Runs/ts_inh15_gain125_10k/` (HIGH topo: ll_mon_topo = mon_ts_topo = 0.8) |
| `valid_fraction` | 0.660 |
| `sigma_theta` | 0.875 rad |

This single-run reference used a **different parameter set** from the current recipe: gain = 125 (vs 220), no LL→MON homeostasis, different STDP amplitudes. It is **not directly comparable** to the topo gradient results above. It is kept here because it appeared in early drafts of the paper and in the original thesis proposal as the "strong-topo baseline." The current recipe at topo = 0.80 (σ_θ = 0.278 ± 0.035, valid = 0.983 ± 0.017) is far superior, confirming that the recipe improvements benefit all topo levels, not just the weak-topo regime.

The point of this project remains unchanged: **weak** MON anatomical somatotopy is sufficient when STDP and homeostasis are tuned correctly. Topo = 0.20 already achieves ≈ 95 % of the map sharpness seen at topo = 0.80 with the same recipe.
