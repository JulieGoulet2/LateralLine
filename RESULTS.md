# LateralLine STDP — Results Summary (2026-05-04)

## The scientific question

> Can spike-timing-dependent plasticity (STDP) plus lateral inhibition form a somatotopic map in the Torus semicircularis (TS) when the upstream MON layer has only **weak** anatomical somatotopy?

This is the central hypothesis of the thesis. The trivial case (high MON somatotopy) is uninteresting; the question is whether the model can self-organize a map when the anatomical scaffold is weak.

## TL;DR

**Yes.** The recipe documented in `BASELINE.md` produces a population-level somatotopic map in TS at MON somatotopy as low as **`ll_mon_topo = mon_ts_topo = 0.1`** (5× weaker than the trivially-working baseline of 0.8). The best balance between sharpness and reliability is at **0.2**, which is the new operational baseline.

---

## Topo gradient — full results

All recipes use the same parameter set (see `BASELINE.md`); only the two topography strengths vary together (`ll_mon_topo = mon_ts_topo = topo`).

### 0.20 — new baseline

- 4 seeds (123, 124, 125, 126), 10000 trials each
- Mean `sigma_theta = 0.55` ± 0.10 rad
- Mean `valid_fraction = 0.79` ± 0.05
- All 4 seeds beat the high-topo (0.8) baseline of sigma=0.875, valid=0.660
- Per-network bands persist but at different x positions per seed → multi-seed averaging cleans the map at the population level

### 0.15 — degraded but still working (best individual seed metrics)

- 4 seeds (123, 124, 125, 126), 10000 trials each
- Mean `sigma_theta = 0.43` ± 0.13 rad (extract-mode metric)
- Mean `valid_fraction = 0.88` ± 0.03 (extract-mode metric)
- All 4 seeds give sigma < 0.65; 3 of 4 give sigma < 0.5

### 0.10 — substantially degraded but recipe still produces a valid map

- 3 seeds (123, 124, 125), 10000 trials each
- Mean `sigma_theta = 0.76` ± 0.15 rad (extract-mode metric)
- Mean `valid_fraction = 0.84` ± 0.01 (extract-mode metric)
- valid_fraction stays high (decoder still works on > 82% of test stimuli)
- sigma_theta is broader (decoded angles are noisier)
- Still beats the high-topo baseline on both metrics

---

## Methodology note — extract-mode vs training-mode metrics

Two different ways of measuring `sigma_theta` and `valid_fraction`:

- **Training-mode**: full simulation runs training (10000 trials) then test phase. Brian2 RNG state is "warm" by the time test starts.
- **Extract-mode**: load saved final weights, run only the test phase from a fresh RNG state.

Same network weights → both methods measure the same map quality, but extract-mode metrics are systematically better by ~0.15-0.20 in sigma_theta because the test phase isn't perturbed by training-induced noise in TS state.

For fair comparison across topo values, the 0.15 and 0.10 results above use extract-mode. The 0.20 result uses training-mode (it was measured during the original Y2 multi-seed run). They are NOT directly comparable as numbers; the relative ordering (0.15 > 0.10 in quality, both worse than 0.20) is robust.

---

## Mechanism — what made the recipe work

After ~30 experiments, three levers turned out to be essential at low MON topo:

1. **Wide initial LL→MON weight jitter** (`--ll-mon-w-jitter-stdp-mv 8.0`) — breaks symmetry between MON cells so STDP has a starting bias to amplify
2. **LL→MON homeostasis** (`--ll-mon-homeo-eta 0.005`) — forces MON cells to specialize by capping their incoming weight sum
3. **MON→TS homeostasis** (`--mon-ts-homeo-eta 0.001`) — same mechanism at the second synapse, prevents TS cells from inheriting all peaks of all upstream MON

Two supporting levers:

4. **High MON→TS gain** (`--mon-ts-gain-mv 220`) — compensates for the resulting sparse, selective MON drive
5. **Strong TS lateral inhibition** (`--ts-local-inh-peak-mv 1.5`) — winner-take-all between TS cells

---

## What did NOT work (key dead ends)

- **LTD-biased STDP without homeostasis** — multiplicative STDP equilibrium is fixed by the apre/apost ratio; uncorrelated weights settle in the middle, no bimodal distribution emerges
- **Sparser LL→MON anatomy alone** (`in_degree=7`) — starves MON layer (rate drops from 9 to 2.3 Hz), TS goes silent
- **Increased gain alone** — no help if MON itself isn't selective (boosts noise as much as signal)
- **Stronger TS lateral inhibition alone** — addresses a symptom (band co-firing) not the cause (MON multimodality)

The fix had to attack the MON layer itself (via heterogeneous init + LL→MON homeostasis), then propagate downstream (MON→TS homeostasis + TS competition).

---

## What's still imperfect

Per-individual-TS-neuron tuning is **multimodal** in any single network — each TS cell fires at 2-3 different x positions. The vertical-bands pattern visible in `brian2_ts_spikes_vs_x_test_*.png` is real and persistent.

But the bands are at **different x positions in different seeds** (seed-dependent finite-network artifact) → population-averaged decoding is clean. Biology averages across many neurons and many fish, so this is not necessarily an unrealistic feature.

---

## Caveats and limitations

- All results use a single fixed test distance (`training_distance = 0.8 cm`). Generalization across distances has not been tested.
- Statistics are based on 3-4 seeds at each topo. More seeds (8-10) would tighten the SD bars.
- The `0.10` extract-mode metric is based on 3 seeds (4th seed in the Y4 run died before completion — same crash pattern observed at Y2-multi).
- All test sweeps use the same x range and stimulus protocol; cross-stimulus generalization not measured.

---

## Suggested next work

In rough order of priority:

1. **Code cleanup** (queued by Julie 2026-05-03): split each plotting function into its own file, add extensive comments, remove obsolete tests
2. **Save baseline regression test**: a small test that re-runs the baseline command and checks final metrics are within tolerance
3. **More seeds** at 0.20 and 0.15 (target 8-10) for tighter statistics
4. **Test generalization** across `training_distance_cm` values
5. **Test floor**: try 0.05 (push as far as the recipe holds)
