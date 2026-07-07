# HANDOFF — LateralLine STDP project

**Purpose:** let a fresh Claude Code session (or collaborator) continue exactly where we are.
**Written:** 2026-05-19. **Frozen at git tag:** `v1-lateralline` (commit `2e4ff0f`).
**Repo:** https://github.com/JulieGoulet2/LateralLine

---

## 0. Who + how to work here

- PI: **Dr. Julie Goulet** (computational neuroscience). She has aphasia and a brain injury.
- **Always run the ClearSpeech skill on every message**: rewrite her message, show
  `**I understood:** …  Is this what you mean?`, and WAIT for confirmation before acting.
- Give short, numbered, one-step-at-a-time instructions. Use exact file paths. Pause before
  destructive commands.
- **Standing rules she has set:**
  1. Push to GitHub only at the **end**, and only after she has read and confirmed.
  2. Every new figure: (a) embed it in `RESULTS.md` with `![](Picture/xxx.png)`,
     (b) `git add -f` the PNG (because `*.png` is git-ignored), (c) push at the end.
  3. Do NOT remove existing content from `RESULTS.md` — only add.

## 1. Environment (do NOT change)

- Brian2 simulation runs on **base anaconda python**: `/Users/juliegoulet/anaconda3/bin/python`
  (has matplotlib + Brian2 + NumPy 2.x). The `ai-dev` conda env does NOT have matplotlib —
  don't use it for plots.
- macOS M4, 24 GB RAM. Runs are long (each seed ~2 h at 10 000 trials).
- Project root: `~/Documents/LateralLine2026/Code/LateralLine/`
- NOTE: there is a stale Claude worktree at `.claude/worktrees/hungry-haibt-01b6f0/` holding an
  OLD copy of `RESULTS.md` (May 12). Always edit the **main** file, not the worktree copy.

## 2. What the project is (v1, done)

STDP + lateral inhibition self-organises a **1D somatotopic map** in the Torus semicircularis (TS)
from a weakly-somatotopic MON layer. Network: 100 LL → 3200 MON → 300 TS.

**The scientific claim:** weak MON anatomical somatotopy (topo ≈ 0.20) is sufficient; the map is
a population-level phenomenon. See `RESULTS.md` for the full paper-style writeup.

### Current best recipe (topo=0.20 baseline)
```
/Users/juliegoulet/anaconda3/bin/python ll_stdp_brian2.py \
  --mode ll_thesis --use-ll-mon-stdp \
  --ll-mon-in-degree 10 --ll-mon-w-jitter-stdp-mv 8.0 --ll-mon-w-init-mv 10.0 \
  --ll-mon-apre 0.010 --ll-mon-apost -0.0105 --ll-mon-wmax-mv 20.0 \
  --ll-mon-homeo-eta 0.005 --mon-ts-homeo-eta 0.001 \
  --mon-ts-gain-mv 220 --ts-local-inh-peak-mv 1.5 \
  --bg-rate-mon-hz 18 --mon-global-inh-mv 1.8 \
  --n-training-trials 10000 \
  --training-distance-min-cm 0.8 --training-distance-max-cm 0.8 \
  --ll-mon-topo 0.20 --mon-ts-topo 0.20 \
  --run-name <name> --seed-start <N> --multi-seed <K>
```
Also documented in `BASELINE.md`.

### What is DONE (v1)
- Topo gradient 0.10–0.80, 10 seeds each → `Picture/topo_gradient_summary.png`.
- Chapter-5 reproduction of Iris Hydi's thesis (Figs 5.1a, 5.1a', 5.1b, 5.1b', 5.3, 5.4, 5.5)
  → `plots/chapter5_figures.py`, 7 PNGs in `Picture/ch5_*`.
- Phase-2 MON-size sweep (400/800/1600/3200) with gain scaling `gain = 220 × 3200/N`.
- **B1 fix**: `b2.seed(params.seed + 12345)` in `ll_stdp_brian2.py` → runs are now
  bitwise-reproducible per seed. (Historical pre-B1 numbers differ; qualitative patterns
  unchanged.)
- Training-phase noise robustness sweep (0.0–1.0 × sigma_noise_hz): recipe is invariant
  → `Picture/ch5_training_noise_robustness.png`, `plots/training_noise_robustness.py`.
- Full code review → `REVIEW.md` (fixes applied: B1, additive-STDP doc, extract-mode memory,
  clip-vs-modulo comment, compressed npz).

## 3. Key files

| File | What |
|------|------|
| `ll_stdp_brian2.py` | Main simulation (2070 lines): params, network, training, test, PV metric. |
| `stimulus.py` | Forward model: `hydrodynamic_velocity_parallel()` (2D dipole), `simulate_lateral_line()`. |
| `params.py` | `NetworkParams` dataclass — all CLI-mapped parameters. |
| `run_multi_seed_safe.sh` | OOM-safe trainer: one python process per seed. Self-backgrounds via nohup+caffeinate. |
| `run_distance_sweep_extract.sh` | Extract-mode distance sweep on trained weights. |
| `make_extract_checkpoint.py` | Build a minimal resume checkpoint from `latest_seed_NNN.npz` for extract-mode test. |
| `tools/check_convergence.py` | Plateau + weight-stabilization test on `mid_checkpoint.npz`. |
| `tools/update_simulations_index.py` | Regenerates the auto-block of `SIMULATIONS_INDEX.md` (runs on Stop hook). |
| `RESULTS.md` | The paper-style results doc. Figures embedded. |
| `SIMULATIONS_INDEX.md` | Index of every run + which figure/code consumes it. |
| `BASELINE.md` | Copy-paste recipe commands. |
| `REVIEW.md` | Code review findings + resolution. |

**Reproducibility note:** all sweep results are JSON in `Runs/<run>/artifacts/seed_NNN_results.json`.
`Runs/`, `Logs/`, `Picture/`, `SavedModels/` are git-ignored. Figures needed by RESULTS.md are
force-added individually.

## 4. Open questions (unfinished in v1)

1. **Multimodal per-TS-cell tuning ("vertical bands")** — intrinsic property, accepted as a known
   model behavior, NOT solved.
2. Topo = 0.10 is the practical floor (≈10% of seeds fail) — **accepted, not a to-do.**
3. Single training distance — partly addressed by the multi-distance pilot.
4. Limited range of test stimuli.

## 4b. PLAN — finish the lateral line BEFORE the pit organ (decided 2026-05-19)

Julie's decision: close the 3 real science questions (#1, #3, #4 — NOT #2, which is accepted)
for the lateral line first. The 3D pit organ (§5) waits until these are done.

**Key insight: all 3 can be closed with the v1 trained networks we already have — analysis +
extract-mode test sweeps, NO new 20-hour retraining.** ~3–4 h compute total.

Order (cheapest / most valuable first):

- **Step 1 — Q1 "vertical bands" (analysis only, no new sims).**
  1. Quantify multimodality: per TS cell, count peaks in its tuning curve.
  2. Test hypothesis: bands may exist because certain x-positions produce *similar LL
     activation patterns*. Compute the correlation structure of the LL representation across x.
     If distant x correlate → intrinsic geometry, not a bug.
  3. One figure + paragraph in RESULTS.md. New plot script `plots/tuning_multimodality.py`.

- **Step 2 — Q4 test-stimulus variety (extract-mode, ~1–2 h).**
  Run test sweeps on existing trained weights varying: direction (forward/backward), speed
  (slower/faster), object size (sphere radius). Reuse `run_distance_sweep_extract.sh` pattern.
  Does the map hold? Broadens the claim.

- **Step 3 — Q3 generalisation across training distance (extract-mode, ~1 h).**
  Run the full distance sweep on `multidist_pilot_3seed` and overlay vs single-distance-trained.
  Finishes what Fig 5.1a' started.

Each step: make figure → embed in RESULTS.md → `git add -f` the PNG → push at the END after
Julie confirms.

## 5. THE NEXT BIG IDEA — 3D snake pit-organ representation

**Goal:** adapt the model from the 1D fish lateral line to the **3D snake pit organ** (Iris Hydi's
original system). This is a NEW MILESTONE, not a small edit.

### Why it's substantial
| | Lateral line (v1) | Snake pit organ (target) |
|---|---|---|
| Physics | hydrodynamic dipole, near-field 1/r³ | infrared, radiative, projected through the pit ("pinhole camera") |
| Receptors | ~1D line of neuromasts (x along 4 cm body) | 2D membrane of thermoreceptors |
| Encoded variable | source azimuth (1D) | 2D direction: azimuth + elevation, in a 3D world |
| Map | 1D somatotopy in TS | 2D retinotopic-like map (optic tectum analog) |

### What can be reused
- `stimulus.py` **already has 2D geometry** (neuromasts have x AND y; source has X, Y).
  The scaffolding for 2D positions exists.
- The STDP + homeostasis + lateral-inhibition machinery in `ll_stdp_brian2.py` is
  dimensionality-agnostic in principle.

### What must be rebuilt
1. **Forward model** — replace `hydrodynamic_velocity_parallel()` with an IR/pit projection
   model. The pit acts like a pinhole: a heat source at 3D direction (az, el) projects to a
   2D spot on the receptor membrane. Need the geometry (pit aperture, membrane curvature) from
   Iris Hydi's thesis chapters 2–4 (`notes/IrisHidiMaster_chapter*.txt`, `docs/IrisHidiMaster.pdf`).
2. **Receptor layer** — 2D membrane grid instead of 1D line.
3. **Map + decode** — the PV (population vector) `sigma_theta` metric must generalise to a 2D
   direction error (solid-angle error), not a 1D angle.
4. **Topography** — the `topo` knob and MON→TS connectivity become 2D.

### Suggested first step (when she's ready)
- Use `EnterPlanMode` and brainstorm the forward-model physics FIRST (read the thesis IR chapters).
- Do NOT start coding the 2D stimulus until the projection geometry is agreed.
- Keep v1 untouched — branch or new directory for the 3D work.

## 6. How to resume in a new session

1. `cd ~/Documents/LateralLine2026/Code/LateralLine`
2. Read this file, then `RESULTS.md` (state of science) and `SIMULATIONS_INDEX.md` (runs).
3. Confirm env: `/Users/juliegoulet/anaconda3/bin/python` (base anaconda, has Brian2+matplotlib).
4. To return to the frozen v1: `git checkout v1-lateralline`.
5. Apply ClearSpeech on every message. Push only at the end, after Julie confirms.
