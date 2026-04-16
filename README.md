# LateralLine

Brian2 simulation of a lateral-line–inspired network: **LL → MON → TS** with STDP on MON→TS (and optional STDP on LL→MON), hydrodynamic input from `stimulus.py`, and population-vector (PV) map-quality metrics during training.

## Requirements

- Python 3.10+ (typical)
- [Brian2](https://brian2.readthedocs.io/), NumPy, Matplotlib
- Run the driver from the project directory (or anywhere): `ll_stdp_brian2.py` adds its own directory to `sys.path` so `stimulus` imports resolve.

```bash
python ll_stdp_brian2.py --help
```

Long runs: redirect stdout/stderr to files under the existing **`Logs/`** directory (e.g. `Logs/run01.txt`). **`Runs/`** holds per-run outputs; both are listed in `.gitignore` for GitHub.

---

## Parameter modes (`--mode`)

| Mode | Purpose |
|------|---------|
| `ll_thesis` (default) | Full thesis-scale network: `n_ll=100`, `n_mon=3200`, `n_ts=300`, long training (`n_training_trials=1000`), tuned STDP/inhibition/topography. |
| `ll_fast` | Smaller/faster smoke test: `n_mon=800`, `n_ts=80`, `n_training_trials=40`, shorter trials. |

CLI overrides (below) are applied **after** the mode preset via `dataclasses.replace` on `NetworkParams`.

---

## CLI reference (`ll_stdp_brian2.py`)

All flags are optional unless noted. Defaults match `NetworkParams` plus mode-specific values from `apply_model_mode()`.

### Run layout and seeds

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | choice | `ll_thesis` | `ll_thesis` or `ll_fast`. |
| `--run-name` | str | *(timestamp)* | Folder name under `Runs/<name>/`. Sanitized (`/`, `\` → `_`). |
| `--save-tag` | str | `latest` | Base name for saved `.npz` / `*_params.json` in `artifacts/`. |
| `--multi-seed` | int | `1` | Number of consecutive seeds to run (`seed`, `seed+1`, …). |
| `--seed-start` | int | `123` | First seed. |
| `--save-weights-dir` | str | `SavedModels/ll_thesis_no_noise` | Directory for learned-weight snapshots (see script usage; primary run outputs use `Runs/.../artifacts`). |

### Training / stimulus overrides

| Flag | Description |
|------|-------------|
| `--n-training-trials` | Override number of training trials (≥ 0). |
| `--distance-cm` | Test distance Y (cm) for `make_test_rates()` (must be > 0). |
| `--training-distance-min-cm` | Min training distance (cm) for snapshot sampling. |
| `--training-distance-max-cm` | Max training distance (cm) for snapshot sampling. |
| `--ll-mon-topo` | `ll_to_mon_topography_strength`. |
| `--mon-ts-topo` | `mon_to_ts_topography_strength`. |
| `--ll-rate-mode` | `raw` or `modulation` (LL drive). |
| `--ll-baseline-subtract-hz` | Extra LL baseline subtraction before Poisson drive (Hz). |
| `--ll-rate-gain` | Multiplicative gain on LL rates. |

### LL→MON STDP (optional plasticity)

| Flag | Description |
|------|-------------|
| `--use-ll-mon-stdp` | Enable STDP on LL→MON. |
| `--ll-mon-apre` | LL→MON pre trace increment. |
| `--ll-mon-apost` | LL→MON post trace increment. |
| `--ll-mon-wmax-mv` | LL→MON weight ceiling (mV). |
| `--ll-mon-w-init-mv` | Initial mean LL→MON weight (mV) when STDP is on. |

### MON→TS structure and inhibition

| Flag | Description |
|------|-------------|
| `--mon-ts-sigma` | Weak-topography spread (TS index units). |
| `--mon-ts-out-degree` | MON→TS out-degree per MON neuron. |
| `--ts-lateral-radius` | TS lateral inhibition radius (indices). |
| `--ts-local-inh-peak-mv` | TS lateral inhibition peak (mV). |
| `--use-ts-feedback-inh` | Enable TS activity-dependent feedback inhibition. |
| `--ts-feedback-drive-mv` | TS→global-inh drive strength (mV). |
| `--ts-feedback-inh-mv` | Global-inh→TS strength (mV). |
| `--ts-feedback-p` | TS→global-inh connection probability in [0, 1]. |
| `--mon-global-inh-mv` | Global inhibition onto MON (mV). |
| `--mon-ts-gain-mv` | MON→TS EPSP gain (mV per unit weight). |
| `--bg-rate-mon-hz` | MON background Poisson rate (Hz). |
| `--bg-rate-ts-hz` | TS background Poisson rate (Hz). |
| `--bg-w-ts-mv` | TS background synaptic weight (mV). |

### Test-phase behavior

| Flag | Description |
|------|-------------|
| `--keep-mon-ts-stdp-during-test` | Do not zero MON→TS STDP before the test phase (default freezes plasticity). |
| `--test-using-held-snapshots` | Test uses held snapshots (training-style sampling) instead of the continuous sweep. |
| `--disable-all-stdp` | Control: LL→MON and MON→TS STDP increments set to zero (no learning). |

---

## Run outputs: directory layout

Each invocation creates:

```text
Runs/<run_name_or_timestamp>/
  params.json          # full resolved config
  run_summary.json     # compact metrics (see below)
  figures/             # PNG diagnostics
  artifacts/           # weights + sidecar JSON
```

`<run_name_or_timestamp>` is `--run-name` if set, otherwise `YYYYMMDD_HHMMSS`.

---

## Figure files (`figures/`)

Filenames embed **`u_<speed>_nMON_<n_mon>_nTS_<n_ts>`** (speed in cm/s). If `--multi-seed` > 1, per-seed figures also include **`_seed_<seed>`**. After all seeds finish, the script saves a second set **without** `_seed_*` using the **first** seed’s result (`results[0]`), so unseeded names align with seed `seed-start`.

| Basename pattern | Content |
|------------------|---------|
| `brian2_spatial_two_stage_*.png` | **Summary:** stimulus overview; LL/MON/TS spike rasters; MON/TS population rates; MON→TS weight hist before/after; stabilization (mean/std + tracked weights); PV metrics vs training time. |
| `brian2_test_phase_only_*.png` | **Test segment only** (time reset to 0): sphere X vs time, LL/MON/TS rasters, population rates (20 ms smooth). |
| `brian2_learning_curves_*.png` | MON→TS weight dynamics + mean abs delta; PV (`sigma_theta`, `delta_trial`) vs training time. |
| `brian2_llmon_weights_*.png` | Example MON neurons: stem plot of LL index vs LL→MON weight (mV). |
| `brian2_ts_tuning_*.png` | Subset of TS neurons: estimated firing vs position **x** during test sweep. |
| `brian2_mon_ts_weight_profile_*.png` | Mean MON→TS weight vs TS index + fraction at `w_max` (twin axis). |
| `brian2_mon_to_ts_receptive_fields_*.png` | Incoming weights vs MON index for example TS cells. |
| `brian2_ts_spikes_vs_x_test_*.png` | TS index vs stimulus **x** during test (somatotopy diagnostic). |
| `brian2_mon_spikes_vs_x_test_*.png` | MON index vs **x** during test. |
| `brian2_mon_tuning_examples_*.png` | Example MON neurons: rate vs **x** during test. |
| `brian2_mon_ts_drive_heatmap_*.png` | Heatmap: MON→TS feedforward drive ∝ Σ spikes·weight·gain vs **x** and TS index. |
| `brian2_mon_ts_drive_winner_*.png` | Argmax TS drive vs **x** (winner take-all index). |
| `brian2_ts_pop_rate_train_test_transition_*.png` | TS population rate (20 ms smooth) around train→test window. |

**Mode `ll_thesis` only (copies of the first summary / test-only figures):**

| File | Description |
|------|-------------|
| `LL_THESIS_BASELINE_ACTIVE_latest.png` | Copy of the main spatial two-stage summary figure (latest run’s unseeded name). |
| `LL_THESIS_BASELINE_TEST_ONLY_latest.png` | Copy of the test-phase-only figure. |

**Multi-seed (`--multi-seed` > 1):**

| File | Description |
|------|-------------|
| `brian2_multiseed_summary_n<N>.png` | PV `sigma_theta` and `delta_trial` vs seed index for all runs. |

---

## JSON and artifacts

### `params.json` (run root)

Written once at run start. Shape:

- **`mode`**: CLI `--mode` string.
- **`seed_start`**: `--seed-start`.
- **`multi_seed`**: effective number of seeds (`--multi-seed`, clamped to ≥ 1).
- **`run_folder`**: directory name under `Runs/`.
- **`network_params`**: full `NetworkParams` as a flat dictionary (`dataclasses.asdict`), including sizes, timings, STDP, inhibition, stimulus, seeds, flags such as `keep_mon_ts_stdp_during_test`, `test_using_held_snapshots`, etc.

### `run_summary.json` (run root)

Compact metrics for archiving:

- Always includes: **`ll_mon_topo`**, **`mon_ts_topo`**, **`mon_ts_sigma`**, **`mon_ts_gain_mV`** (floats from resolved `NetworkParams`).
- **`multi_seed` == 1:** adds **`seed`**, **`ts_spikes_during_test_window`**, and **`pv_map_quality`** with **`sigma_theta_rad`**, **`delta_trial_rad`**, **`valid_fraction`** (PV map-quality scalars from the run).
- **`multi_seed` > 1:** same topology fields as above, plus **`runs`**: an array of objects, each with **`seed`**, **`ts_spikes_during_test_window`**, and **`pv_map_quality`** as for the single-seed case.

### `artifacts/<save-tag>_seed_<seed>.npz`

Compressed NumPy archive:

| Array | Description |
|-------|-------------|
| `mon_ts_w` | MON→TS weights after training. |
| `mon_ts_i`, `mon_ts_j` | MON and TS indices for MON→TS synapses. |
| `ll_mon_i`, `ll_mon_j`, `ll_mon_w_mV` | LL→MON connectivity and weights (mV). |

### `artifacts/<save-tag>_seed_<seed>_params.json`

Small sidecar for reloading; **not** a full `NetworkParams` dump. Fields include:

`mode` (currently written as **`"ll_thesis"`** in code regardless of CLI mode), `n_ll`, `n_mon`, `n_ts`, `speed_cm_s`, `distance_cm`, `direction`, `training_noise_scale_early`, `training_noise_scale_late`, `mon_ts_apre`, `mon_ts_apost`, `mon_ts_wmax`.

If you rely on this file for provenance, cross-check with `params.json` at the run root for the authoritative full configuration.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
