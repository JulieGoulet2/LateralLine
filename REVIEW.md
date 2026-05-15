# LateralLine STDP — Code Review

**Reviewed:** 2026-05-15
**Scope:** `ll_stdp_brian2.py`, `params.py`, `stimulus.py`, `plots/`, `run_*.sh`, `make_extract_checkpoint.py`, `tests/`, `.claude/settings.json`

## Executive summary

LL→MON multiplicative STDP, toroidal TS inhibition, dipole formula, and per-seed-OOM-safety are all correct. Two BLOCKERs threaten Chapter-5 reproducibility: **Brian2's internal RNG is never seeded** (Poisson spikes + `rand()` weight init vary per run for "the same" seed), and `plots/topo_gradient_summary.py` embeds hardcoded per-seed numbers that no longer match `Runs/`. MON→TS STDP is **additive**, conflicting with CLAUDE.md's stated rule — needs an explicit scientific decision. Several smaller issues in extract-mode flow, homeostasis-on-resume, and test coverage.

---

## BLOCKER

### B1 — Brian2 RNG is never seeded
**`ll_stdp_brian2.py:719–722, 749–750, 770, 813, 858, 861–862`**
NumPy RNGs are correctly seeded via `np.random.default_rng(params.seed[+offset])`, but every Brian2 stochastic source — `PoissonGroup` (LL, bg_mon, bg_ts), `rand()` in `mon.v="El+rand()*2*mV"`, and `rand()` inside weight-init strings — uses Brian2's own RNG. `grep -n "b2.seed" *.py` returns nothing. After `b2.start_scope()` Brian2 reseeds from the OS, so two runs of the same Python process with `--seed-start 127` produce different LL spike trains, different MON/TS v(0), and different jittered initial weights.
**Impact:** Multi-seed SD over-estimates true variance; same-seed extract-mode is not bitwise reproducible; chapter-5 means absorb a hidden non-reproducible component.
**Fix:** Add `b2.seed(int(params.seed))` immediately after `b2.start_scope()` on line 720.

### B2 — `plots/topo_gradient_summary.py` hardcodes numbers that no longer match disk
**`plots/topo_gradient_summary.py:29–50`**
The script claims `sigma_010/015/020` are "extract-mode", but spot-checks fail: hardcoded `sigma_020[0]=0.421`, while `Runs/extract_topo020_seed_127/artifacts/seed_127_results.json` gives 0.382 and the corresponding training-mode JSON gives 0.399. Neither matches 0.421. The means are unverifiable from `Runs/`.
**Fix:** Load JSONs via `_collect_sweep`-style code (as in `chapter5_figures.py`). Until then, do not cite this figure as reproducible.

### B3 — `ll_mon_homeo_target` is computed before checkpoint weights are restored
**`ll_stdp_brian2.py:815–820` (target init) vs `:921` (restore) vs `:1043` (use)**
`ll_mon_homeo_target` is set from freshly-initialized `s_ll_mon.w` (random init), then the checkpoint restore overwrites `s_ll_mon.w`. For mid-training crash recovery the homeostatic target is the *initialization* sum, not the trained one. (Extract-mode is unaffected because `remaining_training_trials=0`.)
**Fix:** Recompute `incoming0` and `ll_mon_homeo_target` *after* the resume-restore block, or persist the target in the checkpoint.

---

## HIGH

### H1 — MON→TS STDP is additive, contradicting CLAUDE.md
**`ll_stdp_brian2.py:839–847`**
```
on_pre:  w = clip(w + apost, 0, wmax)
on_post: w = clip(w + apre,  0, wmax)
```
CLAUDE.md (lines 38–40) says the rule must be multiplicative (`apost*w`, `apre*(wmax-w)`). LL→MON (lines 792–800) IS multiplicative and has a regression test; MON→TS is not, and has no rule-test. The published chapter-5 runs use this additive rule, so either CLAUDE.md is wrong or the science is. **Needs a deliberate decision**, not a silent change.
**Fix:** Pick one. If multiplicative is intended: change the rule and re-run baselines. If additive is intentional: update CLAUDE.md and add a `test_mon_ts_rule.py` mirroring `test_stdp_rule.py`.

### H2 — Extract-mode allocates the full 10000-trial training stimulus and discards it
**`ll_stdp_brian2.py:677, 697–702`**
`make_training_rates` always runs. At `n_training_trials=10000`, `dt_s=1e-3` it produces a `(12_000_000, 100)` float64 array (~9.6 GB), then `train_rates[12_000_000:]` slices it away. On a 24 GB Mac this risks OOM and wastes minutes per distance.
**Fix:** When `resume_checkpoint is not None and k_start >= params.n_training_trials`, skip `make_training_rates` and set `train_rates = np.zeros((0, n_ll))`.

### H3 — Topographic index sampling clips, not wraps — but docstring claims a ring wrap
**`ll_stdp_brian2.py:99–100, 153–154`**
The comment at 153 says "Ring wrap in index space (no seam at 0 / n_ll-1)" but the code is `np.clip(np.rint(raw), 0, N-1)`. Both `build_mon_to_ts_indices` and `build_ll_to_mon_indices` clip; neurons near body ends accumulate extra topographic connections. Especially visible when `mon_to_ts_sigma=120` and `n_ts=300` (σ ≈ N/2).
**Fix:** Use `np.mod(np.rint(raw).astype(int), N)` and keep the docstring, or fix the docstring/comment to say "clipped".

### H4 — Two incompatible `seed_NNN_results.json` schemas; readers silently NaN
**`ll_stdp_brian2.py:1881–1899`** writes 9 fields. Older runs on disk have only 4 (`seed`, `sigma_theta_rad`, `valid_fraction`, `delta_trial_rad`). `run_distance_sweep_extract.sh:153–159` reads `sigma_w_ll_cm`/`sigma_w_ts_cm` — works only on new schema. `plots/chapter5_figures.py:71` uses `dict.get(field, np.nan)`, silently dropping seeds rather than erroring.
**Fix:** Stamp a `schema_version`; have plot code warn loudly when expected fields are missing.

### H5 — `make_test_rates_held_snapshots` ignores `test_ll_noise_hz`
**`ll_stdp_brian2.py:377–402`**
This path applies the training noise curriculum (`training_noise_scale_early/_late`) but never reads `test_ll_noise_hz`. A user passing `--test-using-held-snapshots --test-ll-noise-hz 5` gets test-noise=0 silently.
**Fix:** Apply test_ll_noise_hz in this branch too, or error if both are set.

---

## MEDIUM

### M1 — `_save_mid_checkpoint` is not atomic across crash boundaries
**`ll_stdp_brian2.py:640–659`** Unlinks destination *then* renames temp. A kill between the two leaves no `mid_checkpoint.npz` (resume silently starts from scratch). Skip the explicit unlink — POSIX `rename` is atomic and overwrites.

### M2 — `make_extract_checkpoint.py` uses uncompressed `np.savez` while production uses `savez_compressed`
**`make_extract_checkpoint.py:28`** Works because `np.load` handles both; just an inconsistency. Switch to `savez_compressed`.

### M3 — Brian2-driven `connect(p=...)` calls also share the unseeded RNG
**`ll_stdp_brian2.py:870, 898–899`** MON→inh and TS→inh wiring is non-reproducible per seed. Mitigated by fixing B1.

### M4 — `pv_map_quality_from_ts_spikes` returns `sigma_theta = π` (max-uncertainty fallback) when TS is silent
**`ll_stdp_brian2.py:598`** Plot code then averages π in with real values, dragging means up. Return `np.nan` and use `np.nanmean`/`np.nanstd` downstream. Add a `pv_silent_ts` boolean so "real σ near π" is distinguishable from "no spikes".

### M5 — Distance sweep re-runs `make_training_rates` 30 times unnecessarily
Combined with H2's fix this becomes free.

### M6 — Tests do not cover the simulation core meaningfully
- `tests/test_regression.py` is the only end-to-end test, marked `@pytest.mark.slow` → not in the PreToolUse/PostToolUse hooks (`-m 'not slow'`).
- Fast suite covers: regex on LL→MON STDP, `make_training_rates` (no Brian2), checkpoint round-trip, stimulus.
- Fast suite would NOT catch B1, B3, H1, H3, M4. The hooks claim to enforce quality but mostly enforce "syntactically intact".
- No golden-output regression (e.g. seed=0, 5 trials → `pv_sigma_theta` ± 1e-6).
**Fix:** Add (a) `test_brian2_seed_reproducible.py` (two short sims, identical `sp_ts.t`); (b) `test_mon_ts_rule.py` mirroring the LL→MON regex check; (c) a golden-output regression on `ll_fast` pinned to a stored npz.

---

## LOW

- **L1** `stimulus.py:104` — 1e-12 Cholesky regularizer is on the edge for n_neuromasts=100; bump to 1e-9.
- **L2** `ll_stdp_brian2.py:299` — when `training_fixed_distance=True`, code passes `stim_params.mu_distance_cm` (1.5) and silently ignores `training_distance_min/max_cm`. In `ll_thesis` mode fixed_distance=False so safe, but a direct caller can be bitten.
- **L3** Magic seed offsets (`+999`, `+777`, `+8888`, `+31`) scattered across the file; no overlap today, but no contract either. Promote to module-level constants.
- **L4** `pytest.ini` does not register the `slow` marker; harmless without `--strict-markers`.
- **L5** `run_multi_seed_safe.sh:98` PYTHON fallback could resolve to base anaconda; fragile.
- **L6** No `OMP_NUM_THREADS=1` in sweep scripts; fine for sequential, footgun if parallelized.
- **L7** `make_training_rates` and `make_test_rates_held_snapshots` are ~90% duplicate code (H5 is one bug-pair this caused).

## NIT

- `ll_stdp_brian2.py:929` magic `24` for tracked weights — promote to constant.
- `chapter5_figures.py:38` hardcodes `BODY_LEN_CM=4.0`; should read from each run's `params.json`.
- `tests/test_stdp_rule.py` regex-on-source tests give confusing errors if the file is moved.
- `_save_mid_checkpoint` prints unconditionally (~1000 lines / 10k-trial run).

---

## What's done well

- **Multiplicative LL→MON STDP** is correctly implemented (792–800) and protected by `test_stdp_rule.py`.
- **Per-seed-as-process OOM safety** in `run_multi_seed_safe.sh` is a correct, thoughtful design; the `--_bg` self-relaunch idiom is properly idempotent.
- **NumPy RNG discipline** — every numpy call uses `default_rng(seed)`; no `np.random.seed` anywhere.
- **Toroidal TS lateral inhibition** (879–889) is correct (`min(d, N-d)`, peak at 0, zero at radius, no off-by-one).
- **PV decode** correctly produces per-bin NaN when TS is silent and tracks `valid_fraction` (modulo M4's π fallback at the aggregate level).
- **Resume-from-checkpoint** correctly drops completed trials from the stimulus stream and restores both weight projections.
- **Dipole formula** in `stimulus.py` matches the standard moving-sphere parallel-velocity expression; `A_per_cm=300` matches the documented conversion from `A=30000 m⁻¹`.
- **Test-phase plasticity freeze** by zeroing `Apre/Apost` in the namespace (1066–1068) is clean and obviously correct.

---

## Reproducibility verdict

**Partial.** With current `Runs/` present, `chapter5_figures.py` regenerates (modulo H4 NaN curves). `topo_gradient_summary.py` *cannot* be regenerated from data (B2). From a clean `Runs/` nothing is bit-reproducible because Brian2's RNG is unseeded (B1) — same-seed runs differ across machines and across invocations. Chapter-5 means are statistically meaningful but not deterministic.

**Minimum work to make chapter 5 reproducible from a clean checkout:**
1. Fix B1 (one line: `b2.seed(int(params.seed))`).
2. Fix B2 (load from JSON instead of hardcoded arrays).
3. Decide H1 (additive vs multiplicative MON→TS) and document.
4. Re-train + re-extract; archive resulting `Runs/` as the chapter-5 data snapshot.
