# LateralLine Simulation Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fast regression test suite that catches silent breakage in `ll_stdp_brian2.py` without running full simulations.

**Architecture:** Four focused test files, each independent. Three files use only numpy/stdlib (run in milliseconds). One file runs a minimal Brian2 simulation (5 trials, ~30 seconds). All tests run via `pytest tests/`.

**Tech Stack:** Python, pytest 7.4.3, numpy, Brian2 (one file only), pathlib, dataclasses.replace

---

## File Map

| File | Creates / Modifies | What it tests |
|------|--------------------|---------------|
| `tests/__init__.py` | Create | Makes `tests/` a package |
| `tests/test_stdp_rule.py` | Create | LL→MON STDP is multiplicative (text search, no Brian2) |
| `tests/test_checkpoint.py` | Create | Checkpoint save/load round-trip (numpy only) |
| `tests/test_stimulus.py` | Create | `make_training_rates` output shape, range, non-negative |
| `tests/test_training.py` | Create | Weights change after 5 training trials (Brian2, ~30s) |
| `tests/conftest.py` | Create | Shared `ll_fast_params` fixture |

---

## Task 1: Project scaffolding

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create the tests package**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write `tests/conftest.py` with shared fixture**

```python
# tests/conftest.py
import pytest
from dataclasses import replace
from ll_stdp_brian2 import NetworkParams, apply_model_mode


@pytest.fixture
def ll_fast_params():
    """Minimal fast params: n_mon=800, n_ts=80, 40 trials, seed=0."""
    params = apply_model_mode(NetworkParams(), "ll_fast")
    return replace(params, seed=0)
```

- [ ] **Step 3: Verify pytest collects without errors**

```bash
cd ~/Documents/LateralLine2026/Code/LateralLine
python -m pytest tests/ --collect-only
```

Expected output: `no tests ran` with no import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add test scaffold and shared ll_fast_params fixture"
```

---

## Task 2: STDP rule regression test (no Brian2)

**Files:**
- Create: `tests/test_stdp_rule.py`

This test reads `ll_stdp_brian2.py` as text and checks that the LL→MON STDP equations are multiplicative. It will fail immediately if someone accidentally changes `apost*w` to `apost` (additive).

- [ ] **Step 1: Write the failing test first**

```python
# tests/test_stdp_rule.py
import pathlib


def _ll_mon_stdp_block() -> str:
    """Extract LL->MON STDP source block from the simulation file."""
    src = (pathlib.Path(__file__).parent.parent / "ll_stdp_brian2.py").read_text()
    start = src.index("# LL->MON with STDP")
    end = src.index("# MON -> TS:", start)
    return src[start:end]


def test_ll_mon_on_pre_is_multiplicative_depression():
    """on_pre must depress proportional to current weight (apost*w), not additively."""
    block = _ll_mon_stdp_block()
    assert "apost*w" in block, (
        "REGRESSION: LL->MON on_pre uses additive LTD. "
        "Must be: w = clip(w + apost*w, 0*mV, wmax)"
    )


def test_ll_mon_on_post_is_multiplicative_potentiation():
    """on_post must potentiate proportional to distance from wmax (apre*(wmax-w))."""
    block = _ll_mon_stdp_block()
    assert "apre*(wmax - w)" in block, (
        "REGRESSION: LL->MON on_post uses additive LTP. "
        "Must be: w = clip(w + apre*(wmax - w), 0*mV, wmax)"
    )


def test_additive_pattern_not_in_ll_mon_block():
    """Guard: the bare additive pattern 'w + apost)' must not appear in LL->MON block."""
    block = _ll_mon_stdp_block()
    # Additive form ends with just 'apost,' or 'apost)' — no *w after it
    assert "w + apost)" not in block and "w + apost," not in block, (
        "REGRESSION: Found additive LTD pattern in LL->MON STDP block."
    )
```

- [ ] **Step 2: Run to confirm it passes on current code**

```bash
python -m pytest tests/test_stdp_rule.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 3: Manually verify the test catches a regression**

Temporarily edit `ll_stdp_brian2.py` line ~866: change `w + apost*w` to `w + apost`. Run tests — should FAIL with the REGRESSION message. Revert the change immediately.

- [ ] **Step 4: Commit**

```bash
git add tests/test_stdp_rule.py
git commit -m "test: add STDP multiplicative rule regression guard"
```

---

## Task 3: Checkpoint round-trip test (no Brian2)

**Files:**
- Create: `tests/test_checkpoint.py`

Tests `_load_mid_checkpoint` by manually building the `.npz` file that `_save_mid_checkpoint` would write, then loading it and checking all arrays are identical.

- [ ] **Step 1: Write the test**

```python
# tests/test_checkpoint.py
import numpy as np
import pytest
from ll_stdp_brian2 import _load_mid_checkpoint


def _write_fake_checkpoint(run_dir: "pathlib.Path", trial_idx: int, mon_ts_w: np.ndarray):
    """Write a minimal mid_checkpoint.npz matching the format _save_mid_checkpoint uses."""
    import pathlib
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "artifacts" / "mid_checkpoint.npz"
    np.savez_compressed(
        str(ckpt_path),
        trial_idx=np.array(trial_idx),
        mon_ts_w=mon_ts_w,
        checkpoint_t_s=np.array([0.0, 1.2, 2.4]),
        w_mean_series=np.array([0.020, 0.019, 0.018]),
        w_std_series=np.array([0.005, 0.006, 0.007]),
        tracked_weight_series=np.array([[0.020], [0.019], [0.018]]),
        w_mean_abs_delta_series=np.array([0.0, 0.001, 0.002]),
        w_frac_delta_gt_1e3_series=np.array([0.0, 0.05, 0.10]),
        ts_ckpt_rate_series=np.array([4.5]),
        pv_ckpt_t_s=np.array([2.4]),
        pv_sigma_theta_series=np.array([1.1]),
        pv_delta_trial_series=np.array([0.3]),
    )
    return ckpt_path


def test_load_returns_none_when_no_file(tmp_path):
    """_load_mid_checkpoint must return None if no checkpoint exists."""
    result = _load_mid_checkpoint(tmp_path)
    assert result is None


def test_load_returns_correct_trial_idx(tmp_path):
    """trial_idx must round-trip exactly."""
    mon_ts_w = np.linspace(0.010, 0.028, 50)
    _write_fake_checkpoint(tmp_path, trial_idx=499, mon_ts_w=mon_ts_w)

    ckpt = _load_mid_checkpoint(tmp_path)

    assert ckpt is not None
    assert int(ckpt["trial_idx"]) == 499


def test_load_weights_match_saved(tmp_path):
    """mon_ts_w must survive the save/load cycle without precision loss."""
    original_w = np.linspace(0.010, 0.028, 50)
    _write_fake_checkpoint(tmp_path, trial_idx=0, mon_ts_w=original_w)

    ckpt = _load_mid_checkpoint(tmp_path)

    np.testing.assert_array_equal(ckpt["mon_ts_w"], original_w)


def test_load_stats_arrays_present(tmp_path):
    """All stats arrays saved by _save_mid_checkpoint must be present after load."""
    _write_fake_checkpoint(tmp_path, trial_idx=0, mon_ts_w=np.ones(10) * 0.02)

    ckpt = _load_mid_checkpoint(tmp_path)

    required_keys = [
        "trial_idx", "mon_ts_w", "checkpoint_t_s", "w_mean_series",
        "w_std_series", "tracked_weight_series", "w_mean_abs_delta_series",
        "w_frac_delta_gt_1e3_series", "ts_ckpt_rate_series",
        "pv_ckpt_t_s", "pv_sigma_theta_series", "pv_delta_trial_series",
    ]
    for key in required_keys:
        assert key in ckpt, f"Missing key in loaded checkpoint: {key}"
```

- [ ] **Step 2: Run the tests**

```bash
python -m pytest tests/test_checkpoint.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_checkpoint.py
git commit -m "test: add checkpoint save/load round-trip tests"
```

---

## Task 4: Stimulus sanity tests (no Brian2)

**Files:**
- Create: `tests/test_stimulus.py`

Tests `make_training_rates` output shape, value range, and coverage of the full body axis. No Brian2 needed.

- [ ] **Step 1: Write the tests**

```python
# tests/test_stimulus.py
import numpy as np
import pytest
from dataclasses import replace
from ll_stdp_brian2 import NetworkParams, apply_model_mode, make_training_rates


@pytest.fixture
def fast_params():
    return apply_model_mode(NetworkParams(), "ll_fast")


def test_rates_shape_matches_params(fast_params):
    """Output shape must be (n_timesteps, n_ll)."""
    rates, _, x_trace = make_training_rates(fast_params)

    expected_steps = int(round(
        fast_params.n_training_trials * fast_params.trial_duration_s / fast_params.dt_s
    ))
    assert rates.shape == (expected_steps, fast_params.n_ll)
    assert x_trace.shape == (expected_steps,)


def test_rates_nonnegative(fast_params):
    """Firing rates must never be negative."""
    rates, _, _ = make_training_rates(fast_params)
    assert np.all(rates >= 0.0), f"Negative rates found: min={rates.min()}"


def test_x_trace_covers_full_body(fast_params):
    """Stimulus positions must span both ends of the body axis."""
    _, _, x_trace = make_training_rates(fast_params)

    assert x_trace.min() >= 0.0
    assert x_trace.max() <= fast_params.ll_body_length_cm
    # Both ends must be visited (within 10% of body length)
    margin = 0.1 * fast_params.ll_body_length_cm
    assert x_trace.min() < margin, "Stimulus never reaches head end"
    assert x_trace.max() > fast_params.ll_body_length_cm - margin, "Stimulus never reaches tail end"


def test_zero_trials_returns_empty(fast_params):
    """Zero training trials must produce empty arrays without error."""
    params = replace(fast_params, n_training_trials=0)
    rates, samples, x_trace = make_training_rates(params)

    assert rates.shape[0] == 0
    assert x_trace.shape[0] == 0
    assert samples == []


def test_deterministic_with_same_seed(fast_params):
    """Same seed must produce identical output."""
    rates1, _, x1 = make_training_rates(fast_params)
    rates2, _, x2 = make_training_rates(fast_params)

    np.testing.assert_array_equal(rates1, rates2)
    np.testing.assert_array_equal(x1, x2)
```

- [ ] **Step 2: Run the tests**

```bash
python -m pytest tests/test_stimulus.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stimulus.py
git commit -m "test: add stimulus generation sanity tests"
```

---

## Task 5: Weight change detection test (Brian2 required, ~30s)

**Files:**
- Create: `tests/test_training.py`

Runs 5 training trials on the minimal `ll_fast` network and verifies that MON→TS weights actually diverge from their initial values. If STDP is broken (stuck, additive-saturation, or zero learning rate), this test fails.

Mark the test with `@pytest.mark.slow` so it can be skipped during rapid iteration with `pytest -m "not slow"`.

- [ ] **Step 1: Write the test**

```python
# tests/test_training.py
import numpy as np
import pytest
from dataclasses import replace
from ll_stdp_brian2 import NetworkParams, apply_model_mode, run_spatial_two_stage_model


@pytest.mark.slow
def test_mon_ts_weights_diverge_after_training():
    """
    After 5 training trials, MON->TS weight std must exceed initial std.
    Catches: STDP rate=0, broken on_pre/on_post, weight clamp errors.
    """
    params = apply_model_mode(NetworkParams(), "ll_fast")
    params = replace(params, seed=0, n_training_trials=5)

    result = run_spatial_two_stage_model(params)

    w_before = result["w_before"]
    w_after = result["w_after"]

    std_before = float(np.std(w_before))
    std_after = float(np.std(w_after))

    assert std_after > std_before, (
        f"Weight distribution did not diversify after training. "
        f"std before={std_before:.6f}, std after={std_after:.6f}. "
        f"STDP may be broken."
    )


@pytest.mark.slow
def test_some_weights_change_after_training():
    """At least 10% of weights must move by more than 1e-4 after 5 trials."""
    params = apply_model_mode(NetworkParams(), "ll_fast")
    params = replace(params, seed=0, n_training_trials=5)

    result = run_spatial_two_stage_model(params)

    abs_change = np.abs(result["w_after"] - result["w_before"])
    frac_changed = float(np.mean(abs_change > 1e-4))

    assert frac_changed > 0.10, (
        f"Only {frac_changed:.1%} of weights changed by >1e-4. "
        f"Expected >10%. STDP may be inactive."
    )


@pytest.mark.slow
def test_weights_stay_within_bounds():
    """All weights must remain in [0, mon_ts_wmax] after training."""
    params = apply_model_mode(NetworkParams(), "ll_fast")
    params = replace(params, seed=0, n_training_trials=5)

    result = run_spatial_two_stage_model(params)

    w = result["w_after"]
    assert np.all(w >= 0.0), f"Weights went negative: min={w.min()}"
    assert np.all(w <= params.mon_ts_wmax + 1e-9), (
        f"Weights exceeded wmax={params.mon_ts_wmax}: max={w.max()}"
    )
```

- [ ] **Step 2: Run slow tests**

```bash
python -m pytest tests/test_training.py -v -m slow
```

Expected: all 3 tests PASS. Takes ~30–60 seconds (Brian2 compilation + 5 trials).

- [ ] **Step 3: Verify fast tests skip the slow ones**

```bash
python -m pytest tests/ -v -m "not slow"
```

Expected: 12 fast tests PASS, 3 slow tests SKIPPED.

- [ ] **Step 4: Commit**

```bash
git add tests/test_training.py
git commit -m "test: add weight-change detection tests (Brian2, marked slow)"
```

---

## Task 6: Full suite verification

- [ ] **Step 1: Run everything**

```bash
python -m pytest tests/ -v
```

Expected output:
```
tests/test_stdp_rule.py::test_ll_mon_on_pre_is_multiplicative_depression PASSED
tests/test_stdp_rule.py::test_ll_mon_on_post_is_multiplicative_potentiation PASSED
tests/test_stdp_rule.py::test_additive_pattern_not_in_ll_mon_block PASSED
tests/test_checkpoint.py::test_load_returns_none_when_no_file PASSED
tests/test_checkpoint.py::test_load_returns_correct_trial_idx PASSED
tests/test_checkpoint.py::test_load_weights_match_saved PASSED
tests/test_checkpoint.py::test_load_stats_arrays_present PASSED
tests/test_stimulus.py::test_rates_shape_matches_params PASSED
tests/test_stimulus.py::test_rates_nonnegative PASSED
tests/test_stimulus.py::test_x_trace_covers_full_body PASSED
tests/test_stimulus.py::test_zero_trials_returns_empty PASSED
tests/test_stimulus.py::test_deterministic_with_same_seed PASSED
tests/test_training.py::test_mon_ts_weights_diverge_after_training PASSED
tests/test_training.py::test_some_weights_change_after_training PASSED
tests/test_training.py::test_weights_stay_within_bounds PASSED

15 passed in ~60s
```

- [ ] **Step 2: Add pytest config so `pytest` runs fast tests by default**

Add `pytest.ini` at project root:

```ini
[pytest]
testpaths = tests
markers =
    slow: Brian2 simulation tests (~30-60s each). Run with: pytest -m slow
```

- [ ] **Step 3: Final commit**

```bash
git add pytest.ini
git commit -m "test: add pytest.ini with slow marker and default testpaths"
```

---

## Quick Reference

```bash
# Fast tests only (~1s)
python -m pytest -m "not slow" -v

# All tests including Brian2 (~60s)
python -m pytest -v

# Single file
python -m pytest tests/test_stdp_rule.py -v

# After changing STDP equations — run this first
python -m pytest tests/test_stdp_rule.py tests/test_training.py -v
```
