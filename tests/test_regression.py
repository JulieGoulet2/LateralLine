"""
Regression test: scaled-down run with the same STDP + homeostasis configuration
as the baseline (BASELINE.md), but with n_mon=800, n_ts=80, and 200 training trials.

At 200 trials, the map has NOT converged — the baseline needs 10k trials for that.
These tests are smoke tests: did the code run correctly and produce output?
They will catch crashes, broken STDP, silent TS layers, and weight bound violations.

Runtime on M-series MacBook: ~2-3 minutes (run with: pytest -m slow tests/test_regression.py).
"""
import numpy as np
import pytest
from dataclasses import replace

import brian2 as b2
from ll_stdp_brian2 import (
    NetworkParams,
    apply_model_mode,
    run_spatial_two_stage_model,
)


@pytest.fixture(scope="module")
def baseline_proxy_result():
    """
    Run the baseline STDP+homeostasis recipe on a small network (n_mon=800, n_ts=80)
    for 200 training trials. Module scope: shared across all 5 test functions so
    the simulation runs only once (~2-3 min).
    """
    params = apply_model_mode(NetworkParams(), "ll_fast")
    params = replace(
        params,
        n_training_trials=200,
        trial_duration_s=0.8,
        seed=123,
        # Enable the full baseline LL->MON STDP pathway.
        ll_mon_use_stdp=True,
        ll_mon_apre=0.010,
        ll_mon_apost=-0.0105,
        ll_mon_wmax_mV=20.0,
        ll_mon_w_init_mV=10.0,
        ll_mon_w_jitter_stdp_mV=8.0,
        ll_to_mon_in_degree=10,
        ll_to_mon_topography_strength=0.2,
        mon_to_ts_topography_strength=0.2,
        ll_mon_homeo_eta=0.005,
        ll_mon_homeo_every_trials=10,
        mon_ts_homeo_eta=0.001,
        mon_ts_homeo_every_trials=10,
        mon_ts_gain_mV=220.0,
        ts_local_inh_peak_mV=1.5,
        bg_rate_mon_hz=18.0,
        global_inh_to_mon_mV=1.8,
        training_distance_min_cm=0.8,
        training_distance_max_cm=0.8,
    )
    return run_spatial_two_stage_model(params)


@pytest.mark.slow
def test_regression_weights_diverge(baseline_proxy_result):
    """STDP must widen the MON->TS weight distribution after 200 trials."""
    r = baseline_proxy_result
    std_before = float(np.std(r["w_before"]))
    std_after = float(np.std(r["w_after"]))
    assert std_after > std_before, (
        f"Weight std did not increase: before={std_before:.6f}, after={std_after:.6f}. "
        "STDP may be broken."
    )


@pytest.mark.slow
def test_regression_ts_fires_during_test(baseline_proxy_result):
    """TS neurons must produce spikes during the test window."""
    r = baseline_proxy_result
    ts_t = np.asarray(r["sp_ts"].t / b2.second, dtype=float)
    t0 = float(r["train_duration_s"])
    t1 = float(r["train_duration_s"] + r["test_duration_s"])
    n_ts_test = int(np.sum((ts_t >= t0) & (ts_t < t1)))
    assert n_ts_test > 0, (
        "No TS spikes during the test window. "
        "The circuit may not be producing output."
    )


@pytest.mark.slow
def test_regression_weights_within_bounds(baseline_proxy_result):
    """All MON->TS weights must stay in [0, mon_ts_wmax]."""
    r = baseline_proxy_result
    w = r["w_after"]
    p = r["params"]
    assert np.all(w >= 0.0), f"Weights went negative: min={w.min():.6f}"
    assert np.all(w <= p.mon_ts_wmax + 1e-9), (
        f"Weights exceeded wmax={p.mon_ts_wmax}: max={w.max():.6f}"
    )


@pytest.mark.slow
def test_regression_pv_below_max_uncertainty(baseline_proxy_result):
    """
    sigma_theta must be strictly below pi.
    pi is the maximum-uncertainty fallback value — anything less means
    the PV metric ran and TS has some position selectivity.
    At 200 trials convergence is not expected; this is a code-health check.
    """
    r = baseline_proxy_result
    sigma_theta = float(r["pv_sigma_theta"])
    assert sigma_theta < np.pi, (
        f"sigma_theta={sigma_theta:.4f} == pi: PV metric returned max-uncertainty fallback. "
        "The TS map may be completely flat."
    )


@pytest.mark.slow
def test_regression_valid_fraction_nonzero(baseline_proxy_result):
    """
    valid_fraction must be > 0: at least some test time bins must have
    TS activity and a finite position estimate.
    Very lenient — this is a smoke test, not a convergence check.
    """
    r = baseline_proxy_result
    vf = float(r["pv_valid_fraction"])
    assert vf > 0.0, (
        f"valid_fraction={vf:.4f}. No valid PV estimates. "
        "The simulation may have produced no TS activity."
    )
