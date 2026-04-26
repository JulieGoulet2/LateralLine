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

    std_before = float(np.std(result["w_before"]))
    std_after = float(np.std(result["w_after"]))

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
