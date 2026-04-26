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

    margin = 0.1 * fast_params.ll_body_length_cm
    assert x_trace.min() >= 0.0
    assert x_trace.max() <= fast_params.ll_body_length_cm
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
