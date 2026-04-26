import pytest
from dataclasses import replace
from ll_stdp_brian2 import NetworkParams, apply_model_mode


@pytest.fixture
def ll_fast_params():
    """Minimal fast params: n_mon=800, n_ts=80, 40 trials, seed=0."""
    params = apply_model_mode(NetworkParams(), "ll_fast")
    return replace(params, seed=0)
