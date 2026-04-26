import pathlib
import numpy as np
import pytest
from ll_stdp_brian2 import _load_mid_checkpoint


def _write_fake_checkpoint(run_dir: pathlib.Path, trial_idx: int, mon_ts_w: np.ndarray):
    """Write a minimal mid_checkpoint.npz matching the format _save_mid_checkpoint uses."""
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
