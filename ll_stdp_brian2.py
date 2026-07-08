"""
Lateral-line somatotopic map formation via spike-timing-dependent plasticity (STDP).

Three-layer network:
  LL  (lateral-line afferents, n=100): Poisson neurons driven by hydrodynamic velocity
        snapshots. Each neuron encodes one neuromast receptor on the fish body axis.
  MON (medial octavolateral nucleus, n=3200): leaky integrate-and-fire (LIF) neurons.
        Receive sparse LL input (fixed weight, or optionally plastic LL→MON STDP).
        MON cells develop individual position selectivity but not a clean map;
        they form a high-dimensional population code for stimulus position.
  TS  (torus semicircularis, n=300): LIF neurons. Receive mixed random + weakly
        topographic MON input. MON→TS synapses are plastic (multiplicative STDP +
        homeostatic normalization). Over ~10k training trials the TS layer forms a
        somatotopic map: each TS neuron becomes selective for a stimulus position.

Training: sphere visits positions in ordered sweeps (0..4 cm, back-and-forth),
          held 50 ms each. Each trial lasts trial_duration_s.
Test:     sphere moves in a continuous sweep at fixed speed. STDP is frozen.
          The Population Vector (PV) metric evaluates map quality; sigma_theta
          measures somatotopic error (lower = better map).

Entry points:
  run_spatial_two_stage_model(params)  — run one simulation, return result dict
  main()                               — CLI wrapper; reads args, saves figures + JSON
"""
import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

# Allow `from stimulus import ...` when the cwd is not the project directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

from stimulus import StimulusParams, hydrodynamic_velocity_parallel, simulate_lateral_line

# Keep codegen simple/portable.
b2.prefs.codegen.target = "numpy"


# Network parameters and named presets live in params.py (pure Python, no Brian2).
# Re-exported here so existing callers and tests that do
# `from ll_stdp_brian2 import NetworkParams, apply_model_mode` keep working.
from params import NetworkParams, apply_model_mode


# ---------------------------------------------------------
# Test-phase evaluation window helpers — live in plots/_helpers.py
# to break the circular import with the plots/ package.
# ---------------------------------------------------------
from plots._helpers import _eval_window_cm, _test_x_local_bins


# ---------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------
def build_mon_to_ts_indices(
    n_mon: int,
    n_ts: int,
    out_degree: int,
    sigma_ts: float,
    topography_strength: float,
    seed: int,
):
    """
    Build MON->TS indices with mixed random + weak topographic targets.

    topography_strength:
      0.0 -> fully random
      1.0 -> fully weak-somatotopic
    """
    rng = np.random.default_rng(seed)
    mon_pos = np.linspace(0.0, 1.0, n_mon)
    i_list: list[int] = []
    j_list: list[int] = []

    topo_strength = float(np.clip(topography_strength, 0.0, 1.0))

    for mon_idx in range(n_mon):
        n_topo = int(round(out_degree * topo_strength))
        n_rand = out_degree - n_topo

        parts: list[np.ndarray] = []

        # Topographic part.
        if n_topo > 0:
            if sigma_ts >= n_ts:
                topo_targets = rng.integers(0, n_ts, size=n_topo)
            else:
                center = mon_pos[mon_idx] * (n_ts - 1)
                topo_targets = np.rint(rng.normal(center, sigma_ts, size=n_topo)).astype(int)
                topo_targets = np.clip(topo_targets, 0, n_ts - 1)
            parts.append(topo_targets)

        # Random part.
        if n_rand > 0:
            parts.append(rng.integers(0, n_ts, size=n_rand))

        targets = np.concatenate(parts)
        rng.shuffle(targets)

        i_list.extend([mon_idx] * out_degree)
        j_list.extend(targets.tolist())

    return np.asarray(i_list), np.asarray(j_list), mon_pos


def build_ll_to_mon_indices(
    n_ll: int,
    n_mon: int,
    in_degree: int,
    sigma_ll: float,
    topography_strength: float,
    seed: int,
):
    """
    Build LL->MON indices with mixed random + weak topographic sources.

    topography_strength:
      0.0 -> fully random
      1.0 -> fully weak-somatotopic

    Topographic Gaussian draws use the LL ring (period n_ll): rounded samples are reduced
    mod n_ll, matching periodic distance on equispaced receptors on [0, L] (see
    periodic_line_distance_abs in stimulus.py for the continuous analogue).
    """
    rng = np.random.default_rng(seed)
    mon_pos = np.linspace(0.0, 1.0, n_mon)
    i_list: list[int] = []
    j_list: list[int] = []

    topo_strength = float(np.clip(topography_strength, 0.0, 1.0))

    for mon_idx in range(n_mon):
        n_topo = int(round(in_degree * topo_strength))
        n_rand = in_degree - n_topo
        parts: list[np.ndarray] = []

        if n_topo > 0:
            if sigma_ll >= n_ll:
                topo_sources = rng.integers(0, n_ll, size=n_topo)
            else:
                center = mon_pos[mon_idx] * (n_ll - 1)
                raw = rng.normal(center, sigma_ll, size=n_topo)
                # NOTE: indices that fall outside [0, n_ll-1] are CLIPPED to the
                # nearest edge — NOT wrapped around modulo n_ll. This piles a
                # few extra inputs onto the edge neuromasts (LL index 0 and
                # n_ll-1) for MON cells whose preferred body position is near
                # the head or tail. The effect is small relative to the random
                # connection part and is biologically plausible (the fish body
                # is linear, not periodic). If a ring-wrap topology is ever
                # desired use `topo_sources = topo_sources % n_ll` instead.
                topo_sources = np.clip(np.rint(raw).astype(np.int64), 0, n_ll - 1).astype(int)
            parts.append(topo_sources)

        if n_rand > 0:
            parts.append(rng.integers(0, n_ll, size=n_rand))

        sources = np.concatenate(parts)
        rng.shuffle(sources)
        i_list.extend(sources.tolist())
        j_list.extend([mon_idx] * in_degree)

    return np.asarray(i_list), np.asarray(j_list)


# ---------------------------------------------------------
# Stimulus helpers
# ---------------------------------------------------------
def _build_spatial_noise_cholesky(xi_cm: np.ndarray, stim_params: StimulusParams):
    """Spatial correlation for LL noise."""
    dist = xi_cm[:, None] - xi_cm[None, :]
    corr = np.exp(-0.5 * (dist / stim_params.l_noise_cm) ** 2)
    return np.linalg.cholesky(corr + 1e-12 * np.eye(xi_cm.size))


def _sample_instantaneous_rates(
    rng: np.random.Generator,
    params: NetworkParams,
    stim_params: StimulusParams,
    xi_cm: np.ndarray,
    yi_cm: np.ndarray,
    noise_chol: np.ndarray,
    x_cm: float | None = None,
    noise_scale: float = 1.0,
    fixed_distance_cm: float | None = None,
    fixed_direction: float | None = None,
    distance_min_cm: float | None = None,
    distance_max_cm: float | None = None,
):
    """One random hydrodynamic snapshot -> LL rates.

    Distance sampling rule (per call):
      1. If ``fixed_distance_cm`` is set -> use that exact value.
      2. Else if both ``distance_min_cm`` and ``distance_max_cm`` are set
         AND ``min < max`` -> sample d ~ Uniform([min, max]).  This is the
         multi-distance training mode (added 2026-05-08).
      3. Else -> d ~ Normal(mu_distance_cm, sigma_distance_cm), then clamp
         to [min, max] if either bound is provided.  This is the original
         behaviour and is preserved for all single-distance baselines (where
         the user passes min == max, e.g. 0.8, 0.8).
    """
    if fixed_distance_cm is not None:
        d_cm = float(fixed_distance_cm)
    elif (
        distance_min_cm is not None
        and distance_max_cm is not None
        and float(distance_min_cm) < float(distance_max_cm)
    ):
        # Multi-distance training: uniform over the requested range so each
        # distance contributes equally to STDP weight updates.
        d_cm = float(rng.uniform(float(distance_min_cm), float(distance_max_cm)))
    else:
        d_cm = float(rng.normal(stim_params.mu_distance_cm, stim_params.sigma_distance_cm))
        if distance_min_cm is not None:
            d_cm = max(float(distance_min_cm), d_cm)
        if distance_max_cm is not None:
            d_cm = min(float(distance_max_cm), d_cm)
    if fixed_direction is None:
        direction = 1.0 if rng.random() < 0.5 else -1.0
    else:
        direction = float(np.sign(fixed_direction)) if fixed_direction != 0 else 1.0
    if x_cm is None:
        x_cm = float(rng.uniform(-1.0, stim_params.lateral_line_length_cm + 1.0))
    else:
        x_cm = float(x_cm)

    v = hydrodynamic_velocity_parallel(
        xi_cm,
        yi_cm,
        X_cm=x_cm,
        Y_cm=d_cm,
        U_cm_s=params.speed_cm_s,
        R_cm=stim_params.sphere_radius_cm,
        eX=direction,
        eY=0.0,
        sx=1.0,
        sy=0.0,
    )

    eta = (noise_scale * stim_params.sigma_noise_hz) * (noise_chol @ rng.standard_normal(params.n_ll))
    rates = np.clip(stim_params.r0_hz + stim_params.A_per_cm * v + eta, 0.0, stim_params.rmax_hz)
    return rates, {"X_cm": x_cm, "D_cm": d_cm, "direction": direction}


def make_training_rates(params: NetworkParams):
    """Balanced-position snapshot training stream for all training trials."""
    rng = np.random.default_rng(params.seed)
    stim_params = StimulusParams(lateral_line_length_cm=params.ll_body_length_cm, sphere_radius_cm=params.sphere_radius_cm)

    xi_cm = np.linspace(0.0, stim_params.lateral_line_length_cm, params.n_ll)
    yi_cm = np.zeros_like(xi_cm)
    noise_chol = _build_spatial_noise_cholesky(xi_cm, stim_params)

    total_train_s = params.n_training_trials * params.trial_duration_s
    n_steps = int(np.round(total_train_s / params.dt_s))
    if n_steps <= 0:
        return np.zeros((0, params.n_ll), dtype=float), [], np.zeros(0, dtype=float)

    hold_steps = max(1, int(np.round(params.training_position_hold_s / params.dt_s)))

    rates = np.zeros((n_steps, params.n_ll), dtype=float)
    x_trace_cm = np.zeros(n_steps, dtype=float)
    samples = []

    # Balanced x sampling across the full 4 cm lateral line.
    x_bins = np.linspace(0.0, stim_params.lateral_line_length_cm, params.n_ll)
    x_seq = []
    if params.training_ordered_sweeps:
        # Deterministic forward/backward sweeps reduce ambiguity and improve map learning.
        while len(x_seq) * hold_steps < n_steps:
            x_seq.extend(x_bins.tolist())
            x_seq.extend(x_bins[::-1].tolist())
    else:
        while len(x_seq) * hold_steps < n_steps:
            xb = x_bins.copy()
            rng.shuffle(xb)
            x_seq.extend(xb.tolist())

    idx = 0
    k = 0
    while idx < n_steps:
        frac = idx / max(1, n_steps - 1)
        noise_scale = (
            params.training_noise_scale_early
            if frac < params.training_noise_switch_fraction
            else params.training_noise_scale_late
        )
        inst_rates, meta = _sample_instantaneous_rates(
            rng,
            params,
            stim_params,
            xi_cm,
            yi_cm,
            noise_chol,
            x_cm=x_seq[k],
            noise_scale=noise_scale,
            fixed_distance_cm=(stim_params.mu_distance_cm if params.training_fixed_distance else None),
            fixed_direction=(1.0 if (params.training_bidirectional is False or (k % 2 == 0)) else -1.0),
            distance_min_cm=params.training_distance_min_cm,
            distance_max_cm=params.training_distance_max_cm,
        )
        nxt = min(n_steps, idx + hold_steps)
        rates[idx:nxt, :] = inst_rates[None, :]
        x_trace_cm[idx:nxt] = x_seq[k]
        meta["t_start_s"] = idx * params.dt_s
        meta["t_end_s"] = nxt * params.dt_s
        samples.append(meta)
        idx = nxt
        k += 1

    return rates, samples, x_trace_cm


def make_test_rates(params: NetworkParams):
    """Continuous moving-sphere test sweep.

    For the lateral line (0..4 cm), we sometimes want the sphere center to
    traverse a longer path (e.g. -0.5..4.5 cm) so the endpoints are not
    over-emphasized by geometry.
    """
    test_duration_s = params.test_path_cm / max(params.speed_cm_s, 1e-9)
    # Start 0.5 cm before the lateral line; sphere moves forward along the test path.
    stim_params = StimulusParams(lateral_line_length_cm=params.ll_body_length_cm, sphere_radius_cm=params.sphere_radius_cm)
    x0_cm = -0.5 if params.direction >= 0 else (stim_params.lateral_line_length_cm + 0.5)
    sim = simulate_lateral_line(
        duration_s=test_duration_s,
        dt_s=params.dt_s,
        n_neuromasts=params.n_ll,
        seed=params.seed + 999,
        fixed_distance_cm=params.distance_cm,
        direction=params.direction,
        fixed_speed_cm_s=params.speed_cm_s,
        x0_cm=x0_cm,
        params=stim_params,
    )
    # Physical stimulus position X_cm stays linear (no toroidal wrap); use --eval-x-min/--eval-x-max for analysis window.
    return sim


def make_test_rates_held_snapshots(params: NetworkParams):
    """
    Same total duration as make_test_rates (test_path_cm / speed), but LL drive is built from
    held snapshots sampled like training positions (x_bins sweeps / shuffle, hold, noise curriculum).
    """
    rng = np.random.default_rng(params.seed + 999)
    stim_params = StimulusParams(lateral_line_length_cm=params.ll_body_length_cm, sphere_radius_cm=params.sphere_radius_cm)
    xi_cm = np.linspace(0.0, stim_params.lateral_line_length_cm, params.n_ll)
    yi_cm = np.zeros_like(xi_cm)
    noise_chol = _build_spatial_noise_cholesky(xi_cm, stim_params)

    test_duration_s = params.test_path_cm / max(params.speed_cm_s, 1e-9)
    t = np.arange(0.0, test_duration_s, params.dt_s)
    n_steps = int(t.size)
    hold_steps = max(1, int(np.round(params.training_position_hold_s / params.dt_s)))

    rates = np.zeros((n_steps, params.n_ll), dtype=float)
    x_trace_cm = np.zeros(n_steps, dtype=float)
    y_trace_cm = np.zeros(n_steps, dtype=float)

    x_bins = np.linspace(0.0, stim_params.lateral_line_length_cm, params.n_ll)
    x_seq: list[float] = []
    if params.training_ordered_sweeps:
        while len(x_seq) * hold_steps < n_steps:
            x_seq.extend(x_bins.tolist())
            x_seq.extend(x_bins[::-1].tolist())
    else:
        while len(x_seq) * hold_steps < n_steps:
            xb = x_bins.copy()
            rng.shuffle(xb)
            x_seq.extend(xb.tolist())

    idx = 0
    k = 0
    while idx < n_steps:
        frac = idx / max(1, n_steps - 1)
        noise_scale = (
            params.training_noise_scale_early
            if frac < params.training_noise_switch_fraction
            else params.training_noise_scale_late
        )
        inst_rates, meta = _sample_instantaneous_rates(
            rng,
            params,
            stim_params,
            xi_cm,
            yi_cm,
            noise_chol,
            x_cm=x_seq[k],
            noise_scale=noise_scale,
            fixed_distance_cm=(stim_params.mu_distance_cm if params.training_fixed_distance else None),
            fixed_direction=(1.0 if (params.training_bidirectional is False or (k % 2 == 0)) else -1.0),
            distance_min_cm=params.training_distance_min_cm,
            distance_max_cm=params.training_distance_max_cm,
        )
        nxt = min(n_steps, idx + hold_steps)
        rates[idx:nxt, :] = inst_rates[None, :]
        x_trace_cm[idx:nxt] = float(meta["X_cm"])
        y_trace_cm[idx:nxt] = meta["D_cm"]
        idx = nxt
        k += 1

    return {
        "t_s": t,
        "xi_cm": xi_cm,
        "X_cm": x_trace_cm,
        "Y_cm": y_trace_cm,
        "U_cm_s": float(params.speed_cm_s),
        "D_cm": float(np.mean(y_trace_cm)) if n_steps else 0.0,
        "direction": float(params.direction),
        "rates_hz": rates,
    }


# ---------------------------------------------------------
# Map metric + stabilization metric
# ---------------------------------------------------------
# The map-quality metric works in a CIRCULAR coordinate: the linear body axis (0..L cm)
# is mapped onto an angle theta in [0, 2*pi) (see the population-vector decode below),
# so decoding error is an angular error and must be summarised with directional
# (circular) statistics rather than ordinary mean/std.
def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi] so error differences don't jump at the 0/2pi seam."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _circ_mean(x: np.ndarray) -> float:
    """Circular mean = angle of the summed unit vectors (atan2 of mean sin, mean cos)."""
    if x.size == 0:
        return np.nan
    return float(np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x))))


def _circ_std(x: np.ndarray) -> float:
    """Mardia circular standard deviation sqrt(-2 ln R), where R is the mean resultant
    length (R=1 -> perfectly concentrated -> std 0; R->0 -> uniform -> std -> inf).
    This is the `sigma_theta` map-error reported throughout RESULTS.md."""
    if x.size == 0:
        return np.nan
    c = np.mean(np.cos(x))
    s = np.mean(np.sin(x))
    r = np.sqrt(c * c + s * s)          # mean resultant length R in [0, 1]
    r = float(np.clip(r, 1e-12, 1.0))
    return float(np.sqrt(-2.0 * np.log(r)))


def _tuning_fwhm_cm(
    spike_t_s: np.ndarray,
    spike_i: np.ndarray,
    n_neurons: int,
    test_t_s: np.ndarray,
    test_x_cm: np.ndarray,
    train_start_s: float,
    n_bins: int = 50,
    min_peak_hz: float = 1.0,
) -> tuple[float, float, int]:
    """
    Compute mean FWHM (full-width at half-maximum) tuning width in cm.

    Returns (mean_fwhm_cm, sd_fwhm_cm, n_valid_neurons).
    Neurons whose peak firing rate < min_peak_hz are excluded.
    """
    if len(test_t_s) < 2:
        return float("nan"), float("nan"), 0

    x_min, x_max = float(test_x_cm.min()), float(test_x_cm.max())
    if x_max <= x_min:
        return float("nan"), float("nan"), 0

    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_width_cm = bin_edges[1] - bin_edges[0]
    dt_s = float(test_t_s[1] - test_t_s[0])

    # Time spent in each x-bin (seconds).
    x_bin_idx = np.clip(np.digitize(test_x_cm, bin_edges) - 1, 0, n_bins - 1)
    time_in_bin = np.bincount(x_bin_idx, minlength=n_bins) * dt_s

    # Map each test-phase spike to its x-bin.
    t_rel = spike_t_s - train_start_s
    valid = t_rel >= 0.0
    t_rel = t_rel[valid]
    sp_i = spike_i[valid]

    t_idx = np.clip(np.floor(t_rel / dt_s).astype(int), 0, len(test_x_cm) - 1)
    sp_x_bin = x_bin_idx[t_idx]

    counts = np.zeros((n_neurons, n_bins), dtype=float)
    mask = (sp_i >= 0) & (sp_i < n_neurons)
    np.add.at(counts, (sp_i[mask], sp_x_bin[mask]), 1.0)

    safe_time = np.maximum(time_in_bin, 1e-9)
    rates = counts / safe_time[np.newaxis, :]  # (n_neurons, n_bins), Hz

    fwhms = []
    for n in range(n_neurons):
        r = rates[n]
        peak = float(r.max())
        if peak < min_peak_hz:
            continue
        half = peak / 2.0
        above = r >= half
        if not np.any(above):
            continue
        # Width = total number of bins above half-max × bin width.
        fwhm_cm = float(np.count_nonzero(above)) * bin_width_cm
        fwhms.append(fwhm_cm)

    if len(fwhms) < 2:
        return float("nan"), float("nan"), len(fwhms)
    arr = np.array(fwhms)
    return float(arr.mean()), float(arr.std(ddof=1)), len(fwhms)


def pv_map_quality_from_ts_spikes(
    ts_spike_t_s: np.ndarray,
    ts_spike_i: np.ndarray,
    n_ts: int,
    test_t_s: np.ndarray,
    test_x_cm: np.ndarray,
    lateral_line_len_cm: float,
    test_start_s: float,
    dt_s: float,
    n_pos_bins: int = 100,
    eval_x_min_cm: float | None = None,
    eval_x_max_cm: float | None = None,
):
    """
    Population-vector map quality following thesis §4.1.4 ideas:
    - theta_hat(t) from PV direction
    - delta_theta(t) = theta_hat - theta_true (wrapped)
    - bias(theta0), trial variability sigma_trial(theta0)
    - somatotopic error sigma_theta = dispersion of bias over theta0

    If eval_x_min_cm and eval_x_max_cm are set, only time bins whose linear test_x_cm lies in
    [eval_x_min_cm, eval_x_max_cm] contribute (others zeroed before smoothing); theta_true maps
    that window linearly to [0, 2pi). Otherwise theta_true maps the full test x trace linearly
    to [0, 2pi) using min/max of test_x_cm.
    lateral_line_len_cm is unused when eval window is set; kept for API compatibility.
    """
    n_t = test_t_s.size
    rates = np.zeros((n_t, n_ts), dtype=float)

    # Bin TS spikes into test-time bins.
    t_rel = ts_spike_t_s - test_start_s
    k = np.floor(t_rel / dt_s).astype(int)
    valid = (k >= 0) & (k < n_t) & (ts_spike_i >= 0) & (ts_spike_i < n_ts)
    if np.any(valid):
        np.add.at(rates, (k[valid], ts_spike_i[valid]), 1.0 / dt_s)

    use_win = eval_x_min_cm is not None and eval_x_max_cm is not None
    if use_win:
        emin = float(eval_x_min_cm)
        emax = float(eval_x_max_cm)
        ok = (test_x_cm >= emin) & (test_x_cm <= emax)
        rates[~ok, :] = 0.0
    else:
        ok = np.ones(n_t, dtype=bool)

    # Temporal smoothing improves PV robustness when activity is sparse in single ms bins.
    smooth_win = max(1, int(round(0.02 / max(dt_s, 1e-9))))  # ~20 ms
    if smooth_win > 1:
        kernel = np.ones(smooth_win, dtype=float) / float(smooth_win)
        rates = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, rates)

    # --- Population-vector (PV) decode of stimulus position ---
    # Assign TS neuron k a "preferred direction" phi_k = 2*pi*k/n_ts, i.e. lay the
    # 300 TS cells evenly around a ring. (We use a ring, not a line, purely as a decode
    # convenience: it lets us read out position as the angle of a population vector and
    # summarise error with circular statistics; it does NOT imply the fish body is a ring.)
    phi = 2.0 * np.pi * np.arange(n_ts, dtype=float) / float(n_ts)
    ejphi = np.exp(1j * phi)
    denom = np.sum(rates, axis=1)
    z = np.zeros(n_t, dtype=np.complex128)
    nz = denom > 1e-12
    if np.any(nz):
        # Population vector: rate-weighted sum of unit vectors, normalised. Its angle is
        # the decoded position theta_hat; positions with no spikes are left undecoded.
        z[nz] = (rates[nz] @ ejphi) / denom[nz]

    theta_hat = np.zeros(n_t, dtype=float)
    theta_hat[nz] = np.mod(np.angle(z[nz]), 2.0 * np.pi)
    theta_hat[~nz] = np.nan

    theta_true = np.full(n_t, np.nan, dtype=float)
    if use_win:
        wspan = float(emax - emin)
        theta_true[ok] = 2.0 * np.pi * (test_x_cm[ok] - emin) / max(wspan, 1e-12)
    else:
        span = float(np.ptp(test_x_cm)) + 1e-12
        theta_true[:] = 2.0 * np.pi * (test_x_cm - float(np.min(test_x_cm))) / span

    # Estimation error.
    delta = np.full(n_t, np.nan, dtype=float)
    m = np.isfinite(theta_hat) & np.isfinite(theta_true)
    delta[m] = _wrap_to_pi(theta_hat[m] - theta_true[m])

    # Position-dependent bias and trial variability.
    edges = np.linspace(0.0, 2.0 * np.pi, n_pos_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bias = np.full(n_pos_bins, np.nan, dtype=float)
    sigma_trial = np.full(n_pos_bins, np.nan, dtype=float)
    for b in range(n_pos_bins):
        mb = m & (theta_true >= edges[b]) & (theta_true < edges[b + 1])
        eb = delta[mb]
        if eb.size >= 1:
            mu = _circ_mean(eb)
            bias[b] = mu
            sigma_trial[b] = _circ_std(_wrap_to_pi(eb - mu)) if eb.size >= 2 else 0.0

    vb = np.isfinite(bias)
    vd = np.isfinite(delta) & ok
    sigma_theta = _circ_std(bias[vb]) if np.any(vb) else (_circ_std(delta[vd]) if np.any(vd) else np.pi)
    delta_trial = (
        float(np.sqrt(np.nanmean(sigma_trial**2)))
        if np.any(np.isfinite(sigma_trial))
        else (_circ_std(delta[vd]) if np.any(vd) else np.pi)
    )
    valid_fraction = float(np.mean(vd)) if vd.size > 0 else 0.0

    return {
        "theta_true": theta_true,
        "theta_hat": theta_hat,
        "delta_theta": delta,
        "theta_bins": centers,
        "bias_theta": bias,
        "sigma_trial_theta": sigma_trial,
        "sigma_theta": sigma_theta,
        "delta_trial": delta_trial,
        "valid_fraction": valid_fraction,
    }


def estimate_stabilization_time(epoch_t_s: np.ndarray, w_mean_series: np.ndarray):
    """When mean MON->TS weight changes by <1% for 4 consecutive checkpoints."""
    if epoch_t_s.size < 6:
        return None

    dw = np.abs(np.diff(w_mean_series))
    ref = max(float(np.abs(w_mean_series[-1])), 1e-12)
    rel_dw = dw / ref

    # Use a stricter 0.5% threshold so 'stabilized' really means very slow change.
    threshold = 0.005
    window = 4
    for i in range(window - 1, rel_dw.size):
        if np.all(rel_dw[i - window + 1 : i + 1] < threshold):
            return float(epoch_t_s[i + 1])
    return None


# ---------------------------------------------------------
# Mid-run checkpoint helpers
# ---------------------------------------------------------
def _save_mid_checkpoint(ckpt_path, actual_trial_idx, s_mon_ts, ll_mon_use_stdp, s_ll_mon, stats):
    data = {"trial_idx": np.array(actual_trial_idx)}
    data["mon_ts_w"] = np.array(s_mon_ts.w[:], dtype=float, copy=True)
    if ll_mon_use_stdp:
        data["ll_mon_w_mV"] = np.array(s_ll_mon.w[:] / b2.mV, dtype=float, copy=True)
    for key, val in stats.items():
        data[key] = np.asarray(val)
    # Delete the old file first — on macOS, overwriting a file that carries the
    # com.apple.provenance extended attribute raises PermissionError (errno 1).
    # Writing to a fresh .npz path then renaming avoids that security check.
    # NOTE: np.savez_compressed appends .npz if the path doesn't end in .npz,
    # so the temp name must already end in .npz.
    tmp_path = ckpt_path.parent / ("_tmp_" + ckpt_path.name)  # _tmp_mid_checkpoint.npz
    if tmp_path.exists():
        tmp_path.unlink()
    np.savez_compressed(str(tmp_path), **data)
    if ckpt_path.exists():
        ckpt_path.unlink()
    tmp_path.rename(ckpt_path)
    print(f"[checkpoint] saved trial {actual_trial_idx} → {ckpt_path.name}", flush=True)


def _load_mid_checkpoint(run_dir):
    path = Path(run_dir) / "artifacts" / "mid_checkpoint.npz"
    if not path.exists():
        return None
    raw = np.load(str(path), allow_pickle=False)
    ckpt = {k: raw[k] for k in raw.files}
    print(f"[checkpoint] resuming from trial {int(ckpt['trial_idx'])}", flush=True)
    return ckpt


# ---------------------------------------------------------
# Single-training pipeline (no temporal phase split)
# ---------------------------------------------------------
def run_spatial_two_stage_model(params: NetworkParams, checkpoint_path=None, resume_checkpoint=None):
    """Build, train, and test the full LL -> MON -> TS network for one seed.

    This is the master pipeline. Phases, in order:
      1. Build the stimulus streams (training snapshots + continuous test sweep).
      2. Build the Brian2 network: Poisson LL input, LIF MON and TS layers, LL->MON
         and MON->TS synapses (with STDP), global + lateral inhibition, background.
      3. Train: run the STDP-plastic network over n_training_trials, checkpointing
         weights every checkpoint_trials.
      4. Freeze plasticity and run the continuous test sweep.
      5. Compute the population-vector (PV) map-quality metrics (sigma_theta, etc.)
         and save figures + a result dict.

    checkpoint_path : where to write mid_checkpoint.npz during training (crash safety).
    resume_checkpoint : optional dict of restored weights. EXTRACT MODE = a checkpoint
        whose trial_idx already equals n_training_trials-1, i.e. training is complete;
        the training stimulus is then skipped entirely and only the test phase runs
        (this is how run_distance_sweep_extract.sh / run_stimvar_extract.sh re-test
        trained weights under new stimuli without retraining).
    Returns a `result` dict consumed by the plotting helpers and the summary JSON.
    """
    # Decide up front whether ANY training trials remain. Extract-mode runs
    # have resume_checkpoint["trial_idx"] == n_training_trials - 1 (all
    # trials already done), and previously we still allocated the full
    # ~9.6 GB training stimulus only to slice it down to 0. Skip the call
    # entirely in that case.
    extract_only = (
        resume_checkpoint is not None
        and int(resume_checkpoint.get("trial_idx", -1)) + 1 >= params.n_training_trials
    )

    if extract_only:
        train_rates = np.zeros((0, params.n_ll), dtype=float)
        train_samples: list[dict] = []
        train_x_cm = np.zeros(0, dtype=float)
    else:
        train_rates, train_samples, train_x_cm = make_training_rates(params)

    test_sim = make_test_rates_held_snapshots(params) if bool(params.test_using_held_snapshots) else make_test_rates(params)
    test_rates = test_sim["rates_hz"]

    # Iris-Hydi-style test-phase noise: add Gaussian noise (SD = test_ll_noise_hz) to LL rates.
    # Independent per-neuron per-timestep; clipped to [0, +inf). seed offset 8888 keeps reproducibility.
    if float(params.test_ll_noise_hz) > 0.0:
        _noise_rng = np.random.default_rng(int(params.seed) + 8888)
        _noise = _noise_rng.normal(0.0, float(params.test_ll_noise_hz), size=test_rates.shape)
        test_rates = np.clip(test_rates + _noise, 0.0, None)

    # If resuming, drop already-completed trials from the stimulus.
    trial_steps = int(np.round(params.trial_duration_s / params.dt_s))
    k_start = 0
    if resume_checkpoint is not None:
        k_start = int(resume_checkpoint["trial_idx"]) + 1
        if k_start > params.n_training_trials:
            raise ValueError(
                f"Checkpoint trial {k_start - 1} > n_training_trials {params.n_training_trials}; nothing left to run."
            )
        skip_steps = k_start * trial_steps
        train_rates = train_rates[skip_steps:]
        train_x_cm = train_x_cm[skip_steps:]
        print(f"[resume] skipping {k_start} completed trials, {params.n_training_trials - k_start} remaining.", flush=True)

    remaining_training_trials = params.n_training_trials - k_start

    # Concatenate (remaining) training + test in one TimedArray.
    rates_all = np.vstack([train_rates, test_rates])
    baseline_subtract_hz = float(max(0.0, params.ll_rate_baseline_subtract_hz))
    if params.ll_rate_mode == "modulation":
        baseline_subtract_hz += float(StimulusParams().r0_hz)
    elif params.ll_rate_mode != "raw":
        raise ValueError(f"Unknown ll_rate_mode '{params.ll_rate_mode}'. Use 'raw' or 'modulation'.")
    rates_all = np.clip((rates_all - baseline_subtract_hz) * float(max(0.0, params.ll_rate_gain)), 0.0, None)

    train_duration_s = remaining_training_trials * params.trial_duration_s
    test_duration_s = float(test_sim["t_s"][-1] + params.dt_s)
    total_duration_s = train_duration_s + test_duration_s

    # Resets Brian2's global state (neuron groups, synapses, network objects) so
    # sequential calls to this function don't accumulate stale objects.
    b2.start_scope()
    # Seed Brian2's internal RNG used by PoissonGroup spike sampling and the
    # `rand()` calls in synapse/state-variable initialisers. WITHOUT this call,
    # two runs with the same --seed-start give different spike trains and
    # different final weights — only the numpy-side RNG (used for connectivity
    # and weight jitter) was being seeded. Offset 12345 keeps it distinct from
    # other per-seed RNG offsets (+999 test sweep, +8888 noise, etc.).
    b2.seed(int(params.seed) + 12345)
    b2.defaultclock.dt = params.dt_s * b2.second

    all_rates_ta = b2.TimedArray(rates_all * b2.Hz, dt=params.dt_s * b2.second)
    ll = b2.PoissonGroup(params.n_ll, rates="all_rates_ta(t, i)", namespace={"all_rates_ta": all_rates_ta})

    # LIF equations shared by MON, TS, and both inhibitory interneurons.
    # v: membrane potential (volt); ge/gi: excitatory/inhibitory PSP contributions (volt).
    # Each spike onto a post-synaptic neuron increments ge (or gi) by the synaptic weight w.
    # ge and gi decay back to 0 with tau_s; the membrane leaks toward El with tau_m.
    eqs = """
    dv/dt = (El - v + ge - gi) / tau_m : volt (unless refractory)
    dge/dt = -ge / tau_s : volt
    dgi/dt = -gi / tau_s : volt
    """
    ns = {
        "Vth": params.vth_mV * b2.mV,
        "Vreset": params.vreset_mV * b2.mV,
        "El": params.el_mV * b2.mV,
        "tau_ref": params.tau_ref_ms * b2.ms,
        "tau_m": params.tau_m_ms * b2.ms,
        "tau_s": params.tau_s_ms * b2.ms,
    }

    mon = b2.NeuronGroup(params.n_mon, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)
    ts = b2.NeuronGroup(params.n_ts, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)

    mon_inh = b2.NeuronGroup(1, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)
    ts_inh = b2.NeuronGroup(1, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)

    mon.v = "El + rand()*2*mV"
    ts.v = "El + rand()*2*mV"
    mon_inh.v = "El"
    ts_inh.v = "El"

    # LL -> MON: random sparse + optional weak-topographic, optionally plastic.
    if not params.ll_mon_use_stdp:
        # Fixed weights (original behavior).
        s_ll_mon = b2.Synapses(ll, mon, "w : volt", on_pre="ge_post += w")
        if params.ll_to_mon_topography_strength <= 0.0:
            s_ll_mon.connect(p=params.p_ll_to_mon)
        else:
            ll_i_topo, mon_j_topo = build_ll_to_mon_indices(
                n_ll=params.n_ll,
                n_mon=params.n_mon,
                in_degree=params.ll_to_mon_in_degree,
                sigma_ll=params.ll_to_mon_sigma,
                topography_strength=params.ll_to_mon_topography_strength,
                seed=params.seed + 31,
            )
            s_ll_mon.connect(i=ll_i_topo, j=mon_j_topo)
        s_ll_mon.w = f"{params.ll_mon_w_mean_mV}*mV + {params.ll_mon_w_jitter_mV}*mV*rand()"
    else:
        # LL->MON with STDP (Hebbian plastic expansion).
        if params.ll_to_mon_topography_strength <= 0.0:
            ll_i_topo = mon_j_topo = None
        else:
            ll_i_topo, mon_j_topo = build_ll_to_mon_indices(
                n_ll=params.n_ll,
                n_mon=params.n_mon,
                in_degree=params.ll_to_mon_in_degree,
                sigma_ll=params.ll_to_mon_sigma,
                topography_strength=params.ll_to_mon_topography_strength,
                seed=params.seed + 31,
            )
        s_ll_mon = b2.Synapses(
            ll,
            mon,
            # Pair-based STDP with two exponential eligibility traces:
            #   apre  jumps by Apre  on each PREsynaptic spike, decays with taupre;
            #   apost jumps by Apost on each POSTsynaptic spike, decays with taupost.
            # taupre = taupost = 20 ms sets the +/-20 ms coincidence window over which
            # pre-before-post (potentiation) and post-before-pre (depression) are counted.
            model="""
            w : volt
            dapre/dt = -apre/taupre : 1 (event-driven)
            dapost/dt = -apost/taupost : 1 (event-driven)
            """,
            # On a PREsynaptic spike: deliver the EPSP (ge_post += w), then depress by the
            # amount of recent POST activity (apost). MULTIPLICATIVE (w + apost*w): the
            # depression scales with w, a soft lower bound. Apost < 0, so this is LTD.
            on_pre="""
            ge_post += w
            apre += Apre
            w = clip(w + apost*w, 0*mV, wmax)
            """,
            # On a POSTsynaptic spike: potentiate by recent PRE activity (apre), scaled by
            # the room left to wmax (w + apre*(wmax-w)) — soft upper bound. This
            # multiplicative rule is REQUIRED here; additive STDP saturates all weights and
            # makes every MON fire everywhere (see CLAUDE.md / project_stdp_rules memory).
            on_post="""
            apost += Apost
            w = clip(w + apre*(wmax - w), 0*mV, wmax)
            """,
            namespace={
                "taupre": 20 * b2.ms,   # STDP potentiation-window time constant
                "taupost": 20 * b2.ms,  # STDP depression-window time constant
                "Apre": params.ll_mon_apre,    # LTP step on the pre-trace (>0)
                "Apost": params.ll_mon_apost,  # LTD step on the post-trace (<0)
                "wmax": params.ll_mon_wmax_mV * b2.mV,
            },
        )
        if params.ll_to_mon_topography_strength <= 0.0:
            s_ll_mon.connect(p=params.p_ll_to_mon)
        else:
            s_ll_mon.connect(i=ll_i_topo, j=mon_j_topo)
        s_ll_mon.w = f"{params.ll_mon_w_init_mV}*mV + {params.ll_mon_w_jitter_stdp_mV}*mV*rand()"
        # Baseline incoming sum per MON neuron (target for homeostasis).
        w_ll_init_mV = np.array(s_ll_mon.w[:] / b2.mV, dtype=float, copy=True)
        j_ll_conn = np.array(s_ll_mon.j[:], dtype=int, copy=True)
        incoming0 = np.bincount(j_ll_conn, weights=w_ll_init_mV, minlength=params.n_mon)
        ll_mon_homeo_target = float(np.mean(incoming0[incoming0 > 0])) if np.any(incoming0 > 0) else float(
            np.mean(incoming0)
        )
    # MON -> TS: random+weak-topographic, plastic (STDP).
    mon_i, ts_j, _ = build_mon_to_ts_indices(
        n_mon=params.n_mon,
        n_ts=params.n_ts,
        out_degree=params.mon_to_ts_out_degree,
        sigma_ts=params.mon_to_ts_sigma,
        topography_strength=params.mon_to_ts_topography_strength,
        seed=params.seed,
    )

    s_mon_ts = b2.Synapses(
        mon,
        ts,
        model="""
        w : 1
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        """,
        # NOTE: MON->TS STDP is ADDITIVE (Δw does NOT scale with w), unlike the
        # LL->MON synapse above which uses multiplicative STDP. The "additive
        # STDP saturates weights" issue documented in
        # ~/.claude/projects/.../memory/project_stdp_rules.md is real, but here
        # it is compensated by the slow MON->TS multiplicative homeostasis
        # (--mon-ts-homeo-eta 0.001): every 10 trials the incoming weight sum
        # per TS cell is rescaled toward its initial target, which prevents
        # the runaway saturation that pure additive STDP would otherwise show.
        # Changing this to multiplicative would require re-tuning the whole
        # recipe (apre/apost amplitudes, homeo eta) and is a SCIENTIFIC change,
        # not a bug fix. Keep additive + homeostasis as the current design.
        on_pre="""
        ge_post += mon_ts_gain*w*mV
        apre += Apre
        w = clip(w + apost, 0, wmax)
        """,
        on_post="""
        apost += Apost
        w = clip(w + apre, 0, wmax)
        """,
        namespace={
            "taupre": 20 * b2.ms,
            "taupost": 20 * b2.ms,
            "Apre": params.mon_ts_apre,
            "Apost": params.mon_ts_apost,
            "wmax": params.mon_ts_wmax,
            "mon_ts_gain": params.mon_ts_gain_mV,
        },
    )
    s_mon_ts.connect(i=mon_i, j=ts_j)
    s_mon_ts.w = f"{params.mon_ts_w_init} + {params.mon_ts_w_jitter}*rand()"

    # Background noisy drive.
    bg_mon = b2.PoissonGroup(params.n_mon, rates=params.bg_rate_mon_hz * b2.Hz)
    bg_ts = b2.PoissonGroup(params.n_ts, rates=params.bg_rate_ts_hz * b2.Hz)
    s_bg_mon = b2.Synapses(bg_mon, mon, on_pre=f"ge_post += {params.bg_w_mon_mV}*mV")
    s_bg_mon.connect(condition="i == j")
    s_bg_ts = b2.Synapses(bg_ts, ts, on_pre=f"ge_post += {params.bg_w_ts_mV}*mV")
    s_bg_ts.connect(condition="i == j")

    # MON global inhibition.
    s_mon_to_inh = b2.Synapses(mon, mon_inh, on_pre=f"ge_post += {params.mon_to_global_inh_drive_mV}*mV")
    s_mon_to_inh.connect(p=params.mon_to_global_inh_p)
    s_inh_to_mon = b2.Synapses(mon_inh, mon, "w : volt", on_pre="gi_post += w")
    s_inh_to_mon.connect()
    s_inh_to_mon.w = params.global_inh_to_mon_mV * b2.mV

    # TS local lateral inhibition (toroidal: TS 0 and n_ts-1 are neighbors so no edge advantage).
    src, dst, wlat = [], [], []
    radius = params.ts_lateral_radius
    n_ts = params.n_ts
    for ts_src in range(n_ts):
        for ts_dst in range(n_ts):
            if ts_src == ts_dst:
                continue
            ring_dist = abs(ts_src - ts_dst)
            ring_dist = min(ring_dist, n_ts - ring_dist)
            if ring_dist > radius:
                continue
            src.append(ts_src)
            dst.append(ts_dst)
            wlat.append(params.ts_local_inh_peak_mV * (1.0 - ring_dist / radius))

    s_ts_lat = b2.Synapses(ts, ts, "w : volt", on_pre="gi_post += w")
    s_ts_lat.connect(i=np.asarray(src), j=np.asarray(dst))
    wlat_base_mV = np.asarray(wlat, dtype=float)
    s_ts_lat.w = wlat_base_mV * b2.mV

    # Optional TS feedback inhibition (global): TS -> inhibitory cell -> TS.
    if bool(params.use_ts_feedback_inh):
        s_ts_to_inh = b2.Synapses(ts, ts_inh, on_pre=f"ge_post += {params.ts_to_global_inh_drive_mV}*mV")
        s_ts_to_inh.connect(p=float(np.clip(params.ts_to_global_inh_p, 0.0, 1.0)))
        s_inh_to_ts = b2.Synapses(ts_inh, ts, "w : volt", on_pre="gi_post += w")
        s_inh_to_ts.connect()
        s_inh_to_ts.w = float(max(0.0, params.global_inh_to_ts_mV)) * b2.mV

    # Monitors.
    sp_ll = b2.SpikeMonitor(ll)
    sp_mon = b2.SpikeMonitor(mon)
    sp_ts = b2.SpikeMonitor(ts)
    pr_mon = b2.PopulationRateMonitor(mon)
    pr_ts = b2.PopulationRateMonitor(ts)

    # Save fixed LL->MON weights in mV for map metric.
    ll_i = np.array(s_ll_mon.i[:], dtype=int, copy=True)
    ll_j = np.array(s_ll_mon.j[:], dtype=int, copy=True)
    ll_w_mV = np.array(s_ll_mon.w[:] / b2.mV, dtype=float, copy=True)

    # Restore weights from checkpoint if resuming.
    if resume_checkpoint is not None:
        print("[resume] restoring weights from checkpoint ...", flush=True)
        s_mon_ts.w = resume_checkpoint["mon_ts_w"]
        if params.ll_mon_use_stdp and "ll_mon_w_mV" in resume_checkpoint:
            s_ll_mon.w = resume_checkpoint["ll_mon_w_mV"] * b2.mV
        print("[resume] weights restored.", flush=True)

    # Save MON->TS initial state (after any weight restore).
    w_before = np.array(s_mon_ts.w[:], dtype=float, copy=True)

    # Checkpoint records — prepopulate from saved state if resuming.
    rng = np.random.default_rng(params.seed + 777)
    tracked_n = min(24, w_before.size)
    tracked_idx = rng.choice(w_before.size, size=tracked_n, replace=False)

    if resume_checkpoint is not None and "checkpoint_t_s" in resume_checkpoint:
        checkpoint_t_s = list(resume_checkpoint["checkpoint_t_s"])
        w_mean_series = list(resume_checkpoint["w_mean_series"])
        w_std_series = list(resume_checkpoint["w_std_series"])
        tracked_weight_series = list(resume_checkpoint["tracked_weight_series"])
        w_mean_abs_delta_series = list(resume_checkpoint["w_mean_abs_delta_series"])
        w_frac_delta_gt_1e3_series = list(resume_checkpoint["w_frac_delta_gt_1e3_series"])
        ts_ckpt_rate_series = list(resume_checkpoint.get("ts_ckpt_rate_series", np.array([])))
        pv_ckpt_t_s = list(resume_checkpoint.get("pv_ckpt_t_s", np.array([])))
        pv_sigma_theta_series = list(resume_checkpoint.get("pv_sigma_theta_series", np.array([])))
        pv_delta_trial_series = list(resume_checkpoint.get("pv_delta_trial_series", np.array([])))
    else:
        checkpoint_t_s = [0.0]
        w_mean_series = [float(np.mean(w_before))]
        w_std_series = [float(np.std(w_before))]
        tracked_weight_series = [w_before[tracked_idx].copy()]
        w_mean_abs_delta_series = [0.0]
        w_frac_delta_gt_1e3_series = [0.0]
        ts_ckpt_rate_series = []
        pv_ckpt_t_s = []
        pv_sigma_theta_series = []
        pv_delta_trial_series = []

    last_ckpt_t = 0.0
    last_ts_spike_count = 0

    # Each trial runs trial_duration_s of simulation. The TimedArray advances the LL
    # stimulus automatically. Both LL→MON (if enabled) and MON→TS STDP are active.
    # Every checkpoint_trials, weight statistics are recorded and optionally written to disk
    # so a crashed run can be resumed from mid_checkpoint.npz via --resume-from.
    for k in range(remaining_training_trials):
        b2.run(params.trial_duration_s * b2.second)

        if (k + 1) % max(1, params.checkpoint_trials) == 0 or (k + 1) == remaining_training_trials:
            wk = np.array(s_mon_ts.w[:], dtype=float, copy=True)
            t_cur = (k + 1) * params.trial_duration_s

            # Checkpoint TS activity during training.
            dt_ckpt = max(1e-9, t_cur - last_ckpt_t)
            new_ts_spikes = sp_ts.num_spikes - last_ts_spike_count
            ts_rate_ckpt_hz = new_ts_spikes / max(1e-9, params.n_ts * dt_ckpt)

            ts_ckpt_rate_series.append(ts_rate_ckpt_hz)

            # PV quality over training: evaluate on the recent training window.
            ckpt_idx = len(ts_ckpt_rate_series)
            if ckpt_idx % max(1, params.pv_eval_every_checkpoints) == 0:
                i0 = int(np.clip(np.floor(last_ckpt_t / params.dt_s), 0, train_x_cm.size - 1))
                i1 = int(np.clip(np.floor(t_cur / params.dt_s), i0 + 1, train_x_cm.size))
                x_win = train_x_cm[i0:i1]
                t_win = np.arange(x_win.size, dtype=float) * params.dt_s

                sp_t = np.asarray(sp_ts.t / b2.second, dtype=float)
                sp_i = np.asarray(sp_ts.i, dtype=int)
                msp = (sp_t >= last_ckpt_t) & (sp_t < t_cur)
                pv_ck = pv_map_quality_from_ts_spikes(
                    ts_spike_t_s=sp_t[msp],
                    ts_spike_i=sp_i[msp],
                    n_ts=params.n_ts,
                    test_t_s=t_win,
                    test_x_cm=x_win,
                    lateral_line_len_cm=float(params.ll_body_length_cm),
                    test_start_s=last_ckpt_t,
                    dt_s=params.dt_s,
                    n_pos_bins=min(100, params.n_ts),
                    eval_x_min_cm=None,
                    eval_x_max_cm=None,
                )
                pv_ckpt_t_s.append(t_cur)
                pv_sigma_theta_series.append(pv_ck["sigma_theta"])
                pv_delta_trial_series.append(pv_ck["delta_trial"])
            last_ckpt_t = t_cur
            last_ts_spike_count = sp_ts.num_spikes

            checkpoint_t_s.append(t_cur)
            w_mean_series.append(float(np.mean(wk)))
            w_std_series.append(float(np.std(wk)))
            tracked_weight_series.append(wk[tracked_idx].copy())
            dwk = np.abs(wk - w_before)
            w_mean_abs_delta_series.append(float(np.mean(dwk)))
            w_frac_delta_gt_1e3_series.append(float(np.mean(dwk > 1e-3)))

            # Periodic weight save to disk so a crash can be resumed.
            ckpt_count = len(checkpoint_t_s) - 1  # exclude the t=0 entry
            every_n = max(1, int(params.checkpoint_save_every_n_checkpoints))
            if checkpoint_path is not None and ckpt_count > 0 and ckpt_count % every_n == 0:
                actual_trial = k + k_start
                stats_snap = {
                    "checkpoint_t_s": np.asarray(checkpoint_t_s),
                    "w_mean_series": np.asarray(w_mean_series),
                    "w_std_series": np.asarray(w_std_series),
                    "tracked_weight_series": np.asarray(tracked_weight_series),
                    "w_mean_abs_delta_series": np.asarray(w_mean_abs_delta_series),
                    "w_frac_delta_gt_1e3_series": np.asarray(w_frac_delta_gt_1e3_series),
                    "ts_ckpt_rate_series": np.asarray(ts_ckpt_rate_series),
                    "pv_ckpt_t_s": np.asarray(pv_ckpt_t_s),
                    "pv_sigma_theta_series": np.asarray(pv_sigma_theta_series),
                    "pv_delta_trial_series": np.asarray(pv_delta_trial_series),
                }
                _save_mid_checkpoint(checkpoint_path, actual_trial, s_mon_ts, params.ll_mon_use_stdp, s_ll_mon, stats_snap)

        # LL->MON homeostasis (if enabled) every ll_mon_homeo_every_trials.
        if params.ll_mon_use_stdp and params.ll_mon_homeo_eta > 0.0 and (
            (k + 1) % max(1, params.ll_mon_homeo_every_trials) == 0
        ):
            w_ll_arr = np.array(s_ll_mon.w[:] / b2.mV, dtype=float, copy=True)
            j_ll_conn = np.array(s_ll_mon.j[:], dtype=int, copy=True)
            incoming = np.bincount(j_ll_conn, weights=w_ll_arr, minlength=params.n_mon)
            scale = np.ones(params.n_mon, dtype=float)
            nonzero = incoming > 1e-12
            if np.any(nonzero):
                ratio = ll_mon_homeo_target / np.maximum(incoming[nonzero], 1e-12)
                scale[nonzero] = 1.0 + params.ll_mon_homeo_eta * (ratio - 1.0)
                scale = np.clip(scale, 0.9, 1.1)
                w_ll_arr = np.clip(w_ll_arr * scale[j_ll_conn], 0.0, params.ll_mon_wmax_mV)
                s_ll_mon.w = w_ll_arr * b2.mV

        # MON->TS homeostasis every mon_ts_homeo_every_trials.
        if params.mon_ts_homeo_eta > 0.0 and (k + 1) % max(1, params.mon_ts_homeo_every_trials) == 0:
            w_mts_arr = np.array(s_mon_ts.w[:], dtype=float, copy=True)
            j_mts_conn = np.array(s_mon_ts.j[:], dtype=int, copy=True)
            incoming_ts = np.bincount(j_mts_conn, weights=w_mts_arr, minlength=params.n_ts)
            scale_ts = np.ones(params.n_ts, dtype=float)
            nonzero_ts = incoming_ts > 1e-12
            if np.any(nonzero_ts):
                target_ts = float(np.mean(incoming_ts[nonzero_ts]))
                ratio_ts = target_ts / np.maximum(incoming_ts[nonzero_ts], 1e-12)
                scale_ts[nonzero_ts] = 1.0 + params.mon_ts_homeo_eta * (ratio_ts - 1.0)
                scale_ts = np.clip(scale_ts, 0.9, 1.1)
                w_mts_arr = np.clip(w_mts_arr * scale_ts[j_mts_conn], 0.0, params.mon_ts_wmax)
                s_mon_ts.w = w_mts_arr

    # Freeze STDP by zeroing Apre/Apost so synaptic traces still update but weights don't change.
    # This ensures test spikes reflect the map learned during training, not continued learning.
    if not bool(params.keep_mon_ts_stdp_during_test):
        s_mon_ts.namespace["Apre"] = 0.0
        s_mon_ts.namespace["Apost"] = 0.0

    # Run the test sweep. The sphere moves continuously at fixed speed; the SpikeMonitor
    # records all TS spikes which are later passed to pv_map_quality_from_ts_spikes().
    b2.run(test_duration_s * b2.second, report="text")

    w_after = np.array(s_mon_ts.w[:], dtype=float, copy=True)
    dw = w_after - w_before
    abs_dw = np.abs(dw)

    pvq = pv_map_quality_from_ts_spikes(
        ts_spike_t_s=np.asarray(sp_ts.t / b2.second, dtype=float),
        ts_spike_i=np.asarray(sp_ts.i, dtype=int),
        n_ts=params.n_ts,
        test_t_s=np.asarray(test_sim["t_s"], dtype=float),
        test_x_cm=np.asarray(test_sim["X_cm"], dtype=float),
        lateral_line_len_cm=float(params.ll_body_length_cm),
        test_start_s=train_duration_s,
        dt_s=params.dt_s,
        n_pos_bins=min(100, params.n_ts),
        eval_x_min_cm=params.eval_x_min_cm,
        eval_x_max_cm=params.eval_x_max_cm,
    )

    stab_t = estimate_stabilization_time(np.asarray(checkpoint_t_s), np.asarray(w_mean_series))

    # Tuning widths (FWHM in cm) for LL, MON, and TS during the test sweep.
    _ll_t = np.asarray(sp_ll.t / b2.second, dtype=float)
    _ll_i = np.asarray(sp_ll.i, dtype=int)
    _mon_t = np.asarray(sp_mon.t / b2.second, dtype=float)
    _mon_i = np.asarray(sp_mon.i, dtype=int)
    _ts_t = np.asarray(sp_ts.t / b2.second, dtype=float)
    _ts_i = np.asarray(sp_ts.i, dtype=int)
    _t_s = np.asarray(test_sim["t_s"], dtype=float)
    _x_cm = np.asarray(test_sim["X_cm"], dtype=float)

    tw_ll_mean, tw_ll_sd, tw_ll_n = _tuning_fwhm_cm(
        _ll_t, _ll_i, params.n_ll, _t_s, _x_cm, train_duration_s)
    tw_mon_mean, tw_mon_sd, tw_mon_n = _tuning_fwhm_cm(
        _mon_t, _mon_i, params.n_mon, _t_s, _x_cm, train_duration_s)
    tw_ts_mean, tw_ts_sd, tw_ts_n = _tuning_fwhm_cm(
        _ts_t, _ts_i, params.n_ts, _t_s, _x_cm, train_duration_s)

    # LL population-vector quality — gives delta_trial for the afferent layer.
    pvq_ll = pv_map_quality_from_ts_spikes(
        ts_spike_t_s=_ll_t,
        ts_spike_i=_ll_i,
        n_ts=params.n_ll,
        test_t_s=_t_s,
        test_x_cm=_x_cm,
        lateral_line_len_cm=float(params.ll_body_length_cm),
        test_start_s=train_duration_s,
        dt_s=params.dt_s,
        n_pos_bins=min(100, params.n_ll),
        eval_x_min_cm=params.eval_x_min_cm,
        eval_x_max_cm=params.eval_x_max_cm,
    )

    ll_t = np.asarray(sp_ll.t / b2.second, dtype=float)
    mon_t = np.asarray(sp_mon.t / b2.second, dtype=float)
    ts_t_abs = np.asarray(sp_ts.t / b2.second, dtype=float)
    t_tr = float(train_duration_s)
    t_te = float(train_duration_s + test_duration_s)
    d_te = float(test_duration_s)
    n_ll = int(params.n_ll)
    n_mon = int(params.n_mon)
    m_ll_tr = (ll_t >= 0.0) & (ll_t < t_tr)
    m_ll_te = (ll_t >= t_tr) & (ll_t < t_te)
    m_mon_tr = (mon_t >= 0.0) & (mon_t < t_tr)
    m_mon_te = (mon_t >= t_tr) & (mon_t < t_te)
    m_ts_tr = (ts_t_abs >= 0.0) & (ts_t_abs < t_tr)
    m_ts_te = (ts_t_abs >= t_tr) & (ts_t_abs < t_te)
    ll_rate_tr_hz = float(np.count_nonzero(m_ll_tr)) / max(1e-300, n_ll * t_tr)
    ll_rate_te_hz = float(np.count_nonzero(m_ll_te)) / max(1e-300, n_ll * d_te)
    mon_rate_tr_hz = float(np.count_nonzero(m_mon_tr)) / max(1e-300, n_mon * t_tr)
    mon_rate_te_hz = float(np.count_nonzero(m_mon_te)) / max(1e-300, n_mon * d_te)
    n_ts_spikes_tr = int(np.count_nonzero(m_ts_tr))
    n_ts_spikes_te = int(np.count_nonzero(m_ts_te))
    print(f"[debug] mean LL rate (train): {ll_rate_tr_hz:.6g} Hz")
    print(f"[debug] mean LL rate (test): {ll_rate_te_hz:.6g} Hz")
    print(f"[debug] mean MON rate (train): {mon_rate_tr_hz:.6g} Hz")
    print(f"[debug] mean MON rate (test): {mon_rate_te_hz:.6g} Hz")
    print(f"[debug] total TS spikes (train): {n_ts_spikes_tr}")
    print(f"[debug] total TS spikes (test): {n_ts_spikes_te}")

    # Result dict consumed by save_learning_artifacts(), all save_*_figure() functions,
    # and the tests. Key fields: sp_ts (SpikeMonitor for PV analysis), w_before/w_after
    # (MON→TS weights), pv_sigma_theta (map quality), pv_valid_fraction (fraction of test
    # bins with TS activity), checkpoint_t_s / w_mean_series (training learning curves).
    return {
        "params": params,
        "train_samples": train_samples,
        "train_x_cm": np.asarray(train_x_cm, dtype=float),
        "test_sim": test_sim,
        "sp_ll": sp_ll,
        "sp_mon": sp_mon,
        "sp_ts": sp_ts,
        "pr_mon": pr_mon,
        "pr_ts": pr_ts,
        "w_before": w_before,
        "w_after": w_after,
        "ll_to_mon_i": ll_i,
        "ll_to_mon_j": ll_j,
        "ll_to_mon_w_mV": ll_w_mV,
        "mon_to_ts_i": np.array(s_mon_ts.i[:], dtype=int, copy=True),
        "mon_to_ts_j": np.array(s_mon_ts.j[:], dtype=int, copy=True),
        "checkpoint_t_s": np.asarray(checkpoint_t_s),
        "w_mean_series": np.asarray(w_mean_series),
        "w_std_series": np.asarray(w_std_series),
        "tracked_weight_series": np.asarray(tracked_weight_series),
        "w_mean_abs_delta_series": np.asarray(w_mean_abs_delta_series),
        "w_frac_delta_gt_1e3_series": np.asarray(w_frac_delta_gt_1e3_series),
        "w_mean_abs_delta_final": float(np.mean(abs_dw)),
        "w_frac_delta_gt_1e3_final": float(np.mean(abs_dw > 1e-3)),
        "w_frac_at_wmax_final": float(np.mean(np.isclose(w_after, params.mon_ts_wmax))),
        "w_frac_at_zero_final": float(np.mean(np.isclose(w_after, 0.0))),
        "ts_ckpt_rate_series_hz": np.asarray(ts_ckpt_rate_series),
        "pv_ckpt_t_s": np.asarray(pv_ckpt_t_s),
        "pv_sigma_theta_series": np.asarray(pv_sigma_theta_series),
        "pv_delta_trial_series": np.asarray(pv_delta_trial_series),
        "stabilization_time_s": stab_t,
        "pv_theta_true": pvq["theta_true"],
        "pv_theta_hat": pvq["theta_hat"],
        "pv_delta_theta": pvq["delta_theta"],
        "pv_theta_bins": pvq["theta_bins"],
        "pv_bias_theta": pvq["bias_theta"],
        "pv_sigma_trial_theta": pvq["sigma_trial_theta"],
        "pv_sigma_theta": pvq["sigma_theta"],
        "pv_delta_trial": pvq["delta_trial"],
        "pv_valid_fraction": pvq["valid_fraction"],
        "pv_ll_sigma_theta": pvq_ll["sigma_theta"],
        "pv_ll_delta_trial": pvq_ll["delta_trial"],
        "tuning_fwhm_ll_cm_mean": tw_ll_mean,
        "tuning_fwhm_ll_cm_sd": tw_ll_sd,
        "tuning_fwhm_ll_n_valid": tw_ll_n,
        "tuning_fwhm_mon_cm_mean": tw_mon_mean,
        "tuning_fwhm_mon_cm_sd": tw_mon_sd,
        "tuning_fwhm_mon_n_valid": tw_mon_n,
        "tuning_fwhm_ts_cm_mean": tw_ts_mean,
        "tuning_fwhm_ts_cm_sd": tw_ts_sd,
        "tuning_fwhm_ts_n_valid": tw_ts_n,
        "train_duration_s": train_duration_s,
        "test_duration_s": test_duration_s,
        "total_duration_s": total_duration_s,
    }


def save_learning_artifacts(result: dict, out_dir: Path, tag: str):
    """
    Save learned weights and connectivity so future tests can reuse training.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / tag
    weights_path = base.with_suffix(".npz")
    params_path = base.with_name(base.name + "_params.json")

    np.savez_compressed(
        weights_path,
        mon_ts_w=result["w_after"],
        mon_ts_i=result["mon_to_ts_i"],
        mon_ts_j=result["mon_to_ts_j"],
        ll_mon_i=result["ll_to_mon_i"],
        ll_mon_j=result["ll_to_mon_j"],
        ll_mon_w_mV=result["ll_to_mon_w_mV"],
    )

    p = result["params"]
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "ll_thesis",
                "n_ll": p.n_ll,
                "n_mon": p.n_mon,
                "n_ts": p.n_ts,
                "speed_cm_s": p.speed_cm_s,
                "distance_cm": p.distance_cm,
                "direction": p.direction,
                "training_noise_scale_early": p.training_noise_scale_early,
                "training_noise_scale_late": p.training_noise_scale_late,
                "mon_ts_apre": p.mon_ts_apre,
                "mon_ts_apost": p.mon_ts_apost,
                "mon_ts_wmax": p.mon_ts_wmax,
            },
            f,
            indent=2,
        )
    return weights_path, params_path


# ---------------------------------------------------------
# Plotting — functions live in plots/ (grouped by similarity) to keep this
# file focused on simulation logic. Imported here so callers that do
# `from ll_stdp_brian2 import save_summary_figure` continue to work.
# ---------------------------------------------------------
from plots import (
    save_summary_figure,
    save_test_phase_only_figure,
    save_ts_pop_rate_train_test_transition_figure,
    save_ts_tuning_figure,
    save_mon_tuning_examples_figure,
    save_mon_to_ts_weight_profile,
    save_mon_to_ts_receptive_fields_figure,
    save_ll_mon_weights_figure,
    save_ts_spikes_vs_x_test_figure,
    save_mon_spikes_vs_x_test_figure,
    save_ll_spikes_vs_x_test_figure,
    save_mon_ts_feedforward_drive_figures,
    save_multiseed_summary,
    save_learning_curves_figure,
)


def main():
    # CLI workflow: choose a parameter preset (--mode), then optionally override individual
    # parameters with their specific flags. Run one seed or many (--multi-seed).
    # All outputs go under Runs/<run-name>/: figures/, artifacts/, params.json.
    parser = argparse.ArgumentParser(description="Lateral-line MON/TS STDP model (Brian2).")
    parser.add_argument(
        "--mode",
        type=str,
        default="ll_thesis",
        choices=["ll_thesis", "ll_fast"],
        help="Parameter preset. Default is strict lateral-line thesis mode.",
    )
    parser.add_argument(
        "--save-weights-dir",
        type=str,
        default="SavedModels/ll_thesis_no_noise",
        help="Folder where learned weights/connectivity snapshots are stored.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run under Runs/<name>/ (default: timestamp). Figures and artifacts go in subfolders.",
    )
    parser.add_argument(
        "--save-tag",
        type=str,
        default="latest",
        help="Filename tag for saved artifacts (without extension).",
    )
    parser.add_argument(
        "--multi-seed",
        type=int,
        default=1,
        help="Number of seeds to run and summarize (1 = single run).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=123,
        help="Starting seed for multi-seed runs.",
    )
    parser.add_argument(
        "--n-ll",
        type=int,
        default=None,
        help="Override number of LL (afferent) neurons. Scale proportionally with --ll-body-length-cm (e.g. 175 for 7 cm body).",
    )
    parser.add_argument(
        "--n-training-trials",
        type=int,
        default=None,
        help="Override number of training trials (default from --mode).",
    )
    parser.add_argument(
        "--distance-cm",
        type=float,
        default=None,
        help="Override test distance Y (cm) used in make_test_rates().",
    )
    parser.add_argument(
        "--training-distance-min-cm",
        type=float,
        default=None,
        help=(
            "Override minimum training distance (cm) for snapshot sampling. "
            "When used together with --training-distance-max-cm and min < max, "
            "distance is sampled UNIFORMLY on [min, max] each trial "
            "(multi-distance training, since 2026-05-08). "
            "When min == max, distance is fixed at that value."
        ),
    )
    parser.add_argument(
        "--training-distance-max-cm",
        type=float,
        default=None,
        help=(
            "Override maximum training distance (cm) for snapshot sampling. "
            "See --training-distance-min-cm for the sampling rule."
        ),
    )
    parser.add_argument(
        "--ll-body-length-cm",
        type=float,
        default=None,
        help="Physical length of the lateral line / fish body axis (cm). Default 4.0. Set to e.g. 7.0 to reduce boundary effects; use --eval-x-* to evaluate only the middle.",
    )
    parser.add_argument(
        "--test-path-cm",
        type=float,
        default=None,
        help="Length of linear test sweep along x (cm); use > neuromast span (e.g. 7) with --eval-x-* for center window.",
    )
    parser.add_argument(
        "--eval-x-min",
        type=float,
        default=None,
        dest="eval_x_min_cm",
        help="With --eval-x-max: restrict PV and test x-plots to physical x in [min, max] (cm, linear).",
    )
    parser.add_argument(
        "--eval-x-max",
        type=float,
        default=None,
        dest="eval_x_max_cm",
        help="With --eval-x-min: upper bound (cm) of evaluation window.",
    )
    parser.add_argument(
        "--ll-mon-topo",
        type=float,
        default=None,
        help="Override LL->MON weak topography strength.",
    )
    parser.add_argument(
        "--mon-ts-topo",
        type=float,
        default=None,
        help="Override MON->TS weak topography strength.",
    )
    parser.add_argument(
        "--ll-rate-mode",
        type=str,
        choices=["raw", "modulation"],
        default=None,
        help="Use raw LL rates or only modulation above spontaneous baseline.",
    )
    parser.add_argument(
        "--ll-baseline-subtract-hz",
        type=float,
        default=None,
        help="Additional LL baseline subtraction before Poisson drive (Hz).",
    )
    parser.add_argument(
        "--ll-rate-gain",
        type=float,
        default=None,
        help="Multiplicative gain on LL rates before Poisson drive.",
    )
    parser.add_argument(
        "--use-ll-mon-stdp",
        action="store_true",
        help="Enable STDP on LL->MON synapses (plastic expansion layer).",
    )
    parser.add_argument(
        "--ll-mon-apre",
        type=float,
        default=None,
        help="Override LL->MON Apre (pre-synaptic STDP increment).",
    )
    parser.add_argument(
        "--ll-mon-apost",
        type=float,
        default=None,
        help="Override LL->MON Apost (post-synaptic STDP increment).",
    )
    parser.add_argument(
        "--ll-mon-wmax-mv",
        type=float,
        default=None,
        help="Override LL->MON maximal weight (mV).",
    )
    parser.add_argument(
        "--ll-mon-w-init-mv",
        type=float,
        default=None,
        help="Override LL->MON initial mean weight (mV) when STDP is enabled.",
    )
    parser.add_argument(
        "--ll-mon-w-jitter-stdp-mv",
        type=float,
        default=None,
        help="Override LL->MON initial weight jitter (mV) when STDP is enabled.",
    )
    parser.add_argument(
        "--ll-mon-in-degree",
        type=int,
        default=None,
        help="Override LL->MON in-degree (number of LL inputs each MON receives, when topo > 0).",
    )
    parser.add_argument(
        "--ll-mon-homeo-eta",
        type=float,
        default=None,
        help="Override LL->MON homeostatic scaling strength (per incoming-weight target).",
    )
    parser.add_argument(
        "--ll-mon-homeo-every-trials",
        type=int,
        default=None,
        help="Override LL->MON homeostasis period in training trials (every N trials).",
    )
    parser.add_argument(
        "--mon-ts-homeo-eta",
        type=float,
        default=None,
        help="Override MON->TS homeostatic scaling strength (per incoming-weight target).",
    )
    parser.add_argument(
        "--mon-ts-sigma",
        type=float,
        default=None,
        help="Override MON->TS weak-topography spread (TS index units).",
    )
    parser.add_argument(
        "--mon-ts-out-degree",
        type=int,
        default=None,
        help="Override MON->TS outgoing synapses per MON neuron.",
    )
    parser.add_argument(
        "--ts-lateral-radius",
        type=int,
        default=None,
        help="Override TS local lateral inhibition radius (in TS neuron indices).",
    )
    parser.add_argument(
        "--ts-local-inh-peak-mv",
        type=float,
        default=None,
        help="Override TS local lateral inhibition peak strength (mV).",
    )
    parser.add_argument(
        "--use-ts-feedback-inh",
        action="store_true",
        help="Enable TS activity-dependent feedback inhibition (global TS inhibition).",
    )
    parser.add_argument(
        "--ts-feedback-drive-mv",
        type=float,
        default=None,
        help="Override TS->TS-inh drive strength (mV) when feedback inhibition is enabled.",
    )
    parser.add_argument(
        "--ts-feedback-inh-mv",
        type=float,
        default=None,
        help="Override TS-inh->TS inhibition strength (mV) when feedback inhibition is enabled.",
    )
    parser.add_argument(
        "--ts-feedback-p",
        type=float,
        default=None,
        help="Override TS->TS-inh connection probability (0..1) when feedback inhibition is enabled.",
    )
    parser.add_argument(
        "--mon-global-inh-mv",
        type=float,
        default=None,
        help="Override MON global inhibition strength onto MON (mV).",
    )
    parser.add_argument(
        "--mon-ts-gain-mv",
        type=float,
        default=None,
        help="Override MON->TS EPSP gain (mV per unit weight).",
    )
    parser.add_argument(
        "--bg-rate-mon-hz",
        type=float,
        default=None,
        help="Override MON background Poisson drive rate (Hz).",
    )
    parser.add_argument(
        "--bg-rate-ts-hz",
        type=float,
        default=None,
        help="Override TS background Poisson drive rate (Hz).",
    )
    parser.add_argument(
        "--bg-w-ts-mv",
        type=float,
        default=None,
        help="Override TS background synaptic weight (mV).",
    )
    parser.add_argument(
        "--keep-mon-ts-stdp-during-test",
        action="store_true",
        help="If set, do not zero MON->TS STDP (Apre/Apost) before the test phase.",
    )
    parser.add_argument(
        "--test-using-held-snapshots",
        action="store_true",
        help="If set, test phase uses held snapshots (same position sampling as training) for the same duration as the default continuous test.",
    )
    parser.add_argument(
        "--disable-all-stdp",
        action="store_true",
        help="No-learning control: set LL->MON and MON->TS STDP (Apre/Apost) to zero.",
    )
    parser.add_argument(
        "--eval-x-min-cm",
        type=float,
        default=None,
        help="Start of evaluation window along the fish body axis (cm). Must be set together with --eval-x-max-cm.",
    )
    parser.add_argument(
        "--eval-x-max-cm",
        type=float,
        default=None,
        help="End of evaluation window along the fish body axis (cm). Must be set together with --eval-x-min-cm.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to a run directory to resume from (e.g. Runs/stdp_topo02_with_20260426_082031). "
            "Loads artifacts/mid_checkpoint.npz and continues training from the saved trial."
        ),
    )
    # --- Network architecture ---
    parser.add_argument("--n-mon", type=int, default=None, help="Override number of MON neurons.")
    parser.add_argument("--n-ts", type=int, default=None, help="Override number of TS neurons.")
    # --- Simulation timestep and trial structure ---
    parser.add_argument("--dt-s", type=float, default=None, help="Override simulation integration timestep (seconds).")
    parser.add_argument("--trial-duration-s", type=float, default=None, help="Override duration of each training trial (seconds).")
    parser.add_argument("--checkpoint-trials", type=int, default=None, help="Record weight statistics every N training trials.")
    parser.add_argument("--pv-eval-every-checkpoints", type=int, default=None, help="Evaluate PV map quality every N checkpoints.")
    parser.add_argument("--checkpoint-save-every-n", type=int, default=None, help="Flush mid_checkpoint.npz to disk every N checkpoints.")
    parser.add_argument("--training-position-hold-s", type=float, default=None, help="Duration each snapshot position is held during training (s).")
    parser.add_argument("--training-noise-early", type=float, default=None, help="LL noise std during early training (fraction of sigma_noise).")
    parser.add_argument("--training-noise-late", type=float, default=None, help="LL noise std during late training (fraction of sigma_noise).")
    parser.add_argument("--training-noise-switch", type=float, default=None, help="Training fraction where noise transitions early→late (0..1).")
    parser.add_argument("--test-ll-noise-hz", type=float, default=None, help="Gaussian noise (Hz, SD) added to LL rates during test phase only.")
    parser.add_argument("--training-ordered-sweeps", action=argparse.BooleanOptionalAction, default=None, help="Forward/backward sweeps (True) or random shuffle (False).")
    parser.add_argument("--training-fixed-distance", action=argparse.BooleanOptionalAction, default=None, help="Fix sphere distance at mu_distance_cm during training (no jitter).")
    parser.add_argument("--training-bidirectional", action=argparse.BooleanOptionalAction, default=None, help="Alternate sphere direction each snapshot.")
    # --- Stimulus ---
    parser.add_argument("--speed-cm-s", type=float, default=None, help="Sphere speed in cm/s; sets test duration via test_path_cm / speed.")
    parser.add_argument("--direction", type=float, default=None, help="Sphere travel direction (+1 = forward, -1 = backward).")
    parser.add_argument("--sphere-radius-cm", type=float, default=None, help="Radius of the moving sphere (cm); default 0.5. Scales dipole source strength.")
    # --- LIF neuron constants ---
    parser.add_argument("--vth-mv", type=float, default=None, help="Override LIF spike threshold (mV).")
    parser.add_argument("--vreset-mv", type=float, default=None, help="Override LIF reset potential after spike (mV).")
    parser.add_argument("--el-mv", type=float, default=None, help="Override LIF leak/resting potential (mV).")
    parser.add_argument("--tau-ref-ms", type=float, default=None, help="Override absolute refractory period (ms).")
    parser.add_argument("--tau-m-ms", type=float, default=None, help="Override membrane time constant (ms).")
    parser.add_argument("--tau-s-ms", type=float, default=None, help="Override synaptic PSP decay time constant (ms).")
    # --- LL→MON connectivity ---
    parser.add_argument("--p-ll-to-mon", type=float, default=None, help="LL→MON connection probability (used when topo == 0).")
    parser.add_argument("--ll-mon-sigma", type=float, default=None, help="Override LL→MON topographic Gaussian spread (LL-index units).")
    parser.add_argument("--ll-mon-w-mean-mv", type=float, default=None, help="Override LL→MON mean fixed weight (mV).")
    parser.add_argument("--ll-mon-w-jitter-mv", type=float, default=None, help="Override LL→MON fixed-weight jitter (mV).")
    # --- MON→TS STDP ---
    parser.add_argument("--mon-ts-apre", type=float, default=None, help="Override MON→TS STDP pre-synaptic increment (LTP amplitude).")
    parser.add_argument("--mon-ts-apost", type=float, default=None, help="Override MON→TS STDP post-synaptic increment (LTD amplitude, negative).")
    parser.add_argument("--mon-ts-wmax", type=float, default=None, help="Override MON→TS maximum dimensionless weight.")
    parser.add_argument("--mon-ts-w-init", type=float, default=None, help="Override MON→TS initial mean weight.")
    parser.add_argument("--mon-ts-w-jitter", type=float, default=None, help="Override MON→TS initial weight jitter.")
    parser.add_argument("--mon-ts-homeo-every-trials", type=int, default=None, help="Override MON→TS homeostasis period (every N training trials).")
    # --- Inhibition ---
    parser.add_argument("--mon-global-inh-p", type=float, default=None, help="Override MON→inh-cell connection probability.")
    parser.add_argument("--mon-global-inh-drive-mv", type=float, default=None, help="Override MON→inh-cell EPSP drive (mV per spike).")
    # --- Background ---
    parser.add_argument("--bg-w-mon-mv", type=float, default=None, help="Override MON background synapse weight (mV).")
    args = parser.parse_args()
    if (args.eval_x_min_cm is not None) ^ (args.eval_x_max_cm is not None):
        parser.error("--eval-x-min and --eval-x-max must be given together")
    if args.eval_x_min_cm is not None and float(args.eval_x_max_cm) <= float(args.eval_x_min_cm):
        parser.error("--eval-x-max must be greater than --eval-x-min")

    params = apply_model_mode(NetworkParams(), args.mode)
    override = {"seed": args.seed_start}

    # Scalar overrides: (args attribute, NetworkParams field, transform).
    # Each entry is applied only if the user provided the flag (args.X is not None).
    SCALAR_OVERRIDES = [
        # network sizes
        ("n_ll",                      "n_ll",                                lambda x: max(1, int(x))),
        ("n_mon",                     "n_mon",                               lambda x: max(1, int(x))),
        ("n_ts",                      "n_ts",                                lambda x: max(1, int(x))),
        # timestep / trial structure
        ("dt_s",                      "dt_s",                                lambda x: float(max(1e-6, x))),
        ("trial_duration_s",          "trial_duration_s",                    lambda x: float(max(1e-3, x))),
        ("n_training_trials",         "n_training_trials",                   lambda x: max(0, int(x))),
        ("checkpoint_trials",         "checkpoint_trials",                   lambda x: int(max(1, x))),
        ("pv_eval_every_checkpoints", "pv_eval_every_checkpoints",           lambda x: int(max(1, x))),
        ("checkpoint_save_every_n",   "checkpoint_save_every_n_checkpoints", lambda x: int(max(1, x))),
        # training schedule
        ("training_position_hold_s",  "training_position_hold_s",            lambda x: float(max(1e-4, x))),
        ("training_noise_early",      "training_noise_scale_early",          lambda x: float(max(0.0, x))),
        ("training_noise_late",       "training_noise_scale_late",           lambda x: float(max(0.0, x))),
        ("training_noise_switch",     "training_noise_switch_fraction",      lambda x: float(np.clip(x, 0.0, 1.0))),
        ("test_ll_noise_hz",          "test_ll_noise_hz",                    lambda x: float(max(0.0, x))),
        ("training_distance_min_cm",  "training_distance_min_cm",            lambda x: float(max(0.0, x))),
        ("training_distance_max_cm",  "training_distance_max_cm",            lambda x: float(max(0.0, x))),
        # stimulus
        ("speed_cm_s",                "speed_cm_s",                          lambda x: float(max(1e-6, x))),
        ("direction",                 "direction",                           lambda x: float(x)),
        ("sphere_radius_cm",          "sphere_radius_cm",                    lambda x: float(max(1e-3, x))),
        ("distance_cm",               "distance_cm",                         lambda x: float(max(1e-6, x))),
        ("ll_body_length_cm",         "ll_body_length_cm",                   lambda x: float(max(0.1, x))),
        ("test_path_cm",              "test_path_cm",                        lambda x: float(max(1e-9, x))),
        ("ll_rate_mode",              "ll_rate_mode",                        lambda x: x),
        ("ll_baseline_subtract_hz",   "ll_rate_baseline_subtract_hz",        lambda x: float(max(0.0, x))),
        ("ll_rate_gain",              "ll_rate_gain",                        lambda x: float(max(0.0, x))),
        # LIF neuron constants
        ("vth_mv",                    "vth_mV",                              lambda x: float(x)),
        ("vreset_mv",                 "vreset_mV",                           lambda x: float(x)),
        ("el_mv",                     "el_mV",                               lambda x: float(x)),
        ("tau_ref_ms",                "tau_ref_ms",                          lambda x: float(max(0.0, x))),
        ("tau_m_ms",                  "tau_m_ms",                            lambda x: float(max(0.1, x))),
        ("tau_s_ms",                  "tau_s_ms",                            lambda x: float(max(0.1, x))),
        # LL→MON connectivity / fixed weights
        ("p_ll_to_mon",               "p_ll_to_mon",                         lambda x: float(np.clip(x, 0.0, 1.0))),
        ("ll_mon_in_degree",          "ll_to_mon_in_degree",                 lambda x: int(max(1, x))),
        ("ll_mon_sigma",              "ll_to_mon_sigma",                     lambda x: float(max(1e-6, x))),
        ("ll_mon_topo",               "ll_to_mon_topography_strength",       lambda x: float(max(0.0, x))),
        ("ll_mon_w_mean_mv",          "ll_mon_w_mean_mV",                    lambda x: float(max(0.0, x))),
        ("ll_mon_w_jitter_mv",        "ll_mon_w_jitter_mV",                  lambda x: float(max(0.0, x))),
        # LL→MON STDP
        ("ll_mon_apre",               "ll_mon_apre",                         lambda x: float(x)),
        ("ll_mon_apost",              "ll_mon_apost",                        lambda x: float(x)),
        ("ll_mon_wmax_mv",            "ll_mon_wmax_mV",                      lambda x: float(max(0.0, x))),
        ("ll_mon_w_init_mv",          "ll_mon_w_init_mV",                    lambda x: float(max(0.0, x))),
        ("ll_mon_w_jitter_stdp_mv",   "ll_mon_w_jitter_stdp_mV",             lambda x: float(max(0.0, x))),
        ("ll_mon_homeo_eta",          "ll_mon_homeo_eta",                    lambda x: float(x)),
        ("ll_mon_homeo_every_trials", "ll_mon_homeo_every_trials",           lambda x: int(max(1, x))),
        # MON→TS connectivity / STDP
        ("mon_ts_topo",               "mon_to_ts_topography_strength",       lambda x: float(max(0.0, x))),
        ("mon_ts_sigma",              "mon_to_ts_sigma",                     lambda x: float(max(1e-6, x))),
        ("mon_ts_out_degree",         "mon_to_ts_out_degree",                lambda x: int(max(1, x))),
        ("mon_ts_apre",               "mon_ts_apre",                         lambda x: float(x)),
        ("mon_ts_apost",              "mon_ts_apost",                        lambda x: float(x)),
        ("mon_ts_wmax",               "mon_ts_wmax",                         lambda x: float(max(0.0, x))),
        ("mon_ts_w_init",             "mon_ts_w_init",                       lambda x: float(max(0.0, x))),
        ("mon_ts_w_jitter",           "mon_ts_w_jitter",                     lambda x: float(max(0.0, x))),
        ("mon_ts_homeo_eta",          "mon_ts_homeo_eta",                    lambda x: float(x)),
        ("mon_ts_homeo_every_trials", "mon_ts_homeo_every_trials",           lambda x: int(max(1, x))),
        ("mon_ts_gain_mv",            "mon_ts_gain_mV",                      lambda x: float(max(0.0, x))),
        # inhibition (MON + TS)
        ("mon_global_inh_p",          "mon_to_global_inh_p",                 lambda x: float(np.clip(x, 0.0, 1.0))),
        ("mon_global_inh_drive_mv",   "mon_to_global_inh_drive_mV",          lambda x: float(max(0.0, x))),
        ("mon_global_inh_mv",         "global_inh_to_mon_mV",                lambda x: float(max(0.0, x))),
        ("ts_lateral_radius",         "ts_lateral_radius",                   lambda x: int(max(1, x))),
        ("ts_local_inh_peak_mv",      "ts_local_inh_peak_mV",                lambda x: float(max(0.0, x))),
        ("ts_feedback_p",             "ts_to_global_inh_p",                  lambda x: float(np.clip(x, 0.0, 1.0))),
        ("ts_feedback_drive_mv",      "ts_to_global_inh_drive_mV",           lambda x: float(max(0.0, x))),
        ("ts_feedback_inh_mv",        "global_inh_to_ts_mV",                 lambda x: float(max(0.0, x))),
        # background drive
        ("bg_rate_mon_hz",            "bg_rate_mon_hz",                      lambda x: float(max(0.0, x))),
        ("bg_rate_ts_hz",             "bg_rate_ts_hz",                       lambda x: float(max(0.0, x))),
        ("bg_w_mon_mv",               "bg_w_mon_mV",                         lambda x: float(max(0.0, x))),
        ("bg_w_ts_mv",                "bg_w_ts_mV",                          lambda x: float(max(0.0, x))),
    ]
    for arg_attr, field, transform in SCALAR_OVERRIDES:
        val = getattr(args, arg_attr)
        if val is not None:
            override[field] = transform(val)

    # BooleanOptionalAction flags: 3-state (None / True / False); only act when set.
    BOOL_OPT_OVERRIDES = [
        ("training_ordered_sweeps", "training_ordered_sweeps"),
        ("training_fixed_distance", "training_fixed_distance"),
        ("training_bidirectional",  "training_bidirectional"),
    ]
    for arg_attr, field in BOOL_OPT_OVERRIDES:
        val = getattr(args, arg_attr)
        if val is not None:
            override[field] = bool(val)

    # store_true flags: only meaningful when True (default False).
    STORE_TRUE_OVERRIDES = [
        ("use_ll_mon_stdp",              "ll_mon_use_stdp"),
        ("use_ts_feedback_inh",          "use_ts_feedback_inh"),
        ("keep_mon_ts_stdp_during_test", "keep_mon_ts_stdp_during_test"),
        ("test_using_held_snapshots",    "test_using_held_snapshots"),
    ]
    for arg_attr, field in STORE_TRUE_OVERRIDES:
        if bool(getattr(args, arg_attr)):
            override[field] = True

    # Special: --eval-x-min and --eval-x-max are paired (validated above to be set together).
    if args.eval_x_min_cm is not None:
        override["eval_x_min_cm"] = float(args.eval_x_min_cm)
        override["eval_x_max_cm"] = float(args.eval_x_max_cm)

    # Special: --disable-all-stdp zeroes both LL→MON and MON→TS STDP increments.
    if bool(args.disable_all_stdp):
        override["ll_mon_apre"] = 0.0
        override["ll_mon_apost"] = 0.0
        override["mon_ts_apre"] = 0.0
        override["mon_ts_apost"] = 0.0

    params = replace(params, **override)

    # Load mid-run checkpoint if --resume-from was given.
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = _load_mid_checkpoint(args.resume_from)
        if resume_checkpoint is None:
            print(f"WARNING: no mid_checkpoint.npz found in {args.resume_from}/artifacts/ — starting from scratch.")

    n_runs = max(1, args.multi_seed)
    run_folder_name = (
        str(args.run_name).strip().replace("/", "_").replace("\\", "_")
        if args.run_name and str(args.run_name).strip()
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if not run_folder_name:
        run_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("Runs") / run_folder_name
    figures_dir = run_dir / "figures"
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "params.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "seed_start": args.seed_start,
                "multi_seed": n_runs,
                "run_folder": run_folder_name,
                "network_params": asdict(params),
            },
            f,
            indent=2,
        )
    print(f"Run directory: {run_dir.resolve()}")

    # Each seed gets seed_start + k so seeds are contiguous (123, 124, 125, …).
    # This keeps runs reproducible and comparable without overlap.
    results = []
    for k in range(n_runs):
        p_run = replace(params, seed=args.seed_start + k)
        r = run_spatial_two_stage_model(
            p_run,
            checkpoint_path=artifacts_dir / "mid_checkpoint.npz",
            resume_checkpoint=resume_checkpoint,
        )
        results.append(r)

        output = figures_dir / (
            f"brian2_spatial_two_stage_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_summary_figure(r, output)
        test_only_out = figures_dir / (
            f"brian2_test_phase_only_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_test_phase_only_figure(r, test_only_out)
        curves_out = figures_dir / (
            f"brian2_learning_curves_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_learning_curves_figure(r, curves_out)
        llmon_out = figures_dir / (
            f"brian2_llmon_weights_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_ll_mon_weights_figure(r, llmon_out)
        ts_tuning_out = figures_dir / (
            f"brian2_ts_tuning_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_ts_tuning_figure(r, ts_tuning_out)
        mon_ts_prof = figures_dir / (
            f"brian2_mon_ts_weight_profile_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_mon_to_ts_weight_profile(r, mon_ts_prof)
        mon_ts_rf = figures_dir / (
            f"brian2_mon_to_ts_receptive_fields_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_mon_to_ts_receptive_fields_figure(r, mon_ts_rf)
        ts_vs_x_out = figures_dir / (
            f"brian2_ts_spikes_vs_x_test_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_ts_spikes_vs_x_test_figure(r, ts_vs_x_out)
        mon_vs_x_out = figures_dir / (
            f"brian2_mon_spikes_vs_x_test_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_mon_spikes_vs_x_test_figure(r, mon_vs_x_out)
        ll_vs_x_out = figures_dir / (
            f"brian2_ll_spikes_vs_x_test_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_ll_spikes_vs_x_test_figure(r, ll_vs_x_out)
        mon_tune_ex_out = figures_dir / (
            f"brian2_mon_tuning_examples_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_mon_tuning_examples_figure(r, mon_tune_ex_out)
        mon_ts_drive_hm = figures_dir / (
            f"brian2_mon_ts_drive_heatmap_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        mon_ts_drive_win = figures_dir / (
            f"brian2_mon_ts_drive_winner_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_mon_ts_feedforward_drive_figures(r, mon_ts_drive_hm, mon_ts_drive_win)
        ts_tr_out = figures_dir / (
            f"brian2_ts_pop_rate_train_test_transition_u_{p_run.speed_cm_s:.1f}_nMON_{p_run.n_mon}_nTS_{p_run.n_ts}_seed_{p_run.seed}.png"
        )
        save_ts_pop_rate_train_test_transition_figure(r, ts_tr_out)
        weights_path, params_path = save_learning_artifacts(r, artifacts_dir, f"{args.save_tag}_seed_{p_run.seed}")
        print(f"Seed {p_run.seed} saved figure: {output}")
        print(f"Seed {p_run.seed} saved test-phase-only figure: {test_only_out}")
        print(f"Seed {p_run.seed} saved learning curves: {curves_out}")
        print(f"Seed {p_run.seed} saved LL->MON weights figure: {llmon_out}")
        print(f"Seed {p_run.seed} saved MON->TS weight profile: {mon_ts_prof}")
        print(f"Seed {p_run.seed} saved MON->TS receptive fields: {mon_ts_rf}")
        print(f"Seed {p_run.seed} saved TS spikes vs x (test): {ts_vs_x_out}")
        print(f"Seed {p_run.seed} saved MON spikes vs x (test): {mon_vs_x_out}")
        print(f"Seed {p_run.seed} saved MON tuning examples (test): {mon_tune_ex_out}")
        print(f"Seed {p_run.seed} saved MON→TS drive heatmap (test): {mon_ts_drive_hm}")
        print(f"Seed {p_run.seed} saved MON→TS drive winner (test): {mon_ts_drive_win}")
        print(f"Seed {p_run.seed} saved TS pop rate train→test: {ts_tr_out}")
        print(f"Seed {p_run.seed} saved weights: {weights_path}")
        print(f"Seed {p_run.seed} saved params: {params_path}")
        seed_results_path = artifacts_dir / f"seed_{p_run.seed}_results.json"
        with seed_results_path.open("w", encoding="utf-8") as _f:
            json.dump(
                {
                    "seed": int(p_run.seed),
                    "distance_cm": float(p_run.distance_cm),
                    "test_ll_noise_hz": float(p_run.test_ll_noise_hz),
                    "sigma_theta_rad": float(r["pv_sigma_theta"]),
                    "valid_fraction": float(r["pv_valid_fraction"]),
                    "delta_trial_rad": float(r["pv_delta_trial"]),
                    "sigma_theta_ll_rad": float(r["pv_ll_sigma_theta"]),
                    "delta_trial_ll_rad": float(r["pv_ll_delta_trial"]),
                    "sigma_w_ll_cm": float(r["tuning_fwhm_ll_cm_mean"]),
                    "sigma_w_mon_cm": float(r["tuning_fwhm_mon_cm_mean"]),
                    "sigma_w_ts_cm": float(r["tuning_fwhm_ts_cm_mean"]),
                },
                _f,
                indent=2,
            )
        print(
            f"Seed {p_run.seed} results: "
            f"sigma_theta={r['pv_sigma_theta']:.4f} rad, "
            f"valid_fraction={r['pv_valid_fraction']:.3f}"
        )

    result = results[0]
    output = figures_dir / f"brian2_spatial_two_stage_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    save_summary_figure(result, output)
    test_only_out = figures_dir / (
        f"brian2_test_phase_only_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_test_phase_only_figure(result, test_only_out)
    curves_out = figures_dir / f"brian2_learning_curves_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    save_learning_curves_figure(result, curves_out)
    llmon_out = figures_dir / f"brian2_llmon_weights_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    save_ll_mon_weights_figure(result, llmon_out)
    ts_tuning_out = figures_dir / (
        f"brian2_ts_tuning_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_ts_tuning_figure(result, ts_tuning_out)
    mon_ts_prof = figures_dir / (
        f"brian2_mon_ts_weight_profile_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_mon_to_ts_weight_profile(result, mon_ts_prof)
    mon_ts_rf = figures_dir / (
        f"brian2_mon_to_ts_receptive_fields_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_mon_to_ts_receptive_fields_figure(result, mon_ts_rf)
    ts_vs_x_out = figures_dir / (
        f"brian2_ts_spikes_vs_x_test_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_ts_spikes_vs_x_test_figure(result, ts_vs_x_out)
    mon_vs_x_out = figures_dir / (
        f"brian2_mon_spikes_vs_x_test_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_mon_spikes_vs_x_test_figure(result, mon_vs_x_out)
    ll_vs_x_out = figures_dir / (
        f"brian2_ll_spikes_vs_x_test_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_ll_spikes_vs_x_test_figure(result, ll_vs_x_out)
    mon_tune_ex_out = figures_dir / (
        f"brian2_mon_tuning_examples_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_mon_tuning_examples_figure(result, mon_tune_ex_out)
    mon_ts_drive_hm = figures_dir / (
        f"brian2_mon_ts_drive_heatmap_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    mon_ts_drive_win = figures_dir / (
        f"brian2_mon_ts_drive_winner_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_mon_ts_feedforward_drive_figures(result, mon_ts_drive_hm, mon_ts_drive_win)
    ts_tr_out = figures_dir / (
        f"brian2_ts_pop_rate_train_test_transition_u_{params.speed_cm_s:.1f}_nMON_{params.n_mon}_nTS_{params.n_ts}.png"
    )
    save_ts_pop_rate_train_test_transition_figure(result, ts_tr_out)
    if args.mode == "ll_thesis":
        latest_plot = figures_dir / "LL_THESIS_BASELINE_ACTIVE_latest.png"
        latest_plot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, latest_plot)
        latest_test = figures_dir / "LL_THESIS_BASELINE_TEST_ONLY_latest.png"
        shutil.copy2(test_only_out, latest_test)
    if n_runs > 1:
        multi_out = figures_dir / f"brian2_multiseed_summary_n{n_runs}.png"
        save_multiseed_summary(results, multi_out)
        print(f"Saved multiseed summary: {multi_out}")

    summary = {
        "ll_mon_topo": float(params.ll_to_mon_topography_strength),
        "mon_ts_topo": float(params.mon_to_ts_topography_strength),
        "mon_ts_sigma": float(params.mon_to_ts_sigma),
        "mon_ts_gain_mV": float(params.mon_ts_gain_mV),
    }
    per = []
    for r in results:
        ts_t_abs = np.asarray(r["sp_ts"].t / b2.second, dtype=float)
        t0 = float(r["train_duration_s"])
        t1 = float(r["train_duration_s"] + r["test_duration_s"])
        n_ts_test = int(np.sum((ts_t_abs >= t0) & (ts_t_abs < t1)))
        per.append(
            {
                "seed": int(r["params"].seed),
                "ts_spikes_during_test_window": n_ts_test,
                "pv_map_quality": {
                    "sigma_theta_rad": float(r["pv_sigma_theta"]),
                    "delta_trial_rad": float(r["pv_delta_trial"]),
                    "valid_fraction": float(r["pv_valid_fraction"]),
                },
            }
        )
    if n_runs == 1:
        summary["seed"] = per[0]["seed"]
        summary["ts_spikes_during_test_window"] = per[0]["ts_spikes_during_test_window"]
        summary["pv_map_quality"] = per[0]["pv_map_quality"]
    else:
        summary["runs"] = per
    # Map health at a glance: sigma_theta < 1 rad is good; valid_fraction > 0.5 means
    # TS was active for most of the test window and produced meaningful PV estimates.
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Mode: {args.mode}")
    print(
        "Config: "
        f"train_trials={params.n_training_trials}, trial_dur={params.trial_duration_s}s, "
        f"MON->TS out_degree={params.mon_to_ts_out_degree}, sigma={params.mon_to_ts_sigma}, "
        f"topo_strength={params.mon_to_ts_topography_strength}, "
        f"MON->TS gain={params.mon_ts_gain_mV}mV, "
        f"MON inh={params.global_inh_to_mon_mV}mV, TS lat peak={params.ts_local_inh_peak_mV}mV, "
        f"TS lat radius={params.ts_lateral_radius}, "
        f"train_noise=({params.training_noise_scale_early}->{params.training_noise_scale_late}), "
        f"ll_rate_mode={params.ll_rate_mode}, ll_baseline_subtract_hz={params.ll_rate_baseline_subtract_hz}, "
        f"ll_rate_gain={params.ll_rate_gain}"
    )
    print(f"Saved: {output}")
    print(f"Saved test-phase-only figure: {test_only_out}")
    print(f"Saved MON->TS weight profile: {mon_ts_prof}")
    print(f"Saved MON->TS receptive fields: {mon_ts_rf}")
    print(f"Saved TS spikes vs x (test): {ts_vs_x_out}")
    print(f"Saved MON spikes vs x (test): {mon_vs_x_out}")
    print(f"Saved MON tuning examples (test): {mon_tune_ex_out}")
    print(f"Saved MON→TS drive heatmap (test): {mon_ts_drive_hm}")
    print(f"Saved MON→TS drive winner (test): {mon_ts_drive_win}")
    print(f"Saved TS pop rate train→test: {ts_tr_out}")
    print(f"Saved learning curves: {curves_out}")
    if args.mode == "ll_thesis":
        print(f"Saved latest copy: {figures_dir / 'LL_THESIS_BASELINE_ACTIVE_latest.png'}")
        print(f"Saved latest test-only copy: {figures_dir / 'LL_THESIS_BASELINE_TEST_ONLY_latest.png'}")
    if n_runs == 1:
        print(f"Spikes: LL={result['sp_ll'].num_spikes}, MON={result['sp_mon'].num_spikes}, TS={result['sp_ts'].num_spikes}")
        ts_t_abs = np.asarray(result["sp_ts"].t / b2.second, dtype=float)
        t0 = float(result["train_duration_s"])
        t1 = float(result["train_duration_s"] + result["test_duration_s"])
        n_ts_test = int(np.sum((ts_t_abs >= t0) & (ts_t_abs < t1)))
        print(f"TS spikes during test window: {n_ts_test}")
        n_mon_ts = len(result["mon_to_ts_i"])
        tot_s = result["total_duration_s"]
        mon_rate_per_neuron = result["sp_mon"].num_spikes / max(1, result["params"].n_mon * tot_s)
        print(f"MON->TS synapses: {n_mon_ts}, MON rate per neuron: {mon_rate_per_neuron:.4f} Hz")
        ts_rate = result["pr_ts"].smooth_rate(width=20 * b2.ms) / b2.Hz
        print(f"Max TS population rate (20 ms smooth): {float(np.max(ts_rate)):.2f} Hz")
        print(
            "Weight change: "
            f"mean|dw|={result['w_mean_abs_delta_final']:.6f}, "
            f"frac(|dw|>1e-3)={result['w_frac_delta_gt_1e3_final']:.3f}, "
            f"frac(w==wmax)={result['w_frac_at_wmax_final']:.3f}, "
            f"frac(w==0)={result['w_frac_at_zero_final']:.3f}"
        )
    else:
        sig = np.array([r["pv_sigma_theta"] for r in results], dtype=float)
        dtr = np.array([r["pv_delta_trial"] for r in results], dtype=float)
        print(
            "Multi-seed PV mean±std: "
            f"sigma_theta={float(np.mean(sig)):.4f}±{float(np.std(sig)):.4f}, "
            f"delta_trial={float(np.mean(dtr)):.4f}±{float(np.std(dtr)):.4f}"
        )
    if result["stabilization_time_s"] is None:
        print("Estimated stabilization time: not reached")
    else:
        print(f"Estimated stabilization time: {result['stabilization_time_s']:.2f} s")
    print(
        "PV map quality: "
        f"sigma_theta={result['pv_sigma_theta']:.4f} rad, "
        f"delta_trial={result['pv_delta_trial']:.4f} rad, "
        f"valid_fraction={result['pv_valid_fraction']:.3f}"
    )
    print(f"Run directory: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
