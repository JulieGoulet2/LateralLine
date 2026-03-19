import argparse
from dataclasses import replace
from pathlib import Path

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

from ll_stdp_brian2 import (
    NetworkParams,
    apply_model_mode,
    build_ll_to_mon_indices,
    build_mon_to_ts_indices,
    estimate_stabilization_time,
    make_training_rates,
    pv_map_quality_from_ts_spikes,
)
from stimulus import StimulusParams, simulate_lateral_line

b2.prefs.codegen.target = "numpy"


def _apply_ll_rate_transform(rates_hz: np.ndarray, params: NetworkParams) -> np.ndarray:
    baseline_subtract_hz = float(max(0.0, params.ll_rate_baseline_subtract_hz))
    if params.ll_rate_mode == "modulation":
        baseline_subtract_hz += float(StimulusParams().r0_hz)
    elif params.ll_rate_mode != "raw":
        raise ValueError(f"Unknown ll_rate_mode '{params.ll_rate_mode}'")
    return np.clip((rates_hz - baseline_subtract_hz) * float(max(0.0, params.ll_rate_gain)), 0.0, None)


def _make_test_sim(params: NetworkParams, distance_cm: float, noise_scale: float, seed: int):
    stim_params = StimulusParams()
    stim_params.sigma_noise_hz = stim_params.sigma_noise_hz * float(max(0.0, noise_scale))
    duration_s = params.test_path_cm / max(params.speed_cm_s, 1e-9)
    sim = simulate_lateral_line(
        duration_s=duration_s,
        dt_s=params.dt_s,
        n_neuromasts=params.n_ll,
        seed=seed,
        params=stim_params,
        fixed_distance_cm=distance_cm,
        direction=1.0,
        fixed_speed_cm_s=params.speed_cm_s,
    )
    sim["rates_hz"] = _apply_ll_rate_transform(sim["rates_hz"], params)
    return sim


def _make_dynamic_sequence_sim(
    params: NetworkParams,
    distances_cm: list[float],
    noise_scale: float,
    seed: int,
):
    """
    Build a true time-varying test where sphere position changes continuously in time
    by concatenating multiple forward traversals at different Y distances.
    """
    sims = []
    for k, d in enumerate(distances_cm):
        direction = 1.0
        stim_params = StimulusParams()
        stim_params.sigma_noise_hz = stim_params.sigma_noise_hz * float(max(0.0, noise_scale))
        # Use -1 -> +4 cm sweep so clipped x(t) clearly spans 0..4.
        duration_s = (stim_params.lateral_line_length_cm + 1.0) / max(params.speed_cm_s, 1e-9)
        seg = simulate_lateral_line(
            duration_s=duration_s,
            dt_s=params.dt_s,
            n_neuromasts=params.n_ll,
            seed=seed + k,
            params=stim_params,
            fixed_distance_cm=float(d),
            direction=direction,
            fixed_speed_cm_s=params.speed_cm_s,
        )
        seg["rates_hz"] = _apply_ll_rate_transform(seg["rates_hz"], params)
        sims.append(seg)

    t_all = []
    X_all = []
    Y_all = []
    rates_all = []
    t_offset = 0.0
    for seg in sims:
        t_seg = np.asarray(seg["t_s"], dtype=float) + t_offset
        t_all.append(t_seg)
        X_all.append(np.asarray(seg["X_cm"], dtype=float))
        Y_all.append(np.asarray(seg["Y_cm"], dtype=float))
        rates_all.append(np.asarray(seg["rates_hz"], dtype=float))
        t_offset = float(t_seg[-1] + params.dt_s)

    return {
        "t_s": np.concatenate(t_all),
        "X_cm": np.concatenate(X_all),
        "Y_cm": np.concatenate(Y_all),
        "X_clip_cm": np.clip(np.concatenate(X_all), 0.0, 4.0),
        "rates_hz": np.concatenate(rates_all, axis=0),
    }


def _bin_ts_rate_over_x(
    spike_t_s: np.ndarray,
    spike_i: np.ndarray,
    n_ts: int,
    test_start_s: float,
    t_s: np.ndarray,
    x_cm: np.ndarray,
    dt_s: float,
    n_x_bins: int = 80,
):
    x_edges = np.linspace(0.0, 4.0, n_x_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    rates = np.zeros((n_ts, n_x_bins), dtype=float)
    occupancy = np.zeros(n_x_bins, dtype=float)

    xk = np.clip(np.digitize(np.clip(x_cm, 0.0, 4.0), x_edges) - 1, 0, n_x_bins - 1)
    for b in xk:
        occupancy[b] += dt_s

    k = np.floor((spike_t_s - test_start_s) / dt_s).astype(int)
    valid = (k >= 0) & (k < t_s.size) & (spike_i >= 0) & (spike_i < n_ts)
    if np.any(valid):
        bins = xk[k[valid]]
        np.add.at(rates, (spike_i[valid], bins), 1.0)

    occ = np.maximum(occupancy, 1e-9)
    rates = rates / occ[None, :]
    return x_centers, rates


def _bin_ts_rate_over_time(
    spike_t_s: np.ndarray,
    spike_i: np.ndarray,
    n_ts: int,
    test_start_s: float,
    t_s: np.ndarray,
    dt_s: float,
    smooth_s: float = 0.02,
):
    """
    Time-aligned TS rates for a test segment, plus mean active TS index over time.
    """
    n_t = t_s.size
    rates_t = np.zeros((n_ts, n_t), dtype=float)
    k = np.floor((spike_t_s - test_start_s) / dt_s).astype(int)
    valid = (k >= 0) & (k < n_t) & (spike_i >= 0) & (spike_i < n_ts)
    if np.any(valid):
        np.add.at(rates_t, (spike_i[valid], k[valid]), 1.0 / dt_s)

    # Visualization smoothing so sparse TS spikes are visible as activity bands.
    win = max(1, int(round(smooth_s / max(dt_s, 1e-12))))
    if win > 1:
        ker = np.ones(win, dtype=float) / float(win)
        rates_t = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 1, rates_t)

    ts_idx = np.arange(n_ts, dtype=float)[:, None]
    denom = np.sum(rates_t, axis=0)
    mean_idx = np.full(n_t, np.nan, dtype=float)
    nz = denom > 1e-12
    if np.any(nz):
        mean_idx[nz] = np.sum(rates_t[:, nz] * ts_idx, axis=0) / denom[nz]

    return t_s.copy(), rates_t, mean_idx


def _robust_vmax(a: np.ndarray, q: float = 99.5) -> float:
    vals = np.asarray(a, dtype=float)
    if vals.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(vals, q))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else 1.0
    return max(vmax, 1e-6)


def _decode_position_from_rate_matrix(rate_matrix: np.ndarray, n_units: int, x_len_cm: float = 4.0):
    """
    Decode x(t) from population rate matrix [n_units, n_t] using weighted mean index.
    """
    idx = np.arange(n_units, dtype=float)[:, None]
    denom = np.sum(rate_matrix, axis=0)
    x_hat = np.full(rate_matrix.shape[1], np.nan, dtype=float)
    nz = denom > 1e-12
    if np.any(nz):
        mean_idx = np.sum(rate_matrix[:, nz] * idx, axis=0) / denom[nz]
        x_hat[nz] = x_len_cm * mean_idx / max(n_units - 1, 1)
    return x_hat


def _nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return float("nan")
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    da = np.sqrt(np.sum(aa * aa))
    db = np.sqrt(np.sum(bb * bb))
    if da <= 1e-12 or db <= 1e-12:
        return float("nan")
    return float(np.sum(aa * bb) / (da * db))


def _build_training_rates_curriculum(
    p: NetworkParams,
    train_trials: int,
    seed: int,
    phase1_frac: float,
    phase2_frac: float,
    fixed_distance_cm: float,
    final_noise_scale: float,
):
    """
    Curriculum:
    1) clean fixed-distance near-field,
    2) clean variable near-field,
    3) variable near-field + mild noise.
    """
    t1 = int(round(train_trials * phase1_frac))
    t2 = int(round(train_trials * phase2_frac))
    t3 = max(0, train_trials - t1 - t2)
    parts = []

    if t1 > 0:
        p1 = replace(
            p,
            seed=seed + 101,
            n_training_trials=t1,
            training_fixed_distance=True,
            distance_cm=float(fixed_distance_cm),
            training_noise_scale_early=0.0,
            training_noise_scale_late=0.0,
        )
        r1, _, _ = make_training_rates(p1)
        parts.append(_apply_ll_rate_transform(r1, p))
    if t2 > 0:
        p2 = replace(
            p,
            seed=seed + 202,
            n_training_trials=t2,
            training_fixed_distance=False,
            training_noise_scale_early=0.0,
            training_noise_scale_late=0.0,
        )
        r2, _, _ = make_training_rates(p2)
        parts.append(_apply_ll_rate_transform(r2, p))
    if t3 > 0:
        p3 = replace(
            p,
            seed=seed + 303,
            n_training_trials=t3,
            training_fixed_distance=False,
            training_noise_scale_early=float(max(0.0, final_noise_scale)),
            training_noise_scale_late=float(max(0.0, final_noise_scale)),
        )
        r3, _, _ = make_training_rates(p3)
        parts.append(_apply_ll_rate_transform(r3, p))

    if not parts:
        return _apply_ll_rate_transform(make_training_rates(replace(p, n_training_trials=max(1, train_trials)))[0], p)
    return np.vstack(parts)


def run_stage_d(
    params: NetworkParams,
    train_trials: int,
    distances_cm: list[float],
    noise_levels: list[float],
    seed: int,
    use_curriculum: bool = True,
    curriculum_phase1_frac: float = 0.35,
    curriculum_phase2_frac: float = 0.45,
    curriculum_fixed_distance_cm: float = 1.2,
    curriculum_final_noise_scale: float = 0.2,
    prior_boost_strength: float = 0.25,
    prior_decay_fraction: float = 0.35,
    homeo_eta: float = 0.06,
    homeo_every_trials: int = 5,
    mon_ts_prior_scale: float = 0.0,
):
    p = replace(params, n_training_trials=max(1, int(train_trials)), seed=seed)

    if use_curriculum:
        train_rates = _build_training_rates_curriculum(
            p=p,
            train_trials=p.n_training_trials,
            seed=seed,
            phase1_frac=float(np.clip(curriculum_phase1_frac, 0.0, 1.0)),
            phase2_frac=float(np.clip(curriculum_phase2_frac, 0.0, 1.0)),
            fixed_distance_cm=float(curriculum_fixed_distance_cm),
            final_noise_scale=float(max(0.0, curriculum_final_noise_scale)),
        )
        train_samples = None
    else:
        train_rates, train_samples, _ = make_training_rates(p)
        train_rates = _apply_ll_rate_transform(train_rates, p)
    train_duration_s = p.n_training_trials * p.trial_duration_s

    b2.start_scope()
    b2.defaultclock.dt = p.dt_s * b2.second

    input_ta = b2.TimedArray(train_rates * b2.Hz, dt=p.dt_s * b2.second)
    ll = b2.PoissonGroup(p.n_ll, rates="input_ta(t, i)", namespace={"input_ta": input_ta})

    eqs = """
    dv/dt = (El - v + ge - gi) / tau_m : volt (unless refractory)
    dge/dt = -ge / tau_s : volt
    dgi/dt = -gi / tau_s : volt
    """
    ns = {
        "Vth": p.vth_mV * b2.mV,
        "Vreset": p.vreset_mV * b2.mV,
        "El": p.el_mV * b2.mV,
        "tau_ref": p.tau_ref_ms * b2.ms,
        "tau_m": p.tau_m_ms * b2.ms,
        "tau_s": p.tau_s_ms * b2.ms,
    }
    mon = b2.NeuronGroup(p.n_mon, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)
    ts = b2.NeuronGroup(p.n_ts, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)
    mon_inh = b2.NeuronGroup(1, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)
    ts_inh = b2.NeuronGroup(1, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)
    mon.v = "El + rand()*2*mV"
    ts.v = "El + rand()*2*mV"
    mon_inh.v = "El"
    ts_inh.v = "El"

    s_ll_mon = b2.Synapses(ll, mon, "w : volt", on_pre="ge_post += w")
    if p.ll_to_mon_topography_strength <= 0.0:
        s_ll_mon.connect(p=p.p_ll_to_mon)
    else:
        ll_i, mon_j = build_ll_to_mon_indices(
            n_ll=p.n_ll,
            n_mon=p.n_mon,
            in_degree=p.ll_to_mon_in_degree,
            sigma_ll=p.ll_to_mon_sigma,
            topography_strength=p.ll_to_mon_topography_strength,
            seed=p.seed + 31,
        )
        s_ll_mon.connect(i=ll_i, j=mon_j)
    s_ll_mon.w = f"{p.ll_mon_w_mean_mV}*mV + {p.ll_mon_w_jitter_mV}*mV*rand()"

    mon_i, ts_j, _ = build_mon_to_ts_indices(
        n_mon=p.n_mon,
        n_ts=p.n_ts,
        out_degree=p.mon_to_ts_out_degree,
        sigma_ts=p.mon_to_ts_sigma,
        topography_strength=p.mon_to_ts_topography_strength,
        seed=p.seed,
    )
    s_mon_ts = b2.Synapses(
        mon,
        ts,
        model="""
        w : 1
        w_prior : 1
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        """,
        on_pre="""
        ge_post += mon_ts_gain*(w + prior_scale*w_prior)*mV
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
            "Apre": p.mon_ts_apre,
            "Apost": p.mon_ts_apost,
            "wmax": p.mon_ts_wmax,
            "mon_ts_gain": p.mon_ts_gain_mV,
            "prior_scale": float(max(0.0, mon_ts_prior_scale)),
        },
    )
    s_mon_ts.connect(i=mon_i, j=ts_j)
    s_mon_ts.w = f"{p.mon_ts_w_init} + {p.mon_ts_w_jitter}*rand()"
    # Developmental weak topographic prior in initial weights (temporary, decays later).
    prior_sigma = max(1.0, float(p.mon_to_ts_sigma))
    mon_center_ts = np.asarray(mon_i, dtype=float) * (p.n_ts - 1) / max(p.n_mon - 1, 1)
    ts_pos = np.asarray(ts_j, dtype=float)
    topo_prior = np.exp(-0.5 * ((ts_pos - mon_center_ts) / prior_sigma) ** 2)
    s_mon_ts.w_prior = topo_prior
    if prior_boost_strength > 0.0:
        w0 = np.array(s_mon_ts.w[:], dtype=float, copy=True)
        w0 = np.clip(w0 + float(prior_boost_strength) * topo_prior * p.mon_ts_w_init, 0.0, p.mon_ts_wmax)
        s_mon_ts.w = w0
    w_init = np.array(s_mon_ts.w[:], dtype=float, copy=True)

    bg_mon = b2.PoissonGroup(p.n_mon, rates=p.bg_rate_mon_hz * b2.Hz)
    bg_ts = b2.PoissonGroup(p.n_ts, rates=p.bg_rate_ts_hz * b2.Hz)
    s_bg_mon = b2.Synapses(bg_mon, mon, on_pre=f"ge_post += {p.bg_w_mon_mV}*mV")
    s_bg_mon.connect(condition="i == j")
    s_bg_ts = b2.Synapses(bg_ts, ts, on_pre=f"ge_post += {p.bg_w_ts_mV}*mV")
    s_bg_ts.connect(condition="i == j")

    s_mon_to_inh = b2.Synapses(mon, mon_inh, on_pre=f"ge_post += {p.mon_to_global_inh_drive_mV}*mV")
    s_mon_to_inh.connect(p=p.mon_to_global_inh_p)
    s_inh_to_mon = b2.Synapses(mon_inh, mon, "w : volt", on_pre="gi_post += w")
    s_inh_to_mon.connect()
    s_inh_to_mon.w = p.global_inh_to_mon_mV * b2.mV

    src, dst, wlat = [], [], []
    rad = p.ts_lateral_radius
    for i in range(p.n_ts):
        lo = max(0, i - rad)
        hi = min(p.n_ts, i + rad + 1)
        for j in range(lo, hi):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            wlat.append(p.ts_local_inh_peak_mV * (1.0 - abs(i - j) / rad))
    s_ts_lat = b2.Synapses(ts, ts, "w : volt", on_pre="gi_post += w")
    s_ts_lat.connect(i=np.asarray(src), j=np.asarray(dst))
    s_ts_lat.w = np.asarray(wlat, dtype=float) * b2.mV

    # Activity-dependent TS feedback inhibition:
    # more TS spikes -> stronger inhibitory feedback onto TS.
    ts_to_inh_drive_mV = float(getattr(p, "ts_to_global_inh_drive_mV", 0.35))
    ts_inh_to_ts_mV = float(getattr(p, "global_inh_to_ts_mV", 0.8))
    s_ts_to_inh = b2.Synapses(ts, ts_inh, on_pre=f"ge_post += {ts_to_inh_drive_mV}*mV")
    s_ts_to_inh.connect(p=0.12)
    s_inh_to_ts = b2.Synapses(ts_inh, ts, "w : volt", on_pre="gi_post += w")
    s_inh_to_ts.connect()
    s_inh_to_ts.w = ts_inh_to_ts_mV * b2.mV

    sp_ts = b2.SpikeMonitor(ts)
    pr_ts = b2.PopulationRateMonitor(ts)
    pr_mon = b2.PopulationRateMonitor(mon)
    net = b2.Network(b2.collect())

    rng = np.random.default_rng(seed + 777)
    tracked_n = min(24, w_init.size)
    tracked_idx = rng.choice(w_init.size, size=tracked_n, replace=False)
    checkpoint_t_s = [0.0]
    w_mean_series = [float(np.mean(w_init))]
    w_std_series = [float(np.std(w_init))]
    tracked_weight_series = [w_init[tracked_idx].copy()]
    syn_ts_idx = np.asarray(ts_j, dtype=int)
    incoming0 = np.bincount(syn_ts_idx, weights=w_init, minlength=p.n_ts)
    homeo_target_incoming = float(np.mean(incoming0[incoming0 > 0])) if np.any(incoming0 > 0) else float(np.mean(incoming0))

    def run_test_battery(phase: str, seed_offset: int):
        out = []
        local_seed_k = 0
        for noise in noise_levels:
            for d in distances_cm:
                sim = _make_test_sim(p, distance_cm=d, noise_scale=noise, seed=seed + seed_offset + local_seed_k)
                local_seed_k += 1
                test_ta = b2.TimedArray(sim["rates_hz"] * b2.Hz, dt=p.dt_s * b2.second)
                ll.namespace["input_ta"] = test_ta

                t0 = float(b2.defaultclock.t / b2.second)
                net.run(float(sim["t_s"][-1] + p.dt_s) * b2.second, namespace={})
                t1 = float(b2.defaultclock.t / b2.second)

                st = np.asarray(sp_ts.t / b2.second, dtype=float)
                si = np.asarray(sp_ts.i, dtype=int)
                seg = (st >= t0) & (st < t1)

                pv = pv_map_quality_from_ts_spikes(
                    ts_spike_t_s=st[seg],
                    ts_spike_i=si[seg],
                    n_ts=p.n_ts,
                    test_t_s=np.asarray(sim["t_s"], dtype=float),
                    test_x_cm=np.asarray(sim["X_cm"], dtype=float),
                    lateral_line_len_cm=float(StimulusParams().lateral_line_length_cm),
                    test_start_s=t0,
                    dt_s=p.dt_s,
                    n_pos_bins=min(100, p.n_ts),
                )
                x_centers, ts_x_rate = _bin_ts_rate_over_x(
                    spike_t_s=st[seg],
                    spike_i=si[seg],
                    n_ts=p.n_ts,
                    test_start_s=t0,
                    t_s=np.asarray(sim["t_s"], dtype=float),
                    x_cm=np.asarray(sim["X_cm"], dtype=float),
                    dt_s=p.dt_s,
                    n_x_bins=80,
                )
                t_axis_s, ts_t_rate, mean_ts_idx_t = _bin_ts_rate_over_time(
                    spike_t_s=st[seg],
                    spike_i=si[seg],
                    n_ts=p.n_ts,
                    test_start_s=t0,
                    t_s=np.asarray(sim["t_s"], dtype=float),
                    dt_s=p.dt_s,
                )
                out.append(
                    {
                        "phase": phase,
                        "distance_cm": float(d),
                        "noise_scale": float(noise),
                        "pv_sigma_theta": float(pv["sigma_theta"]),
                        "pv_delta_trial": float(pv["delta_trial"]),
                        "pv_valid_fraction": float(pv["valid_fraction"]),
                        "x_centers": x_centers,
                        "ts_x_rate": ts_x_rate,
                        "t_axis_s": t_axis_s,
                        "ts_t_rate": ts_t_rate,
                        "mean_ts_idx_t": mean_ts_idx_t,
                        "ts_spike_count": int(np.sum(seg)),
                    }
                )
        return out

    def run_dynamic_time_test(phase: str, seed_offset: int, noise_scale: float = 0.0):
        sim = _make_dynamic_sequence_sim(
            p,
            distances_cm=distances_cm,
            noise_scale=noise_scale,
            seed=seed + seed_offset,
        )
        dyn_ta = b2.TimedArray(sim["rates_hz"] * b2.Hz, dt=p.dt_s * b2.second)
        ll.namespace["input_ta"] = dyn_ta

        t0 = float(b2.defaultclock.t / b2.second)
        net.run(float(sim["t_s"][-1] + p.dt_s) * b2.second, namespace={})
        t1 = float(b2.defaultclock.t / b2.second)

        st = np.asarray(sp_ts.t / b2.second, dtype=float)
        si = np.asarray(sp_ts.i, dtype=int)
        seg = (st >= t0) & (st < t1)
        t_axis, ts_t_rate, mean_ts_idx_t = _bin_ts_rate_over_time(
            spike_t_s=st[seg],
            spike_i=si[seg],
            n_ts=p.n_ts,
            test_start_s=t0,
            t_s=np.asarray(sim["t_s"], dtype=float),
            dt_s=p.dt_s,
        )
        dur = float(sim["t_s"][-1] + p.dt_s)
        ts_spike_count = int(np.sum(seg))
        ts_mean_rate_hz = float(ts_spike_count / max(dur * p.n_ts, 1e-12))
        return {
            "phase": phase,
            "noise_scale": float(noise_scale),
            "t_axis_s": t_axis,
            "X_cm": np.asarray(sim["X_cm"], dtype=float),
            "X_clip_cm": np.asarray(sim["X_clip_cm"], dtype=float),
            "Y_cm": np.asarray(sim["Y_cm"], dtype=float),
            "ll_t_rate": np.asarray(sim["rates_hz"], dtype=float).T,
            "ll_mean_rate_t": np.asarray(np.mean(sim["rates_hz"], axis=1), dtype=float),
            "ts_t_rate": ts_t_rate,
            "mean_ts_idx_t": mean_ts_idx_t,
            "x_hat_ll_cm": _decode_position_from_rate_matrix(np.asarray(sim["rates_hz"], dtype=float).T, p.n_ll, 4.0),
            "x_hat_ts_cm": _decode_position_from_rate_matrix(ts_t_rate, p.n_ts, 4.0),
            "x_hat_corr": _nan_corr(
                _decode_position_from_rate_matrix(np.asarray(sim["rates_hz"], dtype=float).T, p.n_ll, 4.0),
                _decode_position_from_rate_matrix(ts_t_rate, p.n_ts, 4.0),
            ),
            "ts_spike_count": ts_spike_count,
            "ts_mean_rate_hz": ts_mean_rate_hz,
        }

    # BEFORE learning test (STDP frozen)
    Apre0 = float(s_mon_ts.namespace["Apre"])
    Apost0 = float(s_mon_ts.namespace["Apost"])
    s_mon_ts.namespace["Apre"] = 0.0
    s_mon_ts.namespace["Apost"] = 0.0
    test_results = run_test_battery(phase="before", seed_offset=1000)
    dynamic_results = [run_dynamic_time_test(phase="before", seed_offset=2000, noise_scale=0.0)]

    # Training (STDP ON)
    s_mon_ts.namespace["Apre"] = Apre0
    s_mon_ts.namespace["Apost"] = Apost0
    prior_decay_trials = max(1, int(round(float(np.clip(prior_decay_fraction, 0.0, 1.0)) * p.n_training_trials)))
    prior_step = float(max(0.0, prior_boost_strength)) * p.mon_ts_w_init / max(prior_decay_trials, 1)
    for k in range(p.n_training_trials):
        net.run(p.trial_duration_s * b2.second, namespace={})

        # Early decay of imposed topographic prior.
        if prior_boost_strength > 0.0 and k < prior_decay_trials:
            wk = np.array(s_mon_ts.w[:], dtype=float, copy=True)
            wk = np.clip(wk - prior_step * topo_prior, 0.0, p.mon_ts_wmax)
            s_mon_ts.w = wk

        # Slow homeostatic normalization of incoming weights per TS neuron.
        if homeo_eta > 0.0 and ((k + 1) % max(1, int(homeo_every_trials)) == 0):
            wk = np.array(s_mon_ts.w[:], dtype=float, copy=True)
            incoming = np.bincount(syn_ts_idx, weights=wk, minlength=p.n_ts)
            scale = np.ones(p.n_ts, dtype=float)
            nonzero = incoming > 1e-12
            if np.any(nonzero):
                ratio = homeo_target_incoming / np.maximum(incoming[nonzero], 1e-12)
                scale[nonzero] = 1.0 + float(homeo_eta) * (ratio - 1.0)
                scale = np.clip(scale, 0.9, 1.1)
                wk = np.clip(wk * scale[syn_ts_idx], 0.0, p.mon_ts_wmax)
                s_mon_ts.w = wk

        if (k + 1) % max(1, p.checkpoint_trials) == 0 or (k + 1) == p.n_training_trials:
            wk = np.array(s_mon_ts.w[:], dtype=float, copy=True)
            checkpoint_t_s.append((k + 1) * p.trial_duration_s)
            w_mean_series.append(float(np.mean(wk)))
            w_std_series.append(float(np.std(wk)))
            tracked_weight_series.append(wk[tracked_idx].copy())
    w_after_train = np.array(s_mon_ts.w[:], dtype=float, copy=True)

    # AFTER learning test (STDP frozen)
    s_mon_ts.namespace["Apre"] = 0.0
    s_mon_ts.namespace["Apost"] = 0.0
    test_results.extend(run_test_battery(phase="after", seed_offset=5000))
    dynamic_results.append(run_dynamic_time_test(phase="after", seed_offset=6000, noise_scale=0.0))

    return {
        "params": p,
        "train_samples": train_samples,
        "pr_ts": pr_ts,
        "pr_mon": pr_mon,
        "w_init": w_init,
        "w_after_train": w_after_train,
        "checkpoint_t_s": np.asarray(checkpoint_t_s),
        "w_mean_series": np.asarray(w_mean_series),
        "w_std_series": np.asarray(w_std_series),
        "tracked_weight_series": np.asarray(tracked_weight_series),
        "stabilization_time_s": estimate_stabilization_time(np.asarray(checkpoint_t_s), np.asarray(w_mean_series)),
        "test_results": test_results,
        "dynamic_results": dynamic_results,
        "use_curriculum": bool(use_curriculum),
        "curriculum_phase1_frac": float(curriculum_phase1_frac),
        "curriculum_phase2_frac": float(curriculum_phase2_frac),
        "curriculum_fixed_distance_cm": float(curriculum_fixed_distance_cm),
        "curriculum_final_noise_scale": float(curriculum_final_noise_scale),
        "prior_boost_strength": float(prior_boost_strength),
        "prior_decay_fraction": float(prior_decay_fraction),
        "homeo_eta": float(homeo_eta),
        "homeo_every_trials": int(homeo_every_trials),
        "mon_ts_prior_scale": float(mon_ts_prior_scale),
    }


def _save_no_noise_distance_map(result: dict, out_dir: Path, tag: str):
    rows = [r for r in result["test_results"] if abs(r["noise_scale"]) < 1e-12]
    phases = sorted(set(r["phase"] for r in rows))
    out_paths = []
    for phase in phases:
        prow = sorted([r for r in rows if r["phase"] == phase], key=lambda r: r["distance_cm"])
        if not prow:
            continue
        fig, ax = plt.subplots(len(prow), 1, figsize=(11, 2.7 * len(prow)), sharex=True)
        if len(prow) == 1:
            ax = [ax]
        vmax = max(float(np.max(r["ts_x_rate"])) for r in prow)
        for k, r in enumerate(prow):
            im = ax[k].imshow(
                r["ts_x_rate"],
                aspect="auto",
                origin="lower",
                extent=[0.0, 4.0, 0, result["params"].n_ts - 1],
                vmin=0.0,
                vmax=max(vmax, 1e-9),
                cmap="viridis",
            )
            ax[k].set_ylabel("TS idx")
            ax[k].set_title(
                f"{phase.upper()} learning | no-noise test | Y={r['distance_cm']:.2f} cm | "
                f"sigma={r['pv_sigma_theta']:.3f} | valid={r['pv_valid_fraction']:.3f}"
            )
            fig.colorbar(im, ax=ax[k], label="TS rate (Hz)")
        ax[-1].set_xlabel("Sphere x position along lateral line (cm)")
        fig.tight_layout()
        path = out_dir / f"stageD_torus_map_no_noise_{phase}_{tag}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        out_paths.append(path)
    return out_paths


def _save_noise_level_map(result: dict, out_dir: Path, tag: str, distance_for_noise_panel: float):
    rows = [r for r in result["test_results"] if abs(r["distance_cm"] - distance_for_noise_panel) < 1e-9]
    phases = sorted(set(r["phase"] for r in rows))
    out_paths = []
    for phase in phases:
        prow = sorted([r for r in rows if r["phase"] == phase], key=lambda r: r["noise_scale"])
        if not prow:
            continue
        fig, ax = plt.subplots(len(prow), 1, figsize=(11, 2.7 * len(prow)), sharex=True)
        if len(prow) == 1:
            ax = [ax]
        vmax = max(float(np.max(r["ts_x_rate"])) for r in prow)
        for k, r in enumerate(prow):
            im = ax[k].imshow(
                r["ts_x_rate"],
                aspect="auto",
                origin="lower",
                extent=[0.0, 4.0, 0, result["params"].n_ts - 1],
                vmin=0.0,
                vmax=max(vmax, 1e-9),
                cmap="magma",
            )
            ax[k].set_ylabel("TS idx")
            ax[k].set_title(
                f"{phase.upper()} learning | noise={r['noise_scale']:.2f} | Y={r['distance_cm']:.2f} cm | "
                f"sigma={r['pv_sigma_theta']:.3f} | valid={r['pv_valid_fraction']:.3f}"
            )
            fig.colorbar(im, ax=ax[k], label="TS rate (Hz)")
        ax[-1].set_xlabel("Sphere x position along lateral line (cm)")
        fig.tight_layout()
        path = out_dir / f"stageD_torus_map_noise_levels_{phase}_{tag}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        out_paths.append(path)
    return out_paths


def _save_metrics_csv(result: dict, out_dir: Path, tag: str):
    csv_path = out_dir / f"stageD_torus_metrics_{tag}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("phase,noise_scale,distance_cm,pv_sigma_theta,pv_delta_trial,pv_valid_fraction\n")
        for r in result["test_results"]:
            f.write(
                f"{r['phase']},{r['noise_scale']:.4f},{r['distance_cm']:.4f},"
                f"{r['pv_sigma_theta']:.8f},{r['pv_delta_trial']:.8f},{r['pv_valid_fraction']:.8f}\n"
            )
    return csv_path


def _save_weight_stabilization_plot(result: dict, out_dir: Path, tag: str):
    fig_path = out_dir / f"stageD_weight_stabilization_{tag}.png"
    csv_path = out_dir / f"stageD_weight_stabilization_{tag}.csv"
    ct = result["checkpoint_t_s"]
    wm = result["w_mean_series"]
    ws = result["w_std_series"]
    tw = result["tracked_weight_series"]
    dwdt = np.zeros_like(wm)
    if ct.size >= 2:
        dt = np.diff(ct)
        dt = np.maximum(dt, 1e-12)
        dwdt[1:] = np.diff(wm) / dt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(ct, wm, lw=2.0, color="tab:blue", label="mean(w)")
    ax.fill_between(ct, wm - ws, wm + ws, alpha=0.2, color="tab:blue", label="mean±std")
    for k in range(min(12, tw.shape[1])):
        ax.plot(ct, tw[:, k], lw=0.8, alpha=0.45, color="tab:orange")
    ax2 = ax.twinx()
    ax2.plot(ct, dwdt, lw=1.4, ls="--", color="tab:red", label="d mean(w)/dt")
    ax2.set_ylabel("d mean(w) / dt", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    stab_t = result["stabilization_time_s"]
    if stab_t is not None:
        ax.axvline(stab_t, color="tab:green", ls="--", lw=1.2, label=f"stabilized ~{stab_t:.1f}s")
    ax.set_title("Stage D MON->TS weight stabilization during training")
    ax.set_xlabel("training time (s)")
    ax.set_ylabel("weight")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("checkpoint_t_s,mean_w,std_w,dmean_w_dt\n")
        for t, m, s, d in zip(ct, wm, ws, dwdt):
            f.write(f"{float(t):.8f},{float(m):.10f},{float(s):.10f},{float(d):.10f}\n")

    return fig_path, csv_path


def _save_before_after_comparison_by_noise(result: dict, out_dir: Path, tag: str):
    rows = result["test_results"]
    distances = sorted(set(float(r["distance_cm"]) for r in rows))
    noises = sorted(set(float(r["noise_scale"]) for r in rows))
    paths = []
    for noise in noises:
        subset = [r for r in rows if abs(r["noise_scale"] - noise) < 1e-12]
        if not subset:
            continue
        vmax = max(float(np.max(r["ts_x_rate"])) for r in subset)
        fig, ax = plt.subplots(2, len(distances), figsize=(3.8 * len(distances), 6.0), sharex=True, sharey=True)
        if len(distances) == 1:
            ax = np.array([[ax[0]], [ax[1]]], dtype=object)
        im = None
        for j, d in enumerate(distances):
            rb = next((r for r in subset if r["phase"] == "before" and abs(r["distance_cm"] - d) < 1e-12), None)
            ra = next((r for r in subset if r["phase"] == "after" and abs(r["distance_cm"] - d) < 1e-12), None)
            for i, rr in enumerate([rb, ra]):
                if rr is None:
                    ax[i, j].axis("off")
                    continue
                im = ax[i, j].imshow(
                    rr["ts_x_rate"],
                    aspect="auto",
                    origin="lower",
                    extent=[0.0, 4.0, 0, result["params"].n_ts - 1],
                    vmin=0.0,
                    vmax=max(vmax, 1e-9),
                    cmap="viridis",
                )
                phase_name = "BEFORE" if i == 0 else "AFTER"
                ax[i, j].set_title(
                    f"{phase_name} | Y={d:.2f} cm\n"
                    f"sigma={rr['pv_sigma_theta']:.3f}, valid={rr['pv_valid_fraction']:.3f}",
                    fontsize=9,
                )
                if j == 0:
                    ax[i, j].set_ylabel("TS idx")
                if i == 1:
                    ax[i, j].set_xlabel("x (cm)")
        fig.suptitle(f"Stage D torus map before/after learning | noise scale={noise:.2f}", y=1.02)
        if im is not None:
            cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.85)
            cbar.set_label("TS rate (Hz)")
        fig.tight_layout()
        path = out_dir / f"stageD_before_after_compare_noise_{noise:.2f}_{tag}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _save_before_after_time_panel(result: dict, out_dir: Path, tag: str, noise_for_panel: float = 0.0):
    """
    Stage-A-like panel:
    TS index activity as function of time, plus mean TS index(t), before/after.
    """
    rows = [r for r in result["test_results"] if abs(r["noise_scale"] - noise_for_panel) < 1e-12]
    distances = sorted(set(float(r["distance_cm"]) for r in rows))
    if not distances:
        return None

    fig, ax = plt.subplots(2, len(distances), figsize=(4.0 * len(distances), 6.2), sharex=True, sharey=True)
    if len(distances) == 1:
        ax = np.array([[ax[0]], [ax[1]]], dtype=object)

    vmax = min(50.0, max(_robust_vmax(r["ts_t_rate"], q=99.5) for r in rows))
    last_im = None
    for j, d in enumerate(distances):
        rb = next((r for r in rows if r["phase"] == "before" and abs(r["distance_cm"] - d) < 1e-12), None)
        ra = next((r for r in rows if r["phase"] == "after" and abs(r["distance_cm"] - d) < 1e-12), None)
        for i, rr in enumerate([rb, ra]):
            if rr is None:
                ax[i, j].axis("off")
                continue
            t = rr["t_axis_s"]
            last_im = ax[i, j].imshow(
                rr["ts_t_rate"],
                aspect="auto",
                origin="lower",
                extent=[t[0], t[-1], 0, result["params"].n_ts - 1],
                vmin=0.0,
                vmax=max(vmax, 1e-9),
                cmap="viridis",
            )
            ax[i, j].plot(t, rr["mean_ts_idx_t"], color="white", lw=1.2, alpha=0.9)
            phase_name = "BEFORE" if i == 0 else "AFTER"
            ax[i, j].set_title(
                f"{phase_name} | Y={d:.2f} cm\n"
                f"sigma={rr['pv_sigma_theta']:.3f}, valid={rr['pv_valid_fraction']:.3f}, spikes={rr['ts_spike_count']}",
                fontsize=9,
            )
            if j == 0:
                ax[i, j].set_ylabel("TS index")
            if i == 1:
                ax[i, j].set_xlabel("time (s)")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=ax.ravel().tolist(), shrink=0.85)
        cbar.set_label("TS rate (Hz)")
    fig.suptitle(f"Stage D TS activity vs time (noise={noise_for_panel:.2f})", y=1.02)
    fig.tight_layout()
    path = out_dir / f"stageD_before_after_time_noise_{noise_for_panel:.2f}_{tag}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_dynamic_time_activity(result: dict, out_dir: Path, tag: str):
    """
    Requested plot:
    Sphere position x(t) with explicit 0..4 line, and TS activity over time.
    """
    dyn = result.get("dynamic_results", [])
    if not dyn:
        return None
    phases = sorted(set(d["phase"] for d in dyn))
    fig, ax = plt.subplots(3, len(phases), figsize=(5.2 * len(phases), 9.0), sharex=False)
    if len(phases) == 1:
        ax = np.array([[ax[0]], [ax[1]], [ax[2]]], dtype=object)

    for j, ph in enumerate(phases):
        d = next((z for z in dyn if z["phase"] == ph), None)
        if d is None:
            for i in range(3):
                ax[i, j].axis("off")
            continue

        t = d["t_axis_s"]
        ax[0, j].plot(t, d["X_clip_cm"], lw=1.8, color="tab:blue", label="x(t) in [0,4]")
        ax[0, j].plot(t, d["x_hat_ll_cm"], lw=1.3, color="tab:cyan", alpha=0.9, label="x_hat from afferent")
        ax[0, j].plot(t, d["x_hat_ts_cm"], lw=1.3, color="tab:green", alpha=0.9, label="x_hat from torus")
        ax[0, j].plot(t, d["Y_cm"], lw=1.2, color="tab:orange", alpha=0.85, label="y(t)")
        ax[0, j].set_ylim(-0.2, 4.2)
        ax[0, j].set_ylabel("Position (cm)")
        ax[0, j].set_title(f"{ph.upper()} learning: sphere position vs time")
        ax[0, j].grid(alpha=0.3)
        ax[0, j].legend(fontsize=8)

        # Afferent activity (LL) vs time.
        ll_vmax = _robust_vmax(d["ll_t_rate"], q=99.5)
        im_ll = ax[1, j].imshow(
            d["ll_t_rate"],
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], 0, result["params"].n_ll - 1],
            vmin=0.0,
            vmax=ll_vmax,
            cmap="plasma",
        )
        ax[1, j].plot(t, d["ll_mean_rate_t"] / max(np.nanmax(d["ll_mean_rate_t"]), 1e-9) * (result["params"].n_ll - 1),
                      color="white", lw=1.0, alpha=0.9)
        ax[1, j].set_ylabel("LL index")
        ax[1, j].set_title("Afferent (LL) activity vs time")
        fig.colorbar(im_ll, ax=ax[1, j], label="LL rate (Hz)")

        # Torus activity vs time.
        ts_vmax = min(50.0, _robust_vmax(d["ts_t_rate"], q=99.5))
        im = ax[2, j].imshow(
            d["ts_t_rate"],
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], 0, result["params"].n_ts - 1],
            vmin=0.0,
            vmax=ts_vmax,
            cmap="viridis",
        )
        ax[2, j].set_ylabel("TS index")
        ax[2, j].set_title(
            f"TS index vs time (colormap)\n"
            f"spikes={d['ts_spike_count']}, mean={d['ts_mean_rate_hz']:.2f} Hz/neuron"
        )
        ax[2, j].set_xlabel("time (s)")
        fig.colorbar(im, ax=ax[2, j], label="TS rate (Hz)")

    fig.tight_layout()
    path = out_dir / f"stageD_dynamic_time_activity_{tag}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Stage D: STDP learning and torus map visualization over distance/noise.")
    parser.add_argument("--trials", type=int, default=300, help="Training trials before testing.")
    parser.add_argument("--tag", type=str, default="default", help="Output tag.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--mon-global-inh-mv", type=float, default=None, help="Override MON global inhibition strength (mV).")
    parser.add_argument("--ll-mon-topo", type=float, default=None, help="Override LL->MON topography strength.")
    parser.add_argument("--mon-ts-topo", type=float, default=None, help="Override MON->TS topography strength.")
    parser.add_argument("--mon-ts-gain-mv", type=float, default=None, help="Override MON->TS gain (mV per weight).")
    parser.add_argument("--ts-lat-peak-mv", type=float, default=None, help="Override TS lateral inhibition peak (mV).")
    parser.add_argument("--ts-lat-radius", type=int, default=None, help="Override TS lateral inhibition radius.")
    parser.add_argument("--ts-feedback-drive-mv", type=float, default=None, help="Override TS->TS-inh drive strength (mV).")
    parser.add_argument("--ts-feedback-inh-mv", type=float, default=None, help="Override TS-inh->TS inhibition strength (mV).")
    parser.add_argument("--stdp-apre", type=float, default=None, help="Override STDP Apre on MON->TS.")
    parser.add_argument("--stdp-apost", type=float, default=None, help="Override STDP Apost on MON->TS.")
    parser.add_argument("--distances-cm", type=str, default="0.8,1.2,1.6,2.0", help="Comma-separated fixed Y distances.")
    parser.add_argument("--noise-levels", type=str, default="0.0,0.5,1.0", help="Comma-separated test noise scales.")
    parser.add_argument("--use-curriculum", action="store_true", help="Enable 3-stage curriculum training.")
    parser.add_argument("--curriculum-phase1-frac", type=float, default=0.35, help="Fraction of trials in clean fixed-distance stage.")
    parser.add_argument("--curriculum-phase2-frac", type=float, default=0.45, help="Fraction of trials in clean variable-distance stage.")
    parser.add_argument("--curriculum-fixed-distance-cm", type=float, default=1.2, help="Fixed near-field distance used in stage 1.")
    parser.add_argument("--curriculum-final-noise-scale", type=float, default=0.2, help="Noise scale in final curriculum stage.")
    parser.add_argument("--prior-boost-strength", type=float, default=0.25, help="Temporary initial topographic prior strength.")
    parser.add_argument("--prior-decay-fraction", type=float, default=0.35, help="Fraction of training over which prior decays to zero.")
    parser.add_argument("--homeo-eta", type=float, default=0.06, help="Homeostatic normalization step size on MON->TS incoming sums.")
    parser.add_argument("--homeo-every-trials", type=int, default=5, help="Apply homeostatic normalization every N trials.")
    parser.add_argument("--mon-ts-prior-scale", type=float, default=0.0, help="Fixed anatomical MON->TS somatotopic scaffold scale.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Picture/StageD",
        help="Directory where all figures/CSVs for this run are saved.",
    )
    args = parser.parse_args()

    distances = [float(x) for x in args.distances_cm.split(",") if x.strip()]
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
    if len(distances) == 0:
        raise ValueError("Need at least one distance.")
    if len(noise_levels) == 0:
        raise ValueError("Need at least one noise level.")

    p = apply_model_mode(NetworkParams(), "ll_thesis")
    # Stage-D defaults:
    # - Use modulation-only LL rates (baseline cancelled) as in thesis-style runs.
    # - Stronger LL->MON topography + stronger TS lateral inhibition to bootstrap map emergence.
    p = replace(
        p,
        ll_rate_mode="modulation",
        ll_rate_baseline_subtract_hz=0.0,
        ll_rate_gain=1.0,
        ll_to_mon_topography_strength=0.22,
        ts_local_inh_peak_mV=2.0,
        ts_lateral_radius=24,
        seed=args.seed,
    )
    if args.mon_global_inh_mv is not None:
        p = replace(p, global_inh_to_mon_mV=max(0.0, float(args.mon_global_inh_mv)))
    if args.ll_mon_topo is not None:
        p = replace(p, ll_to_mon_topography_strength=max(0.0, float(args.ll_mon_topo)))
    if args.mon_ts_topo is not None:
        p = replace(p, mon_to_ts_topography_strength=max(0.0, float(args.mon_ts_topo)))
    if args.mon_ts_gain_mv is not None:
        p = replace(p, mon_ts_gain_mV=max(0.0, float(args.mon_ts_gain_mv)))
    if args.ts_lat_peak_mv is not None:
        p = replace(p, ts_local_inh_peak_mV=max(0.0, float(args.ts_lat_peak_mv)))
    if args.ts_lat_radius is not None:
        p = replace(p, ts_lateral_radius=max(1, int(args.ts_lat_radius)))
    if args.stdp_apre is not None:
        p = replace(p, mon_ts_apre=float(args.stdp_apre))
    if args.stdp_apost is not None:
        p = replace(p, mon_ts_apost=float(args.stdp_apost))
    # Attach custom Stage-D attributes used by run_stage_d feedback inhibition.
    setattr(p, "ts_to_global_inh_drive_mV", 0.35 if args.ts_feedback_drive_mv is None else max(0.0, float(args.ts_feedback_drive_mv)))
    setattr(p, "global_inh_to_ts_mV", 0.8 if args.ts_feedback_inh_mv is None else max(0.0, float(args.ts_feedback_inh_mv)))

    result = run_stage_d(
        p,
        train_trials=args.trials,
        distances_cm=distances,
        noise_levels=noise_levels,
        seed=args.seed,
        use_curriculum=bool(args.use_curriculum),
        curriculum_phase1_frac=float(args.curriculum_phase1_frac),
        curriculum_phase2_frac=float(args.curriculum_phase2_frac),
        curriculum_fixed_distance_cm=float(args.curriculum_fixed_distance_cm),
        curriculum_final_noise_scale=float(args.curriculum_final_noise_scale),
        prior_boost_strength=float(args.prior_boost_strength),
        prior_decay_fraction=float(args.prior_decay_fraction),
        homeo_eta=float(args.homeo_eta),
        homeo_every_trials=int(max(1, args.homeo_every_trials)),
        mon_ts_prior_scale=float(args.mon_ts_prior_scale),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_no_noise = _save_no_noise_distance_map(result, out_dir, args.tag)
    map_noise = _save_noise_level_map(result, out_dir, args.tag, distance_for_noise_panel=distances[min(1, len(distances)-1)])
    csv_path = _save_metrics_csv(result, out_dir, args.tag)
    weight_path, weight_csv = _save_weight_stabilization_plot(result, out_dir, args.tag)
    compare_paths = _save_before_after_comparison_by_noise(result, out_dir, args.tag)
    time_panel = _save_before_after_time_panel(result, out_dir, args.tag, noise_for_panel=0.0)
    dynamic_time_panel = _save_dynamic_time_activity(result, out_dir, args.tag)

    ts_rate = result["pr_ts"].smooth_rate(width=20 * b2.ms) / b2.Hz
    mon_rate = result["pr_mon"].smooth_rate(width=20 * b2.ms) / b2.Hz

    print(f"Saved: {csv_path}")
    for pth in map_no_noise:
        print(f"Saved: {pth}")
    for pth in map_noise:
        print(f"Saved: {pth}")
    print(f"Saved: {weight_path}")
    print(f"Saved: {weight_csv}")
    for pth in compare_paths:
        print(f"Saved: {pth}")
    if time_panel is not None:
        print(f"Saved: {time_panel}")
    if dynamic_time_panel is not None:
        print(f"Saved: {dynamic_time_panel}")

    print(f"MON max population rate (20 ms smooth): {float(np.max(mon_rate)):.2f} Hz")
    print(f"TS max population rate (20 ms smooth): {float(np.max(ts_rate)):.2f} Hz")
    if result["stabilization_time_s"] is None:
        print("Estimated weight stabilization time: not reached")
    else:
        print(f"Estimated weight stabilization time: {result['stabilization_time_s']:.2f} s")
    print(
        "Weight summary: "
        f"mean_w_init={float(result['w_mean_series'][0]):.6f}, "
        f"mean_w_final={float(result['w_mean_series'][-1]):.6f}, "
        f"mean_dmeanw_dt={float(np.mean(np.abs(np.diff(result['w_mean_series']) / np.maximum(np.diff(result['checkpoint_t_s']), 1e-12)))) if result['checkpoint_t_s'].size>1 else 0.0:.6e}"
    )

    print("Condition metrics:")
    for r in sorted(result["test_results"], key=lambda z: (z["phase"], z["noise_scale"], z["distance_cm"])):
        print(
            f"{r['phase']}: noise={r['noise_scale']:.2f}, Y={r['distance_cm']:.2f} cm -> "
            f"sigma={r['pv_sigma_theta']:.4f}, valid={r['pv_valid_fraction']:.3f}"
        )
    print("Dynamic x-hat tracking (afferent vs torus):")
    for r in result.get("dynamic_results", []):
        print(f"{r['phase']}: corr(x_hat_ll, x_hat_ts)={r['x_hat_corr']:.4f}")

    # Save a compact run summary document for traceability.
    summary_path = out_dir / f"RUN_SUMMARY_{args.tag}.md"
    dyn_rows = {r["phase"]: r for r in result.get("dynamic_results", [])}
    before_corr = dyn_rows.get("before", {}).get("x_hat_corr", float("nan"))
    after_corr = dyn_rows.get("after", {}).get("x_hat_corr", float("nan"))
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Stage D Run Summary: {args.tag}\n\n")
        f.write("## Parameters\n")
        f.write(f"- trials: {args.trials}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- distances_cm: {args.distances_cm}\n")
        f.write(f"- noise_levels: {args.noise_levels}\n")
        f.write(f"- ll_mon_topography_strength: {p.ll_to_mon_topography_strength}\n")
        f.write(f"- mon_ts_topography_strength: {p.mon_to_ts_topography_strength}\n")
        f.write(f"- mon_ts_gain_mV: {p.mon_ts_gain_mV}\n")
        f.write(f"- stdp_apre: {p.mon_ts_apre}\n")
        f.write(f"- stdp_apost: {p.mon_ts_apost}\n")
        f.write(f"- mon_global_inh_mV: {p.global_inh_to_mon_mV}\n")
        f.write(f"- ts_lateral_peak_mV: {p.ts_local_inh_peak_mV}\n")
        f.write(f"- ts_lateral_radius: {p.ts_lateral_radius}\n")
        f.write(
            f"- ts_feedback_drive_mV: {getattr(p, 'ts_to_global_inh_drive_mV', float('nan'))}\n"
        )
        f.write(f"- ts_feedback_inh_mV: {getattr(p, 'global_inh_to_ts_mV', float('nan'))}\n")
        f.write(f"- use_curriculum: {result.get('use_curriculum')}\n")
        f.write(f"- curriculum_phase1_frac: {result.get('curriculum_phase1_frac')}\n")
        f.write(f"- curriculum_phase2_frac: {result.get('curriculum_phase2_frac')}\n")
        f.write(f"- curriculum_fixed_distance_cm: {result.get('curriculum_fixed_distance_cm')}\n")
        f.write(f"- curriculum_final_noise_scale: {result.get('curriculum_final_noise_scale')}\n")
        f.write(f"- prior_boost_strength: {result.get('prior_boost_strength')}\n")
        f.write(f"- prior_decay_fraction: {result.get('prior_decay_fraction')}\n")
        f.write(f"- homeo_eta: {result.get('homeo_eta')}\n")
        f.write(f"- homeo_every_trials: {result.get('homeo_every_trials')}\n")
        f.write(f"- mon_ts_prior_scale: {result.get('mon_ts_prior_scale')}\n")
        f.write("\n## Key Results\n")
        f.write(f"- MON max population rate (20 ms): {float(np.max(mon_rate)):.4f} Hz\n")
        f.write(f"- TS max population rate (20 ms): {float(np.max(ts_rate)):.4f} Hz\n")
        if result["stabilization_time_s"] is None:
            f.write("- Estimated weight stabilization time: not reached\n")
        else:
            f.write(f"- Estimated weight stabilization time: {result['stabilization_time_s']:.4f} s\n")
        f.write(f"- Dynamic corr before: {before_corr:.6f}\n")
        f.write(f"- Dynamic corr after: {after_corr:.6f}\n")
        f.write("\n## Main Files\n")
        f.write(f"- {csv_path.name}\n")
        f.write(f"- {weight_path.name}\n")
        f.write(f"- {weight_csv.name}\n")
        if time_panel is not None:
            f.write(f"- {Path(time_panel).name}\n")
        if dynamic_time_panel is not None:
            f.write(f"- {Path(dynamic_time_panel).name}\n")
        for pth in map_no_noise + map_noise + compare_paths:
            f.write(f"- {Path(pth).name}\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
