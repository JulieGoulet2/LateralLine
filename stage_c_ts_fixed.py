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
    make_test_rates,
    make_training_rates,
    pv_map_quality_from_ts_spikes,
)
from stimulus import StimulusParams

b2.prefs.codegen.target = "numpy"


def run_stage_c(params: NetworkParams):
    train_rates, train_samples, _ = make_training_rates(params)
    test_sim = make_test_rates(params)

    ll_rates = np.vstack([train_rates, test_sim["rates_hz"]])
    baseline_subtract_hz = float(max(0.0, params.ll_rate_baseline_subtract_hz))
    if params.ll_rate_mode == "modulation":
        baseline_subtract_hz += float(StimulusParams().r0_hz)
    elif params.ll_rate_mode != "raw":
        raise ValueError(f"Unknown ll_rate_mode '{params.ll_rate_mode}'")
    ll_rates = np.clip((ll_rates - baseline_subtract_hz) * float(max(0.0, params.ll_rate_gain)), 0.0, None)

    train_duration_s = params.n_training_trials * params.trial_duration_s
    test_duration_s = float(test_sim["t_s"][-1] + params.dt_s)
    total_duration_s = train_duration_s + test_duration_s

    b2.start_scope()
    b2.defaultclock.dt = params.dt_s * b2.second

    ll_ta = b2.TimedArray(ll_rates * b2.Hz, dt=params.dt_s * b2.second)
    ll = b2.PoissonGroup(params.n_ll, rates="ll_ta(t, i)", namespace={"ll_ta": ll_ta})

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

    mon.v = "El + rand()*2*mV"
    ts.v = "El + rand()*2*mV"
    mon_inh.v = "El"

    # LL -> MON fixed projection
    s_ll_mon = b2.Synapses(ll, mon, "w : volt", on_pre="ge_post += w")
    if params.ll_to_mon_topography_strength <= 0.0:
        s_ll_mon.connect(p=params.p_ll_to_mon)
    else:
        ll_i, mon_j = build_ll_to_mon_indices(
            n_ll=params.n_ll,
            n_mon=params.n_mon,
            in_degree=params.ll_to_mon_in_degree,
            sigma_ll=params.ll_to_mon_sigma,
            topography_strength=params.ll_to_mon_topography_strength,
            seed=params.seed + 31,
        )
        s_ll_mon.connect(i=ll_i, j=mon_j)
    s_ll_mon.w = f"{params.ll_mon_w_mean_mV}*mV + {params.ll_mon_w_jitter_mV}*mV*rand()"

    # MON -> TS fixed weak-topographic projection (NO STDP in Stage C)
    mon_i, ts_j, _ = build_mon_to_ts_indices(
        n_mon=params.n_mon,
        n_ts=params.n_ts,
        out_degree=params.mon_to_ts_out_degree,
        sigma_ts=params.mon_to_ts_sigma,
        topography_strength=params.mon_to_ts_topography_strength,
        seed=params.seed,
    )
    s_mon_ts = b2.Synapses(mon, ts, "w : volt", on_pre="ge_post += w")
    s_mon_ts.connect(i=mon_i, j=ts_j)
    s_mon_ts.w = f"{params.mon_ts_gain_mV * params.mon_ts_w_init}*mV + {params.mon_ts_gain_mV * params.mon_ts_w_jitter}*mV*rand()"

    # Background
    bg_mon = b2.PoissonGroup(params.n_mon, rates=params.bg_rate_mon_hz * b2.Hz)
    bg_ts = b2.PoissonGroup(params.n_ts, rates=params.bg_rate_ts_hz * b2.Hz)
    s_bg_mon = b2.Synapses(bg_mon, mon, on_pre=f"ge_post += {params.bg_w_mon_mV}*mV")
    s_bg_mon.connect(condition="i == j")
    s_bg_ts = b2.Synapses(bg_ts, ts, on_pre=f"ge_post += {params.bg_w_ts_mV}*mV")
    s_bg_ts.connect(condition="i == j")

    # MON global inhibition
    s_mon_to_inh = b2.Synapses(mon, mon_inh, on_pre=f"ge_post += {params.mon_to_global_inh_drive_mV}*mV")
    s_mon_to_inh.connect(p=params.mon_to_global_inh_p)
    s_inh_to_mon = b2.Synapses(mon_inh, mon, "w : volt", on_pre="gi_post += w")
    s_inh_to_mon.connect()
    s_inh_to_mon.w = params.global_inh_to_mon_mV * b2.mV

    # TS lateral inhibition
    src, dst, wlat = [], [], []
    rad = params.ts_lateral_radius
    for i in range(params.n_ts):
        lo = max(0, i - rad)
        hi = min(params.n_ts, i + rad + 1)
        for j in range(lo, hi):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            wlat.append(params.ts_local_inh_peak_mV * (1.0 - abs(i - j) / rad))
    s_ts_lat = b2.Synapses(ts, ts, "w : volt", on_pre="gi_post += w")
    s_ts_lat.connect(i=np.asarray(src), j=np.asarray(dst))
    s_ts_lat.w = np.asarray(wlat, dtype=float) * b2.mV

    # Monitors
    sp_mon = b2.SpikeMonitor(mon)
    sp_ts = b2.SpikeMonitor(ts)
    pr_mon = b2.PopulationRateMonitor(mon)
    pr_ts = b2.PopulationRateMonitor(ts)

    b2.run(total_duration_s * b2.second, report="text", namespace={})

    pv = pv_map_quality_from_ts_spikes(
        ts_spike_t_s=np.asarray(sp_ts.t / b2.second, dtype=float),
        ts_spike_i=np.asarray(sp_ts.i, dtype=int),
        n_ts=params.n_ts,
        test_t_s=np.asarray(test_sim["t_s"], dtype=float),
        test_x_cm=np.asarray(test_sim["X_cm"], dtype=float),
        lateral_line_len_cm=float(StimulusParams().lateral_line_length_cm),
        test_start_s=train_duration_s,
        dt_s=params.dt_s,
        n_pos_bins=min(100, params.n_ts),
    )

    return {
        "params": params,
        "train_samples": train_samples,
        "test_sim": test_sim,
        "sp_mon": sp_mon,
        "sp_ts": sp_ts,
        "pr_mon": pr_mon,
        "pr_ts": pr_ts,
        "pv": pv,
        "train_duration_s": train_duration_s,
        "total_duration_s": total_duration_s,
    }


def save_stage_c_outputs(result: dict, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"stageC_ts_fixed_summary_{tag}.png"
    csv_path = out_dir / f"stageC_ts_fixed_metrics_{tag}.csv"

    p = result["params"]
    sp_mon = result["sp_mon"]
    sp_ts = result["sp_ts"]
    pr_mon = result["pr_mon"]
    pr_ts = result["pr_ts"]
    pv = result["pv"]
    t_train = result["train_duration_s"]
    t_end = result["total_duration_s"]

    fig, ax = plt.subplots(4, 1, figsize=(11, 12))

    mon_subset = min(400, p.n_mon)
    mm = sp_mon.i < mon_subset
    ax[0].scatter(sp_mon.t[mm] / b2.second, sp_mon.i[mm], s=0.25, color="tab:purple")
    ax[0].axvline(t_train, color="tab:red", ls="--", lw=1.0)
    ax[0].set_xlim(0, t_end)
    ax[0].set_title(f"Stage C MON raster (first {mon_subset})")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("MON index")

    ax[1].scatter(sp_ts.t / b2.second, sp_ts.i, s=0.5, color="tab:green")
    ax[1].axvline(t_train, color="tab:red", ls="--", lw=1.0)
    ax[1].set_xlim(0, t_end)
    ax[1].set_title("Stage C TS raster (fixed MON->TS)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("TS index")

    ax[2].plot(pr_mon.t / b2.second, pr_mon.smooth_rate(width=20 * b2.ms) / b2.Hz, label="MON")
    ax[2].plot(pr_ts.t / b2.second, pr_ts.smooth_rate(width=20 * b2.ms) / b2.Hz, label="TS")
    ax[2].axvline(t_train, color="tab:red", ls="--", lw=1.0)
    ax[2].set_xlim(0, t_end)
    ax[2].set_title("Population rates")
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("rate (Hz)")
    ax[2].grid(alpha=0.3)
    ax[2].legend()

    valid = np.isfinite(pv["theta_hat"])
    ax[3].plot(pv["theta_true"][valid], pv["theta_hat"][valid], ".", ms=2, alpha=0.5)
    lim = [0.0, 2.0 * np.pi]
    ax[3].plot(lim, lim, "r--", lw=1.2)
    ax[3].set_xlim(*lim)
    ax[3].set_ylim(*lim)
    ax[3].set_title(
        f"PV decoding (fixed TS map): sigma={pv['sigma_theta']:.3f}, "
        f"delta_trial={pv['delta_trial']:.3f}, valid={pv['valid_fraction']:.3f}"
    )
    ax[3].set_xlabel("theta true (rad)")
    ax[3].set_ylabel("theta decoded (rad)")
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"pv_sigma_theta,{float(pv['sigma_theta']):.8f}\n")
        f.write(f"pv_delta_trial,{float(pv['delta_trial']):.8f}\n")
        f.write(f"pv_valid_fraction,{float(pv['valid_fraction']):.8f}\n")
        ts_rate = pr_ts.smooth_rate(width=20 * b2.ms) / b2.Hz
        mon_rate = pr_mon.smooth_rate(width=20 * b2.ms) / b2.Hz
        f.write(f"mon_max_rate_hz,{float(np.max(mon_rate)):.8f}\n")
        f.write(f"ts_max_rate_hz,{float(np.max(ts_rate)):.8f}\n")

    return fig_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Stage C: fixed MON->TS + TS lateral inhibition (no STDP).")
    parser.add_argument("--trials", type=int, default=200, help="Training trials before fixed-map test.")
    parser.add_argument("--tag", type=str, default="default", help="Output tag.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--ts-lat-peak-mv", type=float, default=None, help="Override TS lateral inhibition peak (mV).")
    parser.add_argument("--ts-lat-radius", type=int, default=None, help="Override TS lateral inhibition radius.")
    args = parser.parse_args()

    p = apply_model_mode(NetworkParams(), "ll_thesis")
    p = replace(
        p,
        n_training_trials=max(1, args.trials),
        seed=args.seed,
    )
    if args.ts_lat_peak_mv is not None:
        p = replace(p, ts_local_inh_peak_mV=max(0.0, float(args.ts_lat_peak_mv)))
    if args.ts_lat_radius is not None:
        p = replace(p, ts_lateral_radius=max(1, int(args.ts_lat_radius)))

    result = run_stage_c(p)
    out_dir = Path("Picture/StageC")
    fig_path, csv_path = save_stage_c_outputs(result, out_dir, args.tag)

    pv = result["pv"]
    ts_rate = result["pr_ts"].smooth_rate(width=20 * b2.ms) / b2.Hz
    mon_rate = result["pr_mon"].smooth_rate(width=20 * b2.ms) / b2.Hz
    print(f"Saved: {fig_path}")
    print(f"Saved: {csv_path}")
    print(f"MON max population rate (20 ms smooth): {float(np.max(mon_rate)):.2f} Hz")
    print(f"TS max population rate (20 ms smooth): {float(np.max(ts_rate)):.2f} Hz")
    print(
        "PV (fixed map): "
        f"sigma_theta={pv['sigma_theta']:.4f}, "
        f"delta_trial={pv['delta_trial']:.4f}, "
        f"valid_fraction={pv['valid_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
