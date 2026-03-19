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
    make_training_rates,
)
from stimulus import StimulusParams

b2.prefs.codegen.target = "numpy"


def run_stage_b(params: NetworkParams):
    train_rates, train_samples, _ = make_training_rates(params)

    ll_rates = train_rates.copy()
    baseline_subtract_hz = float(max(0.0, params.ll_rate_baseline_subtract_hz))
    if params.ll_rate_mode == "modulation":
        baseline_subtract_hz += float(StimulusParams().r0_hz)
    elif params.ll_rate_mode != "raw":
        raise ValueError(f"Unknown ll_rate_mode '{params.ll_rate_mode}'")
    ll_rates = np.clip((ll_rates - baseline_subtract_hz) * float(max(0.0, params.ll_rate_gain)), 0.0, None)

    total_train_s = params.n_training_trials * params.trial_duration_s

    b2.start_scope()
    b2.defaultclock.dt = params.dt_s * b2.second

    rates_ta = b2.TimedArray(ll_rates * b2.Hz, dt=params.dt_s * b2.second)
    ll = b2.PoissonGroup(params.n_ll, rates="rates_ta(t, i)", namespace={"rates_ta": rates_ta})

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

    mon = b2.NeuronGroup(
        params.n_mon,
        eqs,
        threshold="v > Vth",
        reset="v = Vreset",
        refractory="tau_ref",
        method="euler",
        namespace=ns,
    )
    mon_inh = b2.NeuronGroup(
        1,
        eqs,
        threshold="v > Vth",
        reset="v = Vreset",
        refractory="tau_ref",
        method="euler",
        namespace=ns,
    )
    mon.v = "El + rand()*2*mV"
    mon_inh.v = "El"

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

    bg_mon = b2.PoissonGroup(params.n_mon, rates=params.bg_rate_mon_hz * b2.Hz)
    s_bg_mon = b2.Synapses(bg_mon, mon, on_pre=f"ge_post += {params.bg_w_mon_mV}*mV")
    s_bg_mon.connect(condition="i == j")

    s_mon_to_inh = b2.Synapses(mon, mon_inh, on_pre=f"ge_post += {params.mon_to_global_inh_drive_mV}*mV")
    s_mon_to_inh.connect(p=params.mon_to_global_inh_p)
    s_inh_to_mon = b2.Synapses(mon_inh, mon, "w : volt", on_pre="gi_post += w")
    s_inh_to_mon.connect()
    s_inh_to_mon.w = params.global_inh_to_mon_mV * b2.mV

    sp_ll = b2.SpikeMonitor(ll)
    sp_mon = b2.SpikeMonitor(mon)
    pr_mon = b2.PopulationRateMonitor(mon)

    ckpt_t, ckpt_rate, ckpt_active = [0.0], [0.0], [0.0]
    prev_spikes = 0
    prev_counts = np.array(sp_mon.count[:], dtype=int, copy=True)

    for k in range(params.n_training_trials):
        b2.run(params.trial_duration_s * b2.second)

        if (k + 1) % max(1, params.checkpoint_trials) == 0 or (k + 1) == params.n_training_trials:
            t_cur = (k + 1) * params.trial_duration_s
            dt_ckpt = params.trial_duration_s * max(1, params.checkpoint_trials)
            d_spikes = sp_mon.num_spikes - prev_spikes
            prev_spikes = sp_mon.num_spikes

            cnt = np.array(sp_mon.count[:], dtype=int, copy=True)
            dcnt = cnt - prev_counts
            prev_counts = cnt

            mon_rate_hz = d_spikes / max(1e-9, params.n_mon * dt_ckpt)
            active_frac = float(np.mean(dcnt > 0))
            ckpt_t.append(t_cur)
            ckpt_rate.append(mon_rate_hz)
            ckpt_active.append(active_frac)

    return {
        "params": params,
        "train_samples": train_samples,
        "sp_ll": sp_ll,
        "sp_mon": sp_mon,
        "pr_mon": pr_mon,
        "checkpoint_t_s": np.asarray(ckpt_t),
        "checkpoint_mon_rate_hz": np.asarray(ckpt_rate),
        "checkpoint_mon_active_frac": np.asarray(ckpt_active),
        "total_train_s": total_train_s,
    }


def save_stage_b_outputs(result: dict, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"stageB_mon_summary_{tag}.png"
    csv_path = out_dir / f"stageB_mon_metrics_{tag}.csv"

    p = result["params"]
    sp_mon = result["sp_mon"]
    pr_mon = result["pr_mon"]
    t_end = result["total_train_s"]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    tx = np.array([s["t_start_s"] for s in result["train_samples"]], dtype=float)
    dy = np.array([s["D_cm"] for s in result["train_samples"]], dtype=float)
    ax[0].plot(tx, dy, lw=1.1, color="tab:blue")
    ax[0].set_title("Stage B: training distance schedule (near-field)")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("distance Y (cm)")
    ax[0].grid(alpha=0.3)

    subset = min(500, p.n_mon)
    m = sp_mon.i < subset
    ax[1].scatter(sp_mon.t[m] / b2.second, sp_mon.i[m], s=0.4, color="tab:purple")
    ax[1].set_xlim(0, t_end)
    ax[1].set_title(f"MON raster (first {subset})")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("MON index")

    tck = result["checkpoint_t_s"]
    rck = result["checkpoint_mon_rate_hz"]
    ack = result["checkpoint_mon_active_frac"]
    ax[2].plot(pr_mon.t / b2.second, pr_mon.smooth_rate(width=20 * b2.ms) / b2.Hz, lw=1.0, alpha=0.6, label="MON pop rate")
    ax[2].plot(tck, rck, "o-", lw=1.6, label="ckpt mean MON rate")
    ax2 = ax[2].twinx()
    ax2.plot(tck, ack, "s--", lw=1.3, color="tab:red", label="ckpt active fraction")
    ax[2].set_xlim(0, t_end)
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("rate (Hz)")
    ax2.set_ylabel("active fraction")
    ax[2].grid(alpha=0.3)
    ax[2].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("checkpoint_t_s,mon_rate_hz,mon_active_fraction\n")
        for t, r, a in zip(tck, rck, ack):
            f.write(f"{float(t):.6f},{float(r):.6f},{float(a):.6f}\n")

    return fig_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Stage B: LL->MON expansion with global MON inhibition only.")
    parser.add_argument("--trials", type=int, default=200, help="Number of training trials.")
    parser.add_argument("--tag", type=str, default="default", help="Output tag.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = parser.parse_args()

    p = apply_model_mode(NetworkParams(), "ll_thesis")
    p = replace(
        p,
        n_training_trials=max(1, args.trials),
        seed=args.seed,
        # Stage B focuses MON sparsity/gain-control only.
        n_ts=1,
        mon_to_ts_out_degree=1,
    )

    result = run_stage_b(p)
    out_dir = Path("Picture/StageB")
    fig_path, csv_path = save_stage_b_outputs(result, out_dir, args.tag)

    mon_pop_rate = result["pr_mon"].smooth_rate(width=20 * b2.ms) / b2.Hz
    print(f"Saved: {fig_path}")
    print(f"Saved: {csv_path}")
    print(f"MON max population rate (20 ms smooth): {float(np.max(mon_pop_rate)):.2f} Hz")
    print(
        "Checkpoint summary: "
        f"rate_mean={float(np.mean(result['checkpoint_mon_rate_hz'][1:])):.3f} Hz, "
        f"active_frac_mean={float(np.mean(result['checkpoint_mon_active_frac'][1:])):.3f}"
    )


if __name__ == "__main__":
    main()
