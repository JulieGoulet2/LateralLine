# design_ll_mon_learning.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import brian2 as b2

from stimulus import StimulusParams, simulate_lateral_line


def build_ll_mon_indices(n_ll, n_mon, in_degree, sigma_ll, topo_strength, seed):
    rng = np.random.default_rng(seed)
    mon_pos = np.linspace(0.0, 1.0, n_mon)
    i_list = []
    j_list = []

    topo_strength = float(np.clip(topo_strength, 0.0, 1.0))
    for mon_idx in range(n_mon):
        n_topo = int(round(in_degree * topo_strength))
        n_rand = in_degree - n_topo
        parts = []

        if n_topo > 0:
            if sigma_ll >= n_ll:
                topo_sources = rng.integers(0, n_ll, size=n_topo)
            else:
                center = mon_pos[mon_idx] * (n_ll - 1)
                topo_sources = np.rint(rng.normal(center, sigma_ll, size=n_topo)).astype(int)
                topo_sources = np.clip(topo_sources, 0, n_ll - 1)
            parts.append(topo_sources)

        if n_rand > 0:
            parts.append(rng.integers(0, n_ll, size=n_rand))

        sources = np.concatenate(parts)
        rng.shuffle(sources)
        i_list.extend(sources.tolist())
        j_list.extend([mon_idx] * in_degree)

    return np.asarray(i_list), np.asarray(j_list)


def make_ll_snapshots(n_trials, trial_duration_s, dt_s, n_ll, seed, distance_cm=0.8):
    rng = np.random.default_rng(seed)
    stim_params = StimulusParams()
    xi = np.linspace(0.0, stim_params.lateral_line_length_cm, n_ll)

    total_s = n_trials * trial_duration_s
    n_steps = int(np.round(total_s / dt_s))

    rates = np.zeros((n_steps, n_ll), dtype=float)
    x_trace = np.zeros(n_steps, dtype=float)

    t = 0
    idx = 0
    while idx < n_steps:
        # One snapshot: sample x along lateral line, fixed distance and speed.
        x_cm = float(rng.uniform(0.0, stim_params.lateral_line_length_cm))
        sim = simulate_lateral_line(
            duration_s=trial_duration_s,
            dt_s=dt_s,
            n_neuromasts=n_ll,
            seed=seed + t,
            params=stim_params,
            fixed_distance_cm=distance_cm,
            direction=1.0,
            fixed_speed_cm_s=stim_params.mu_speed_cm_s,
        )
        # Take mean rate over time as a static snapshot for this trial.
        snap = sim["rates_hz"].mean(axis=0)
        snap = np.clip(snap - stim_params.r0_hz, 0.0, None)  # modulation only
        j0 = idx
        j1 = min(n_steps, idx + int(trial_duration_s / dt_s))
        rates[j0:j1, :] = snap[None, :]
        x_trace[j0:j1] = x_cm
        idx = j1
        t += 1

    return rates, x_trace


def run_ll_mon_learning(
    n_ll=100,
    n_mon=800,
    n_trials=200,
    trial_duration_s=1.0,
    dt_s=0.001,
    in_degree=10,
    topo_strength=0.3,
    sigma_ll=15.0,
    ll_mon_apre=0.01,
    ll_mon_apost=-0.0105,
    ll_mon_wmax_mV=14.0,
    ll_mon_w_init_mV=7.0,
    ll_mon_homeo_eta=0.04,
    ll_mon_homeo_every_trials=10,
    bg_rate_mon_hz=15.0,
    bg_w_mon_mV=1.5,
    mon_global_inh_mV=1.0,
    seed=123,
):
    # Build LL snapshots as static training.
    rates, x_trace = make_ll_snapshots(
        n_trials=n_trials,
        trial_duration_s=trial_duration_s,
        dt_s=dt_s,
        n_ll=n_ll,
        seed=seed,
        distance_cm=0.8,
    )
    total_s = n_trials * trial_duration_s

    b2.start_scope()
    b2.defaultclock.dt = dt_s * b2.second

    all_rates_ta = b2.TimedArray(rates * b2.Hz, dt=dt_s * b2.second)
    ll = b2.PoissonGroup(n_ll, rates="all_rates_ta(t, i)", namespace={"all_rates_ta": all_rates_ta})

    eqs = """
    dv/dt = (El - v + ge - gi) / tau_m : volt (unless refractory)
    dge/dt = -ge / tau_s : volt
    dgi/dt = -gi / tau_s : volt
    """
    ns = {
        "El": -74.0 * b2.mV,
        "Vth": -54.0 * b2.mV,
        "Vreset": -60.0 * b2.mV,
        "tau_ref": 2.0 * b2.ms,
        "tau_m": 10.0 * b2.ms,
        "tau_s": 2.0 * b2.ms,
    }
    mon = b2.NeuronGroup(n_mon, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)
    mon_inh = b2.NeuronGroup(1, eqs, threshold="v > Vth", reset="v = Vreset", refractory="tau_ref", method="euler", namespace=ns)
    mon.v = "El + rand()*2*mV"
    mon_inh.v = "El"

    # LL->MON STDP.
    ll_i_topo, mon_j_topo = build_ll_mon_indices(
        n_ll=n_ll,
        n_mon=n_mon,
        in_degree=in_degree,
        sigma_ll=sigma_ll,
        topo_strength=topo_strength,
        seed=seed + 31,
    )
    s_ll_mon = b2.Synapses(
        ll,
        mon,
        model="""
        w : volt
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        """,
        on_pre="""
        ge_post += w
        apre += Apre
        w = clip(w + apost*mV, 0*mV, wmax)
        """,
        on_post="""
        apost += Apost
        w = clip(w + apre*mV, 0*mV, wmax)
        """,
        namespace={
            "taupre": 20 * b2.ms,
            "taupost": 20 * b2.ms,
            "Apre": ll_mon_apre,
            "Apost": ll_mon_apost,
            "wmax": ll_mon_wmax_mV * b2.mV,
        },
    )
    s_ll_mon.connect(i=ll_i_topo, j=mon_j_topo)
    s_ll_mon.w = f"{ll_mon_w_init_mV}*mV + 2.0*mV*rand()"

    # Target incoming sum per MON.
    w0 = np.array(s_ll_mon.w[:] / b2.mV, dtype=float, copy=True)
    jj = np.array(s_ll_mon.j[:], dtype=int, copy=True)
    incoming0 = np.bincount(jj, weights=w0, minlength=n_mon)
    target_incoming = float(np.mean(incoming0[incoming0 > 0])) if np.any(incoming0 > 0) else float(np.mean(incoming0))

    # Background and global inhibition.
    bg_mon = b2.PoissonGroup(n_mon, rates=bg_rate_mon_hz * b2.Hz)
    s_bg_mon = b2.Synapses(bg_mon, mon, on_pre=f"ge_post += {bg_w_mon_mV}*mV")
    s_bg_mon.connect(condition="i == j")

    s_mon_to_inh = b2.Synapses(mon, mon_inh, on_pre="ge_post += 0.5*mV")
    s_mon_to_inh.connect(p=0.05)
    s_inh_to_mon = b2.Synapses(mon_inh, mon, "w : volt", on_pre="gi_post += w")
    s_inh_to_mon.connect()
    s_inh_to_mon.w = mon_global_inh_mV * b2.mV

    sp_ll = b2.SpikeMonitor(ll)
    sp_mon = b2.SpikeMonitor(mon)
    pr_mon = b2.PopulationRateMonitor(mon)

    n_steps = n_trials
    for k in range(n_steps):
        b2.run(trial_duration_s * b2.second)

        # Homeostatic normalization every ll_mon_homeo_every_trials trials.
        if ll_mon_homeo_eta > 0.0 and ((k + 1) % max(1, int(ll_mon_homeo_every_trials)) == 0):
            w = np.array(s_ll_mon.w[:] / b2.mV, dtype=float, copy=True)
            jj = np.array(s_ll_mon.j[:], dtype=int, copy=True)
            incoming = np.bincount(jj, weights=w, minlength=n_mon)
            scale = np.ones(n_mon, dtype=float)
            nonzero = incoming > 1e-12
            if np.any(nonzero):
                ratio = target_incoming / np.maximum(incoming[nonzero], 1e-12)
                scale[nonzero] = 1.0 + ll_mon_homeo_eta * (ratio - 1.0)
                scale = np.clip(scale, 0.9, 1.1)
                w = np.clip(w * scale[jj], 0.0, ll_mon_wmax_mV)
                s_ll_mon.w = w * b2.mV

    # Collect final weights.
    w_final = np.array(s_ll_mon.w[:] / b2.mV, dtype=float, copy=True)
    jj_final = np.array(s_ll_mon.j[:], dtype=int, copy=True)
    ii_final = np.array(s_ll_mon.i[:], dtype=int, copy=True)

    return {
        "sp_ll": sp_ll,
        "sp_mon": sp_mon,
        "pr_mon": pr_mon,
        "w_before": w0,
        "w_after": w_final,
        "ll_to_mon_i": ii_final,
        "ll_to_mon_j": jj_final,
        "target_incoming": target_incoming,
        "incoming0": incoming0,
        "incoming_final": np.bincount(jj_final, weights=w_final, minlength=n_mon),
        "total_duration_s": total_s,
        "n_ll": n_ll,
        "n_mon": n_mon,
    }


def plot_results(result: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # MON spikes + rate.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    sp = result["sp_mon"]
    pr = result["pr_mon"]
    axes[0].scatter(sp.t / b2.second, sp.i, s=0.3, color="tab:purple")
    axes[0].set_ylabel("MON index")
    axes[0].set_title("MON spikes over training (LL->MON STDP + homeostasis)")
    axes[1].plot(pr.t / b2.second, pr.smooth_rate(width=20 * b2.ms) / b2.Hz)
    axes[1].set_ylabel("MON rate (Hz)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ll_mon_mon_activity.png", dpi=180)
    plt.close(fig)

    # LL->MON weights for a few MON neurons, before vs after.
    n_mon = result["n_mon"]
    ii = result["ll_to_mon_i"]
    jj = result["ll_to_mon_j"]
    w0 = result["w_before"]
    w1 = result["w_after"]

    mon_indices = np.linspace(0, n_mon - 1, num=min(6, n_mon), dtype=int)
    fig, axes = plt.subplots(len(mon_indices), 1, figsize=(8, 2.4 * len(mon_indices)), sharex=True)
    if len(mon_indices) == 1:
        axes = [axes]

    for ax, mj in zip(axes, mon_indices):
        m = jj == mj
        if not np.any(m):
            ax.text(0.5, 0.5, f"MON {mj}: no inputs", ha="center", va="center", transform=ax.transAxes)
            continue
        x = ii[m]
        w_before = w0[m]
        w_after = w1[m]
        ax.stem(x, w_before, basefmt=" ", linefmt="C0-", markerfmt="C0o", use_line_collection=True, label="before")
        ax.stem(x, w_after, basefmt=" ", linefmt="C1-", markerfmt="C1^", use_line_collection=True, label="after")
        ax.set_ylabel(f"MON {mj}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("LL index")
    axes[0].legend(fontsize=8)
    fig.suptitle("LL→MON weights before vs after learning", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "ll_mon_weights_before_after.png", dpi=180)
    plt.close(fig)

def main():
    out_dir = Path("Picture/LL_MON_DESIGN")
    result = run_ll_mon_learning(
        n_ll=100,
        n_mon=800,
        n_trials=200,
        trial_duration_s=1.0,
        dt_s=0.001,
        in_degree=10,
        topo_strength=0.3,
        sigma_ll=15.0,
        ll_mon_apre=0.01,
        ll_mon_apost=-0.0105,
        ll_mon_wmax_mV=14.0,
        ll_mon_w_init_mV=7.0,
        ll_mon_homeo_eta=0.04,
        ll_mon_homeo_every_trials=10,
        bg_rate_mon_hz=80.0,      # was 15.0
        bg_w_mon_mV=4.0,          # was 1.5
        mon_global_inh_mV=0.0,    # turn OFF inhibition for design run
        seed=123,
    )
    plot_results(result, out_dir)
    main()