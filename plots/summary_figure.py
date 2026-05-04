from pathlib import Path

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np


def save_summary_figure(result, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    sp_ll = result["sp_ll"]
    sp_mon = result["sp_mon"]
    sp_ts = result["sp_ts"]
    pr_mon = result["pr_mon"]
    pr_ts = result["pr_ts"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    ax = axes.ravel()

    # Stimulus overview.
    x_line = np.linspace(0.0, 4.0, p.n_ll)
    ax[0].plot(x_line, np.zeros_like(x_line), "k.", ms=2, label="Lateral line")

    tx = np.array([s["X_cm"] for s in result["train_samples"]])
    ty = np.array([s["D_cm"] for s in result["train_samples"]])
    ax[0].scatter(tx, ty, s=7, alpha=0.25, color="tab:blue", label="Training snapshots")

    tsim = result["test_sim"]
    ax[0].plot(tsim["X_cm"], tsim["Y_cm"], color="tab:red", lw=2.0, label="Test (4 cm)")
    ax[0].set_title("Hydrodynamic stimuli")
    ax[0].set_xlabel("x (cm)")
    ax[0].set_ylabel("y (cm)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)

    t_train = result["train_duration_s"]
    t_end = result["total_duration_s"]

    # LL spikes.
    ax[1].scatter(sp_ll.t / b2.second, sp_ll.i, s=0.3, color="black")
    ax[1].axvline(t_train, color="tab:red", ls="--", lw=1.0)
    ax[1].set_xlim(0, t_end)
    ax[1].set_title("LL spikes")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("LL index")

    # MON spikes subset.
    mon_subset = min(400, p.n_mon)
    m = sp_mon.i < mon_subset
    ax[2].scatter((sp_mon.t[m] / b2.second), sp_mon.i[m], s=0.3, color="tab:purple")
    ax[2].axvline(t_train, color="tab:red", ls="--", lw=1.0)
    ax[2].set_xlim(0, t_end)
    ax[2].set_title(f"MON spikes (first {mon_subset})")
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("MON index")

    # TS spikes.
    ax[3].scatter(sp_ts.t / b2.second, sp_ts.i, s=0.5, color="tab:green")
    ax[3].axvline(t_train, color="tab:red", ls="--", lw=1.0)
    ax[3].set_xlim(0, t_end)
    ax[3].set_title("TS spikes")
    ax[3].set_xlabel("time (s)")
    ax[3].set_ylabel("TS index")

    # Population rates.
    ax[4].plot(pr_mon.t / b2.second, pr_mon.smooth_rate(width=20 * b2.ms) / b2.Hz, label="MON")
    ax[4].plot(pr_ts.t / b2.second, pr_ts.smooth_rate(width=20 * b2.ms) / b2.Hz, label="TS")
    ax[4].axvline(t_train, color="tab:red", ls="--", lw=1.0)
    ax[4].set_xlim(0, t_end)
    ax[4].set_title("Population rates")
    ax[4].set_xlabel("time (s)")
    ax[4].set_ylabel("rate (Hz)")
    ax[4].legend()
    ax[4].grid(alpha=0.3)

    # MON->TS weight distribution.
    bins = np.linspace(0.0, p.mon_ts_wmax, 40)
    ax[5].hist(result["w_before"], bins=bins, alpha=0.55, label="before")
    ax[5].hist(result["w_after"], bins=bins, alpha=0.55, label="after")
    ax[5].set_title("MON->TS weights")
    ax[5].set_xlabel("weight")
    ax[5].set_ylabel("count")
    ax[5].legend()

    # Weight stabilization curve.
    ct = result["checkpoint_t_s"]
    wm = result["w_mean_series"]
    ws = result["w_std_series"]
    tw = result["tracked_weight_series"]

    ax[6].plot(ct, wm, lw=2.0, color="tab:blue", label="mean(w)")
    ax[6].fill_between(ct, wm - ws, wm + ws, color="tab:blue", alpha=0.2, label="mean±std")
    for k in range(min(8, tw.shape[1])):
        ax[6].plot(ct, tw[:, k], lw=0.8, alpha=0.5, color="tab:orange")
    if result["stabilization_time_s"] is not None:
        ax[6].axvline(result["stabilization_time_s"], color="tab:green", ls="--", lw=1.2, label="stabilized")
    ax[6].set_title("MON->TS stabilization")
    ax[6].set_xlabel("training time (s)")
    ax[6].set_ylabel("weight")
    ax[6].grid(alpha=0.3)
    ax[6].legend(fontsize=8)

    # PV-based map quality over training checkpoints.
    tpv = result["pv_ckpt_t_s"]
    s_th = result["pv_sigma_theta_series"]
    d_tr = result["pv_delta_trial_series"]
    if tpv.size > 0:
        ax[7].plot(tpv, s_th, lw=1.6, color="tab:blue", label="sigma_theta(t)")
        ax[7].plot(tpv, d_tr, lw=1.6, color="tab:orange", label="delta_trial(t)")
    ax[7].set_title(
        "PV over training "
        f"(final sigma_theta={result['pv_sigma_theta']:.3f}, delta_trial={result['pv_delta_trial']:.3f})"
    )
    ax[7].set_xlabel("training time (s)")
    ax[7].set_ylabel("error (rad)")
    ax[7].grid(alpha=0.3)
    ax[7].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
