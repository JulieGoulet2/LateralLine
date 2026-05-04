from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_learning_curves_figure(result: dict, out_path: Path):
    """Dedicated learning curves: MON->TS weights and PV quality over training."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ct = result["checkpoint_t_s"]
    wm = result["w_mean_series"]
    ws = result["w_std_series"]
    tw = result["tracked_weight_series"]
    wmad = result["w_mean_abs_delta_series"]
    tpv = result["pv_ckpt_t_s"]
    s_th = result["pv_sigma_theta_series"]
    d_tr = result["pv_delta_trial_series"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax[0].plot(ct, wm, lw=2.0, color="tab:blue", label="mean(w)")
    ax[0].fill_between(ct, wm - ws, wm + ws, color="tab:blue", alpha=0.2, label="mean±std")
    for k in range(min(12, tw.shape[1])):
        ax[0].plot(ct, tw[:, k], lw=0.8, alpha=0.45, color="tab:orange")
    ax0b = ax[0].twinx()
    ax0b.plot(ct, wmad, lw=1.6, ls="--", color="tab:red", label="mean|w-w0|")
    ax0b.set_ylabel("mean |delta w|", color="tab:red")
    ax0b.tick_params(axis="y", labelcolor="tab:red")
    ax[0].set_title("MON->TS weights over training")
    ax[0].set_xlabel("training time (s)")
    ax[0].set_ylabel("weight")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    if tpv.size > 0:
        ax[1].plot(tpv, s_th, lw=1.8, color="tab:blue", label="sigma_theta(t)")
        ax[1].plot(tpv, d_tr, lw=1.8, color="tab:orange", label="delta_trial(t)")
    ax[1].set_title("PV quality over training (lower is better)")
    ax[1].set_xlabel("training time (s)")
    ax[1].set_ylabel("error (rad)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
