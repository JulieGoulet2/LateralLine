from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_mon_to_ts_weight_profile(result: dict, out_path: Path):
    """
    Diagnostic: average MON->TS synaptic weight vs TS index after learning.
    Helps detect edge attractors or collapsed maps (peaks at TS 0 / n_ts-1).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result.get("params", None)
    if p is None:
        return
    w = np.asarray(result.get("w_after", []), dtype=float)
    j = np.asarray(result.get("mon_to_ts_j", []), dtype=int)
    if w.size == 0 or j.size == 0:
        return

    n_ts = int(p.n_ts)
    avg_w = np.zeros(n_ts, dtype=float)
    sat_frac = np.zeros(n_ts, dtype=float)
    for ts_idx in range(n_ts):
        m = j == ts_idx
        if not np.any(m):
            avg_w[ts_idx] = 0.0
            sat_frac[ts_idx] = 0.0
            continue
        avg_w[ts_idx] = float(np.mean(w[m]))
        sat_frac[ts_idx] = float(np.mean(np.isclose(w[m], float(p.mon_ts_wmax))))

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.2))
    ax.plot(np.arange(n_ts), avg_w, lw=2.0, color="tab:blue")
    ax.set_xlabel("TS index")
    ax.set_ylabel("average MON->TS weight (dimensionless)")
    ax.set_title("MON->TS weight profile vs TS index (after learning)")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(np.arange(n_ts), sat_frac, lw=1.5, color="tab:red", alpha=0.7)
    ax2.set_ylabel("fraction at wmax")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_mon_to_ts_receptive_fields_figure(result: dict, out_path: Path, n_examples: int = 8):
    """
    Diagnostic: incoming MON->TS weights vs MON index for example TS neurons (receptive fields).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result.get("params", None)
    if p is None:
        return
    w = np.asarray(result.get("w_after", []), dtype=float)
    i_mon = np.asarray(result.get("mon_to_ts_i", []), dtype=int)
    j_ts = np.asarray(result.get("mon_to_ts_j", []), dtype=int)
    if w.size == 0 or i_mon.size == 0 or j_ts.size == 0:
        return
    if w.size != i_mon.size or w.size != j_ts.size:
        return

    n_ts = int(p.n_ts)
    n_mon = int(p.n_mon)
    n_plot = min(int(n_examples), n_ts)
    ts_pick = np.linspace(0, n_ts - 1, num=n_plot, dtype=int)

    ncols = 2
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes_flat):
        if idx >= n_plot:
            ax.set_visible(False)
            continue
        ts_idx = int(ts_pick[idx])
        m = j_ts == ts_idx
        if not np.any(m):
            ax.set_title(f"TS {ts_idx} (no synapses)")
            ax.text(0.5, 0.5, "no incoming", ha="center", va="center", transform=ax.transAxes)
            continue
        mon_idx = i_mon[m]
        w_sub = w[m]
        ax.scatter(mon_idx, w_sub, s=4, alpha=0.45, color="tab:blue", edgecolors="none")
        ax.set_xlim(-0.5, max(n_mon - 0.5, 0.5))
        ax.set_xlabel("MON index")
        ax.set_ylabel("MON->TS weight")
        ax.set_title(f"TS {ts_idx} (incoming synapses: {int(np.count_nonzero(m))})")
        ax.grid(alpha=0.3)

    fig.suptitle("MON->TS receptive fields (incoming weights)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_ll_mon_weights_figure(result: dict, out_path: Path, n_examples: int = 6):
    """
    Visualize LL->MON connectivity for a small set of MON neurons.

    For each selected MON neuron, plot LL index vs LL->MON weight (mV).
    This helps see whether LL->MON learning + topography produces peaked,
    position-selective input profiles.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ll_i = result["ll_to_mon_i"]
    ll_j = result["ll_to_mon_j"]
    ll_w = result["ll_to_mon_w_mV"]
    p = result["params"]

    n_mon = p.n_mon
    if n_mon <= 0 or ll_i.size == 0:
        return

    # Pick a few MON neurons spread across the array.
    mon_indices = np.linspace(0, n_mon - 1, num=min(n_examples, n_mon), dtype=int)

    fig, axes = plt.subplots(len(mon_indices), 1, figsize=(8, 2.2 * len(mon_indices)), sharex=True)
    if len(mon_indices) == 1:
        axes = [axes]

    for ax, mj in zip(axes, mon_indices):
        m = ll_j == mj
        if not np.any(m):
            ax.set_ylabel(f"MON {mj}")
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "no inputs", ha="center", va="center", transform=ax.transAxes)
            continue
        x = ll_i[m]
        w = ll_w[m]
        ax.stem(x, w, basefmt=" ")
        ax.set_ylabel(f"MON {mj}")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("LL index")
    fig.suptitle("LL→MON weights for example MON neurons", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
