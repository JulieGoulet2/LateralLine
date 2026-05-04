from pathlib import Path

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

from plots._helpers import _test_x_local_bins


def save_ts_tuning_figure(result, out_path: Path, n_examples: int = 16):
    """
    Diagnostic: TS tuning curves vs position during the 4 cm test sweep.
    Shows, for a subset of TS neurons, their firing rate as a function of x.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    sp_ts = result["sp_ts"]
    tsim = result["test_sim"]
    t_train = result["train_duration_s"]

    t_test = np.asarray(tsim["t_s"], dtype=float)
    x_test = np.asarray(tsim["X_cm"], dtype=float)

    # Spike times relative to test start (t_train).
    ts_t = np.asarray(sp_ts.t / b2.second, dtype=float)
    ts_i = np.asarray(sp_ts.i, dtype=int)
    mtest = (ts_t >= t_train) & (ts_t <= result["total_duration_s"])
    ts_t = ts_t[mtest] - t_train
    ts_i = ts_i[mtest]

    if ts_t.size == 0:
        return

    # Use the same time grid as test_sim.
    dt = float(max(p.dt_s, 1e-4))
    n_t = t_test.size
    rates = np.zeros((n_t, p.n_ts), dtype=float)
    k = np.floor(ts_t / dt).astype(int)
    valid = (k >= 0) & (k < n_t) & (ts_i >= 0) & (ts_i < p.n_ts)
    if np.any(valid):
        np.add.at(rates, (k[valid], ts_i[valid]), 1.0 / dt)

    n_pos_bins = min(40, n_t)
    x_local, ok_t, x_edges, x_centers, xtag = _test_x_local_bins(x_test, p, n_pos_bins)
    tuning = np.zeros((p.n_ts, n_pos_bins), dtype=float)
    for b in range(n_pos_bins):
        mb = ok_t & (x_local >= x_edges[b]) & (x_local < x_edges[b + 1])
        if not np.any(mb):
            continue
        tuning[:, b] = rates[mb, :].mean(axis=0)

    # Plot tuning for a subset of TS neurons.
    ts_indices = np.linspace(0, p.n_ts - 1, num=min(n_examples, p.n_ts), dtype=int)
    fig, axes = plt.subplots(len(ts_indices), 1, figsize=(8, 2.0 * len(ts_indices)), sharex=True)
    if len(ts_indices) == 1:
        axes = [axes]
    for ax, j in zip(axes, ts_indices):
        ax.plot(x_centers, tuning[j, :], "-o", ms=3)
        ax.set_ylabel(f"TS {j}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(f"position x during test ({xtag})")
    fig.suptitle("TS tuning curves vs x (test sweep)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_mon_tuning_examples_figure(result: dict, out_path: Path, n_examples: int = 8):
    """
    Example MON neurons: firing rate vs position x during the test sweep.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    sp_mon = result["sp_mon"]
    tsim = result["test_sim"]
    t_train = float(result["train_duration_s"])

    t_test = np.asarray(tsim["t_s"], dtype=float)
    x_test = np.asarray(tsim["X_cm"], dtype=float)

    mon_t = np.asarray(sp_mon.t / b2.second, dtype=float)
    mon_i = np.asarray(sp_mon.i, dtype=int)
    mtest = (mon_t >= t_train) & (mon_t <= float(result["total_duration_s"]))
    mon_t = mon_t[mtest] - t_train
    mon_i = mon_i[mtest]

    if mon_t.size == 0:
        return

    dt = float(max(p.dt_s, 1e-4))
    n_t = t_test.size
    rates = np.zeros((n_t, p.n_mon), dtype=float)
    k = np.floor(mon_t / dt).astype(int)
    valid = (k >= 0) & (k < n_t) & (mon_i >= 0) & (mon_i < p.n_mon)
    if not np.any(valid):
        return
    np.add.at(rates, (k[valid], mon_i[valid]), 1.0 / dt)

    n_pos_bins = min(40, n_t)
    x_local, ok_t, x_edges, x_centers, xtag = _test_x_local_bins(x_test, p, n_pos_bins)
    tuning = np.zeros((p.n_mon, n_pos_bins), dtype=float)
    for b in range(n_pos_bins):
        mb = ok_t & (x_local >= x_edges[b]) & (x_local < x_edges[b + 1])
        if not np.any(mb):
            continue
        tuning[:, b] = rates[mb, :].mean(axis=0)

    mon_indices = np.linspace(0, p.n_mon - 1, num=min(int(n_examples), p.n_mon), dtype=int)
    fig, axes = plt.subplots(len(mon_indices), 1, figsize=(8, 2.0 * len(mon_indices)), sharex=True)
    if len(mon_indices) == 1:
        axes = [axes]
    for ax, j in zip(axes, mon_indices):
        ax.plot(x_centers, tuning[j, :], "-o", ms=3, color="tab:purple")
        ax.set_ylabel(f"MON {j}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(f"position x during test ({xtag})")
    fig.suptitle("MON tuning curves vs x (test sweep, examples)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
