"""
Test-phase raster figures: spike index vs stimulus position x.

Helper module IMPORTED by ll_stdp_brian2.py (not run standalone). Each function takes
the `result` dict returned by run_spatial_two_stage_model (spike monitors + test-sweep
timing/position) and writes one scatter PNG into Runs/<run>/figures/. The three
functions are near-identical for the LL, MON, and TS layers. The TS raster is where the
"vertical bands" (multimodal tuning) are visible — see RESULTS.md §1.
"""

from pathlib import Path

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

from plots._helpers import _eval_window_cm


def save_ts_spikes_vs_x_test_figure(result: dict, out_path: Path):
    """
    Diagnostic scatter: TS index vs stimulus position x during the test window only.
    A somatotopic map appears as a diagonal band (not vertical stripes).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    sp_ts = result["sp_ts"]
    test_sim = result["test_sim"]
    train_duration_s = float(result["train_duration_s"])
    test_duration_s = float(result["test_duration_s"])
    t_test_end = train_duration_s + test_duration_s

    x_test = np.asarray(test_sim["X_cm"], dtype=float)
    dt = float(p.dt_s)

    ts_t_abs = np.asarray(sp_ts.t / b2.second, dtype=float)
    ts_i = np.asarray(sp_ts.i, dtype=int)

    m = (ts_t_abs >= train_duration_s) & (ts_t_abs < t_test_end)
    if not np.any(m):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
        ax.text(0.5, 0.5, "No TS spikes during test", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("stimulus x (cm, linear)")
        ax.set_ylabel("TS index")
        ax.set_title("TS spikes vs x during test window")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    ts_t_rel = ts_t_abs[m] - train_duration_s
    ts_i = ts_i[m]

    k = np.floor(ts_t_rel / dt).astype(int)
    valid = (k >= 0) & (k < x_test.size) & (ts_i >= 0) & (ts_i < p.n_ts)
    if not np.any(valid):
        print("No valid TS spikes after time-to-index mapping")
        return

    x_sp = x_test[k[valid]]
    ts_i = ts_i[valid]

    w = _eval_window_cm(p)
    if w is not None:
        emin, emax = w
        mwin = (x_sp >= emin) & (x_sp <= emax)
        if not np.any(mwin):
            fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
            ax.text(
                0.5,
                0.5,
                "No TS spikes in eval x window",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xlabel(f"eval x [{emin:.3f},{emax:.3f}] cm (local)")
            ax.set_ylabel("TS index")
            ax.set_title("TS spikes vs x during test window")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            return
        x_sp = x_sp[mwin] - emin
        ts_i = ts_i[mwin]
        xlab = f"x in eval window (local 0 at {emin:.3f} cm)"
    else:
        xlab = "stimulus x (cm, linear)"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    ax.scatter(x_sp, ts_i, s=2.0, alpha=0.4, color="tab:green")
    ax.set_xlabel(xlab)
    ax.set_ylabel("TS index")
    ax.set_title("TS spikes vs x during test window")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_mon_spikes_vs_x_test_figure(result: dict, out_path: Path):
    """
    Diagnostic scatter: MON index vs stimulus position x during the test window only.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    sp_mon = result["sp_mon"]
    test_sim = result["test_sim"]
    train_duration_s = float(result["train_duration_s"])
    test_duration_s = float(result["test_duration_s"])
    t_test_end = train_duration_s + test_duration_s

    x_test = np.asarray(test_sim["X_cm"], dtype=float)
    dt = float(p.dt_s)

    mon_t_abs = np.asarray(sp_mon.t / b2.second, dtype=float)
    mon_i = np.asarray(sp_mon.i, dtype=int)

    m = (mon_t_abs >= train_duration_s) & (mon_t_abs < t_test_end)
    if not np.any(m):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
        ax.text(0.5, 0.5, "No MON spikes during test", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("stimulus x (cm, linear)")
        ax.set_ylabel("MON index")
        ax.set_title("MON spikes vs x during test window")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    mon_t_rel = mon_t_abs[m] - train_duration_s
    mon_i = mon_i[m]

    k = np.floor(mon_t_rel / dt).astype(int)
    valid = (k >= 0) & (k < x_test.size) & (mon_i >= 0) & (mon_i < p.n_mon)
    if not np.any(valid):
        print("No valid MON spikes after time-to-index mapping")
        return

    x_sp = x_test[k[valid]]
    mon_i = mon_i[valid]

    w = _eval_window_cm(p)
    if w is not None:
        emin, emax = w
        mwin = (x_sp >= emin) & (x_sp <= emax)
        if not np.any(mwin):
            fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
            ax.text(
                0.5,
                0.5,
                "No MON spikes in eval x window",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xlabel(f"eval x [{emin:.3f},{emax:.3f}] cm (local)")
            ax.set_ylabel("MON index")
            ax.set_title("MON spikes vs x during test window")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            return
        x_sp = x_sp[mwin] - emin
        mon_i = mon_i[mwin]
        xlab = f"x in eval window (local 0 at {emin:.3f} cm)"
    else:
        xlab = "stimulus x (cm, linear)"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    ax.scatter(x_sp, mon_i, s=0.5, alpha=0.25, color="tab:purple")
    ax.set_xlabel(xlab)
    ax.set_ylabel("MON index")
    ax.set_title("MON spikes vs x during test window")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_ll_spikes_vs_x_test_figure(result: dict, out_path: Path):
    """
    Diagnostic scatter: LL afferent index vs stimulus position x during the test window only.
    LL is a pure Poisson source driven by the stimulus rate function — this plot shows the
    INPUT to the network and should be the same across runs with the same stimulus and seed.
    Useful as a baseline to verify input is identical when comparing downstream MON/TS results.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    sp_ll = result["sp_ll"]
    test_sim = result["test_sim"]
    train_duration_s = float(result["train_duration_s"])
    test_duration_s = float(result["test_duration_s"])
    t_test_end = train_duration_s + test_duration_s

    x_test = np.asarray(test_sim["X_cm"], dtype=float)
    dt = float(p.dt_s)

    ll_t_abs = np.asarray(sp_ll.t / b2.second, dtype=float)
    ll_i = np.asarray(sp_ll.i, dtype=int)

    m = (ll_t_abs >= train_duration_s) & (ll_t_abs < t_test_end)
    if not np.any(m):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
        ax.text(0.5, 0.5, "No LL spikes during test", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("stimulus x (cm, linear)")
        ax.set_ylabel("LL index")
        ax.set_title("LL spikes vs x during test window")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    ll_t_rel = ll_t_abs[m] - train_duration_s
    ll_i = ll_i[m]

    k = np.floor(ll_t_rel / dt).astype(int)
    valid = (k >= 0) & (k < x_test.size) & (ll_i >= 0) & (ll_i < p.n_ll)
    if not np.any(valid):
        print("No valid LL spikes after time-to-index mapping")
        return

    x_sp = x_test[k[valid]]
    ll_i = ll_i[valid]

    w = _eval_window_cm(p)
    if w is not None:
        emin, emax = w
        mwin = (x_sp >= emin) & (x_sp <= emax)
        if not np.any(mwin):
            return
        x_sp = x_sp[mwin] - emin
        ll_i = ll_i[mwin]
        xlab = f"x in eval window (local 0 at {emin:.3f} cm)"
    else:
        xlab = "stimulus x (cm, linear)"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    ax.scatter(x_sp, ll_i, s=0.5, alpha=0.35, color="tab:blue")
    ax.set_xlabel(xlab)
    ax.set_ylabel("LL index")
    ax.set_title("LL spikes vs x during test window")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
