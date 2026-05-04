from pathlib import Path

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np


def save_test_phase_only_figure(result, out_path: Path):
    """
    Plot ONLY the test segment: time axis 0 .. test_duration_s (not full train+test).
    Lets you see LL / MON / TS during the continuous moving-sphere test without training clutter.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    sp_ll = result["sp_ll"]
    sp_mon = result["sp_mon"]
    sp_ts = result["sp_ts"]
    pr_mon = result["pr_mon"]
    pr_ts = result["pr_ts"]
    tsim = result["test_sim"]

    t_train = float(result["train_duration_s"])
    t_end = float(result["total_duration_s"])
    test_dur = float(result["test_duration_s"])

    # Stimulus trajectory during test (test_sim time base is already 0 .. test_dur).
    t_stim = np.asarray(tsim["t_s"], dtype=float)
    x_stim = np.asarray(tsim["X_cm"], dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(t_stim, x_stim, color="tab:red", lw=1.8)
    axes[0].set_ylabel("sphere X (cm)")
    axes[0].set_title("Test phase only — stimulus (continuous sweep)")
    axes[0].grid(alpha=0.3)

    # Spikes: map absolute time -> time since test start.
    ll_t = np.asarray(sp_ll.t / b2.second, dtype=float)
    ll_i = np.asarray(sp_ll.i, dtype=int)
    m = (ll_t >= t_train) & (ll_t < t_end)
    axes[1].scatter(ll_t[m] - t_train, ll_i[m], s=0.35, color="black")
    axes[1].set_ylabel("LL index")
    axes[1].set_title("LL spikes (test only)")
    axes[1].set_xlim(0.0, test_dur)
    axes[1].grid(alpha=0.3)

    mon_subset = min(400, p.n_mon)
    mt = np.asarray(sp_mon.t / b2.second, dtype=float)
    mi = np.asarray(sp_mon.i, dtype=int)
    m = (mt >= t_train) & (mt < t_end) & (mi < mon_subset)
    axes[2].scatter(mt[m] - t_train, mi[m], s=0.35, color="tab:purple")
    axes[2].set_ylabel("MON index")
    axes[2].set_title(f"MON spikes (test only, first {mon_subset})")
    axes[2].set_xlim(0.0, test_dur)
    axes[2].grid(alpha=0.3)

    tt = np.asarray(sp_ts.t / b2.second, dtype=float)
    ti = np.asarray(sp_ts.i, dtype=int)
    m = (tt >= t_train) & (tt < t_end)
    axes[3].scatter(tt[m] - t_train, ti[m], s=0.6, color="tab:green")
    axes[3].set_ylabel("TS index")
    axes[3].set_title("TS spikes (test only)")
    axes[3].set_xlim(0.0, test_dur)
    axes[3].grid(alpha=0.3)

    t_abs_m = np.asarray(pr_mon.t / b2.second, dtype=float)
    r_m = np.asarray(pr_mon.smooth_rate(width=20 * b2.ms) / b2.Hz, dtype=float)
    t_abs_t = np.asarray(pr_ts.t / b2.second, dtype=float)
    r_t = np.asarray(pr_ts.smooth_rate(width=20 * b2.ms) / b2.Hz, dtype=float)
    mm = (t_abs_m >= t_train) & (t_abs_m < t_end)
    mt_ = (t_abs_t >= t_train) & (t_abs_t < t_end)
    axes[4].plot(t_abs_m[mm] - t_train, r_m[mm], label="MON", color="tab:purple", lw=1.2)
    axes[4].plot(t_abs_t[mt_] - t_train, r_t[mt_], label="TS", color="tab:green", lw=1.2)
    axes[4].set_xlabel("time since test start (s)")
    axes[4].set_ylabel("rate (Hz)")
    axes[4].set_title("Population rates (test only, 20 ms smooth)")
    axes[4].set_xlim(0.0, test_dur)
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    fig.suptitle(
        f"TEST ONLY (duration {test_dur:.3f} s) — train ends at {t_train:.3f} s in full run",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_ts_pop_rate_train_test_transition_figure(result: dict, out_path: Path):
    """
    TS population rate (20 ms smooth) in a window around train→test: [train-2s, train+test].
    Vertical line at train end.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pr_ts = result["pr_ts"]
    t_train = float(result["train_duration_s"])
    test_dur = float(result["test_duration_s"])
    t_lo = max(0.0, t_train - 2.0)
    t_hi = t_train + test_dur

    t_abs = np.asarray(pr_ts.t / b2.second, dtype=float)
    r_ts = np.asarray(pr_ts.smooth_rate(width=20 * b2.ms) / b2.Hz, dtype=float)
    m = (t_abs >= t_lo) & (t_abs <= t_hi)
    if not np.any(m):
        fig, ax = plt.subplots(1, 1, figsize=(9, 3.8))
        ax.text(0.5, 0.5, "no TS rate samples in window", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("TS population rate (Hz)")
        ax.set_title("TS population rate around train→test transition")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 3.8))
        ax.plot(t_abs[m], r_ts[m], color="tab:green", lw=1.2)
        ax.axvline(t_train, color="tab:red", ls="--", lw=1.2, label="train end")
        ax.set_xlim(t_lo, t_hi)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("TS population rate (Hz)")
        ax.set_title("TS population rate around train→test transition (20 ms smooth)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
