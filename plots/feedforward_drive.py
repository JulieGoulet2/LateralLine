from pathlib import Path

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

from plots._helpers import _test_x_local_bins


def save_mon_ts_feedforward_drive_figures(result: dict, heatmap_path: Path, winner_path: Path):
    """
    Diagnostic: estimated MON->TS feedforward drive during test per x-bin.
    For each x-bin b and synapse (i->j, w): contribution += spike_count_MON[i,b] * w * mon_ts_gain_mV.
    """
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    w = np.asarray(result["w_after"], dtype=float)
    i_mon = np.asarray(result["mon_to_ts_i"], dtype=int)
    j_ts = np.asarray(result["mon_to_ts_j"], dtype=int)
    if w.size == 0 or w.size != i_mon.size or w.size != j_ts.size:
        return

    test_sim = result["test_sim"]
    sp_mon = result["sp_mon"]
    t_train = float(result["train_duration_s"])
    t_test_end = float(result["train_duration_s"] + result["test_duration_s"])
    x_test = np.asarray(test_sim["X_cm"], dtype=float)
    t_test = np.asarray(test_sim["t_s"], dtype=float)
    n_t = int(t_test.size)
    dt = float(max(p.dt_s, 1e-4))

    n_pos_bins = min(40, n_t)
    x_local_series, ok_t, x_edges, x_centers, xtag = _test_x_local_bins(x_test, p, n_pos_bins)

    spike_count = np.zeros((p.n_mon, n_pos_bins), dtype=float)
    mon_t_abs = np.asarray(sp_mon.t / b2.second, dtype=float)
    mon_i = np.asarray(sp_mon.i, dtype=int)
    m = (mon_t_abs >= t_train) & (mon_t_abs < t_test_end)
    mon_t_abs = mon_t_abs[m]
    mon_i = mon_i[m]
    if mon_t_abs.size > 0:
        mon_t_rel = mon_t_abs - t_train
        k = np.floor(mon_t_rel / dt).astype(int)
        k = np.clip(k, 0, n_t - 1)
        x_loc = x_local_series[k]
        bin_idx = np.searchsorted(x_edges, x_loc, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, n_pos_bins - 1)
        valid = (mon_i >= 0) & (mon_i < p.n_mon) & ok_t[k]
        np.add.at(spike_count, (mon_i[valid], bin_idx[valid]), 1.0)

    g = float(p.mon_ts_gain_mV)
    n_ts = int(p.n_ts)
    drive = np.zeros((n_ts, n_pos_bins), dtype=float)
    for s in range(w.size):
        drive[j_ts[s], :] += spike_count[i_mon[s], :] * float(w[s]) * g

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    vmax = float(np.max(drive)) if drive.size else 1.0
    vmax = max(vmax, 1e-12)
    im = ax.imshow(
        drive,
        aspect="auto",
        origin="lower",
        extent=(x_edges[0], x_edges[-1], -0.5, n_ts - 0.5),
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_xlabel(f"x during test ({xtag})")
    ax.set_ylabel("TS index")
    ax.set_title("MON→TS feedforward drive (test, Σ_i spikes_i·w_ij·gain)")
    plt.colorbar(im, ax=ax, label="drive (mV · spike count per bin)")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    winner = np.argmax(drive, axis=0).astype(int)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4.2))
    ax2.plot(x_centers, winner, "o-", ms=4, color="tab:green")
    ax2.set_xlabel(f"x during test ({xtag})")
    ax2.set_ylabel("winning TS index (max drive)")
    ax2.set_title("MON→TS feedforward drive winner vs x (test)")
    ax2.set_ylim(-0.5, n_ts - 0.5)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(winner_path, dpi=180)
    plt.close(fig2)
