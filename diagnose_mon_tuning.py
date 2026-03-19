import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2

from ll_stdp_brian2 import NetworkParams, apply_model_mode, make_test_rates, run_spatial_two_stage_model


def main():
    # Start from thesis params but with STDP off so weights don't change.
    params = apply_model_mode(NetworkParams(), "ll_thesis")
    params.ll_mon_use_stdp = False
    params.mon_ts_apre = 0.0
    params.mon_ts_apost = 0.0

    # Run the full pipeline once to get MON spikes and test stimulus.
    result = run_spatial_two_stage_model(params)

    sp_mon = result["sp_mon"]
    tsim = result["test_sim"]
    t_train = result["train_duration_s"]

    t_test = np.asarray(tsim["t_s"], dtype=float)
    x_test = np.asarray(tsim["X_cm"], dtype=float)

    # Spike times relative to test start.
    mon_t = np.asarray(sp_mon.t / b2.second, dtype=float)
    mon_i = np.asarray(sp_mon.i, dtype=int)
    mtest = (mon_t >= t_train) & (mon_t <= result["total_duration_s"])
    mon_t = mon_t[mtest] - t_train
    mon_i = mon_i[mtest]

    if mon_t.size == 0:
        print("No MON spikes in test window.")
        return

    p = result["params"]
    dt = float(max(p.dt_s, 1e-4))
    n_t = t_test.size
    rates = np.zeros((n_t, p.n_mon), dtype=float)
    k = np.floor(mon_t / dt).astype(int)
    valid = (k >= 0) & (k < n_t) & (mon_i >= 0) & (mon_i < p.n_mon)
    if np.any(valid):
        np.add.at(rates, (k[valid], mon_i[valid]), 1.0 / dt)

    # Bin by position along the 4 cm test path.
    n_pos_bins = min(40, n_t)
    x_edges = np.linspace(x_test.min(), x_test.max(), n_pos_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    tuning = np.zeros((p.n_mon, n_pos_bins), dtype=float)
    for b in range(n_pos_bins):
        mb = (x_test >= x_edges[b]) & (x_test < x_edges[b + 1])
        if not np.any(mb):
            continue
        tuning[:, b] = rates[mb, :].mean(axis=0)

    mon_indices = np.linspace(0, p.n_mon - 1, 16, dtype=int)
    fig, axes = plt.subplots(len(mon_indices), 1, figsize=(8, 2.0 * len(mon_indices)), sharex=True)
    if len(mon_indices) == 1:
        axes = [axes]
    for ax, j in zip(axes, mon_indices):
        ax.plot(x_centers, tuning[j, :], "-o", ms=3)
        ax.set_ylabel(f"MON {j}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("position x during test (cm)")
    fig.suptitle("MON tuning curves vs x (test sweep, no STDP)", y=1.02)
    fig.tight_layout()
    out_path = "Picture/diagnose_mon_tuning.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print("Saved MON tuning figure:", out_path)


if __name__ == "__main__":
    main()