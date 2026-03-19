import numpy as np
import matplotlib.pyplot as plt

from stimulus import StimulusParams, simulate_lateral_line


def main():
    # Thesis-like test sweep: 4 cm path at 5 cm/s, fixed distance.
    stim_params = StimulusParams()
    dt_s = 0.001
    speed_cm_s = 5.0
    path_cm = 4.0
    test_duration_s = path_cm / speed_cm_s

    sim = simulate_lateral_line(
        duration_s=test_duration_s,
        dt_s=dt_s,
        n_neuromasts=100,          # <- use 100 LL afferents explicitly
        seed=123,
        fixed_distance_cm=stim_params.mu_distance_cm,
        direction=1.0,
        fixed_speed_cm_s=speed_cm_s,
        params=stim_params,
    )

    rates = sim["rates_hz"]         # shape (T, n_ll)
    x = np.asarray(sim["X_cm"])     # shape (T,)

    n_ll = rates.shape[1]
    n_bins = 40
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tuning = np.zeros((n_ll, n_bins), dtype=float)

    for b in range(n_bins):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if not np.any(m):
            continue
        tuning[:, b] = rates[m, :].mean(axis=0)

    # Plot tuning of a subset of LL neurons.
    ll_indices = np.linspace(0, n_ll - 1, 10, dtype=int)
    fig, axes = plt.subplots(len(ll_indices), 1, figsize=(8, 2 * len(ll_indices)), sharex=True)
    if len(ll_indices) == 1:
        axes = [axes]

    for ax, j in zip(axes, ll_indices):
        ax.plot(centers, tuning[j], "-o")
        ax.set_ylabel(f"LL {j}")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("x (cm)")
    fig.suptitle("LL neuron tuning vs position (4 cm test sweep)", y=1.02)
    fig.tight_layout()
    out_path = "Picture/diagnose_ll_tuning.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print("Saved LL tuning figure:", out_path)


if __name__ == "__main__":
    main()