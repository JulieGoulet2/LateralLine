from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_multiseed_summary(results: list[dict], out_path: Path):
    """Save PV map-quality summary across seeds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sigma_theta = np.array([r["pv_sigma_theta"] for r in results], dtype=float)
    delta_trial = np.array([r["pv_delta_trial"] for r in results], dtype=float)
    x = np.arange(len(results))

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(x, sigma_theta, "o-", color="tab:blue", label="sigma_theta (somatotopic error)")
    ax.plot(x, delta_trial, "s-", color="tab:orange", label="delta_trial (trial variability)")
    ax.set_title("PV map quality across seeds (lower is better)")
    ax.set_xlabel("seed run index")
    ax.set_ylabel("error (rad)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
