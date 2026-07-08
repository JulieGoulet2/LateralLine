"""
Step 3 (Q3): does the learned somatotopic map generalise across TEST distance?

Overlays the distance-generalisation curve (sigma_theta and valid_fraction vs test
distance D) for two training protocols, using extract-mode distance sweeps:
  - Single-distance trained (D = 0.8 cm): distswp_topo020_seed{127..132}  (6 seeds)
  - Multi-distance  trained (D in [0.6,1.2]): distswp_multidist_seed{123..125}  (3 seeds)

Usage:
    /Users/juliegoulet/anaconda3/bin/python plots/distance_generalization.py
"""

from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BODY_LEN_CM = 4.0
DISTS = ["020", "040", "060", "080", "100", "120", "150", "200", "250", "300"]
GROUPS = {
    "single": dict(prefix="topo020", seeds=[127, 128, 129, 130, 131, 132],
                   color="#1976D2", label="Single-D training (D = 0.8 cm, 6 seeds)"),
    "multi":  dict(prefix="multidist", seeds=[123, 124, 125],
                   color="#E65100", label="Multi-D training (D ∈ [0.6, 1.2] cm, 3 seeds)"),
}


def collect(prefix, seeds, key):
    """Return (D_cm array, mean, sd) for the given metric across seeds per distance."""
    per_d = {int(d) / 100: [] for d in DISTS}
    for s in seeds:
        for d in DISTS:
            p = Path(f"Runs/distswp_{prefix}_seed{s}_d{d}/artifacts/seed_{s}_results.json")
            if not p.exists():
                continue
            per_d[int(d) / 100].append(json.load(open(p))[key])
    D = np.array(sorted(per_d))
    mean = np.array([np.mean(per_d[k]) if per_d[k] else np.nan for k in D])
    sd = np.array([np.std(per_d[k]) if per_d[k] else np.nan for k in D])
    return D, mean, sd


def main():
    out_dir = Path("Picture"); out_dir.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for g in GROUPS.values():
        for ax, key in [(ax1, "sigma_theta_rad"), (ax2, "valid_fraction")]:
            D, mean, sd = collect(g["prefix"], g["seeds"], key)
            ax.plot(D, mean, "-o", color=g["color"], ms=5, lw=1.8, label=g["label"])
            ax.fill_between(D, mean - sd, mean + sd, color=g["color"], alpha=0.18)
        # print table
        D, m, s = collect(g["prefix"], g["seeds"], "sigma_theta_rad")
        print(f"--- {g['label']} : sigma_theta vs test D ---")
        for dd, mm, ss in zip(D, m, s):
            print(f"    D={dd:.2f}  sigma={mm:.3f} ± {ss:.3f}")

    # training-distance markers
    for ax in (ax1, ax2):
        ax.axvspan(0.6, 1.2, color="#E65100", alpha=0.08, label="multi-D train range")
        ax.axvline(0.8, color="black", ls="--", lw=0.9, alpha=0.6, label="single-D train (0.8)")
        ax.set_xlabel("test distance D (cm)")
        ax.grid(alpha=0.3)

    ax1.axhline(np.pi / 2, color="gray", ls=":", lw=0.9, label="π/2 (chance)")
    ax1.set_ylabel(r"$\sigma_\theta$ (rad) — lower = sharper map")
    ax1.set_title("Map sharpness vs test distance")
    ax1.set_ylim(0, np.pi / 2 + 0.3)
    ax1.legend(fontsize=8, loc="upper left")

    ax2.set_ylabel("valid_fraction — higher = better")
    ax2.set_title("Decodable fraction vs test distance")
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Step 3 (Q3): distance generalisation — near-field band, widened by multi-distance training",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "distance_generalization.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
