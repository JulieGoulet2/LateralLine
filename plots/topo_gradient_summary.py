#!/usr/bin/env python3
"""
plots/topo_gradient_summary.py
------------------------------
Summary figure: sigma_theta and valid_fraction across topo levels.

All values for topo = 0.10 / 0.15 / 0.20 are extract-mode (load saved
weights, run test phase from fresh RNG state).
Values for topo = 0.40 / 0.60 / 0.80 are training-mode metrics logged
directly by run_multi_seed_safe.sh (seed_NNN_results.json).

Usage (from project root):
    python plots/topo_gradient_summary.py

Output:
    Picture/topo_gradient_summary.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------
# Per-seed data
# -----------------------------------------------------------------------

# topo = 0.10 — 10 seeds (123-132); extract-mode
# Seeds 124, 125 retrained post-B1 fix on 2026-05-17 (commit 3392ea8 onwards);
# other seeds are from the original (pre-B1) Y4 run. Mixed-RNG dataset.
sigma_010 = np.array([1.482, 0.775, 0.897, 0.915, 0.775, 0.699, 0.834, 0.765, 0.770, 1.108])
valid_010 = np.array([0.539, 0.868, 0.886, 0.860, 0.906, 0.884, 0.848, 0.889, 0.940, 0.844])

# topo = 0.15 — 10 seeds (123-132); extract-mode
sigma_015 = np.array([0.459, 0.580, 0.278, 0.583, 0.418, 0.398, 0.528, 0.477, 0.376, 0.453])
valid_015 = np.array([0.864, 0.880, 0.887, 0.896, 0.917, 0.903, 0.869, 0.916, 0.929, 0.864])

# topo = 0.20 — 10 seeds (123-132); extract-mode
sigma_020 = np.array([0.421, 0.401, 0.420, 0.323, 0.382, 0.336, 0.291, 0.293, 0.400, 0.267])
valid_020 = np.array([0.926, 0.888, 0.890, 0.902, 0.931, 0.891, 0.881, 0.960, 0.952, 0.901])

# topo = 0.40 — 10 seeds (123-132); training-mode (run_multi_seed_safe.sh)
sigma_040 = np.array([0.360, 0.344, 0.432, 0.375, 0.249, 0.304, 0.317, 0.317, 0.331, 0.272])
valid_040 = np.array([0.909, 0.925, 0.865, 0.909, 0.937, 0.818, 0.944, 0.925, 0.900, 0.929])

# topo = 0.60 — 10 seeds (123-132); training-mode (run_multi_seed_safe.sh)
sigma_060 = np.array([0.343, 0.346, 0.250, 0.278, 0.256, 0.247, 0.289, 0.294, 0.289, 0.269])
valid_060 = np.array([0.980, 0.985, 0.974, 0.971, 0.978, 0.974, 0.963, 0.957, 0.950, 0.973])

# topo = 0.80 — 10 seeds (123-132); training-mode (run_multi_seed_safe.sh)
sigma_080 = np.array([0.280, 0.280, 0.232, 0.327, 0.227, 0.297, 0.326, 0.281, 0.286, 0.240])
valid_080 = np.array([0.977, 0.993, 0.984, 0.987, 1.000, 0.980, 0.999, 0.987, 0.981, 0.940])

# -----------------------------------------------------------------------
# Build arrays for error-bar plots
# -----------------------------------------------------------------------
topos = [0.10, 0.15, 0.20, 0.40, 0.60, 0.80]
x_pos = np.array(topos)

all_sigma = [sigma_010, sigma_015, sigma_020, sigma_040, sigma_060, sigma_080]
all_valid = [valid_010, valid_015, valid_020, valid_040, valid_060, valid_080]

sigma_means = np.array([s.mean() for s in all_sigma])
sigma_sds   = np.array([s.std(ddof=1) for s in all_sigma])
valid_means = np.array([v.mean() for v in all_valid])
valid_sds   = np.array([v.std(ddof=1) for v in all_valid])

seed_counts = [10, 10, 10, 10, 10, 10]

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

SEED_COLOR = "#2196F3"   # blue for individual seeds
MEAN_COLOR = "#0D47A1"   # dark blue for mean ± SD

rng = np.random.default_rng(42)

def _scatter(ax, x_center, values, **kw):
    jitter = rng.uniform(-0.008, 0.008, len(values))
    ax.scatter(x_center + jitter, values, **kw)


# ============================================================
# LEFT panel — sigma_theta
# ============================================================
ax1.axhline(np.pi / 2, color="lightgray", linestyle=":", linewidth=0.8,
            label="π/2 (chance level)")

for topo, sigma in zip(topos, all_sigma):
    _scatter(ax1, topo, sigma, color=SEED_COLOR, alpha=0.4, s=22, zorder=2)

# Add label only once for legend
ax1.scatter([], [], color=SEED_COLOR, alpha=0.4, s=22, label="Individual seeds")

ax1.errorbar(
    x_pos, sigma_means, yerr=sigma_sds,
    fmt="o", color=MEAN_COLOR, markersize=9,
    capsize=6, linewidth=2, zorder=4, label="Mean ± SD",
)

ax1.set_xlabel("topo  (ll_mon_topo = mon_ts_topo)", fontsize=10)
ax1.set_ylabel("σ_θ  (rad)", fontsize=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(
    [f"{t:.2f}\n({n} seeds)" for t, n in zip(topos, seed_counts)],
    fontsize=8,
)
ax1.set_xlim(0.04, 0.88)
ax1.set_ylim(0, 1.7)
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(axis="y", alpha=0.3)
ax1.text(-0.12, 1.05, "A", transform=ax1.transAxes,
         fontsize=16, fontweight="bold", va="top", ha="left")

# ============================================================
# RIGHT panel — valid_fraction
# ============================================================
ax2.axhline(0.60, color="lightgray", linestyle=":", linewidth=0.8,
            label="Threshold (0.60)")

for topo, valid in zip(topos, all_valid):
    _scatter(ax2, topo, valid, color=SEED_COLOR, alpha=0.4, s=22, zorder=2)

ax2.scatter([], [], color=SEED_COLOR, alpha=0.4, s=22, label="Individual seeds")

ax2.errorbar(
    x_pos, valid_means, yerr=valid_sds,
    fmt="o", color=MEAN_COLOR, markersize=9,
    capsize=6, linewidth=2, zorder=4, label="Mean ± SD",
)

ax2.set_xlabel("topo  (ll_mon_topo = mon_ts_topo)", fontsize=10)
ax2.set_ylabel("valid_fraction", fontsize=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(
    [f"{t:.2f}\n({n} seeds)" for t, n in zip(topos, seed_counts)],
    fontsize=8,
)
ax2.set_xlim(0.04, 0.88)
ax2.set_ylim(0.40, 1.05)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(axis="y", alpha=0.3)
ax2.text(-0.12, 1.05, "B", transform=ax2.transAxes,
         fontsize=16, fontweight="bold", va="top", ha="left")

# ============================================================
# Save
# ============================================================
fig.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "Picture"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "topo_gradient_summary.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")

# Print summary statistics for reference
print("\nSummary statistics:")
for t, sm, ss, vm, vs, n in zip(topos, sigma_means, sigma_sds, valid_means, valid_sds, seed_counts):
    print(f"  topo={t:.2f} (N={n}): σ_θ = {sm:.3f} ± {ss:.3f},  valid = {vm:.3f} ± {vs:.3f}")

plt.close(fig)
