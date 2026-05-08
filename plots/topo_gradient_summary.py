#!/usr/bin/env python3
"""
plots/topo_gradient_summary.py
------------------------------
Summary figure: sigma_theta and valid_fraction across topo levels.

All values are extract-mode (load saved weights, run test phase from
fresh RNG state).  Hardcoded from Logs/extract_evaluation.log (2026-05-08)
and earlier Y2/Y4 runs.

Usage (from project root):
    python plots/topo_gradient_summary.py

Output:
    Picture/topo_gradient_summary.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------
# Per-seed data — extract-mode
# Seeds 123–132 at topo=0.20 and topo=0.15 (Logs/extract_evaluation.log)
# Seeds 123–125 at topo=0.10 — aggregate only (Y4 run, earlier session)
# -----------------------------------------------------------------------

# topo = 0.20 — 10 seeds
sigma_020 = np.array([0.421, 0.401, 0.420, 0.323, 0.382, 0.336, 0.291, 0.293, 0.400, 0.267])
valid_020 = np.array([0.926, 0.888, 0.890, 0.902, 0.931, 0.891, 0.881, 0.960, 0.952, 0.901])

# topo = 0.15 — 10 seeds
sigma_015 = np.array([0.459, 0.580, 0.278, 0.583, 0.418, 0.398, 0.528, 0.477, 0.376, 0.453])
valid_015 = np.array([0.864, 0.880, 0.887, 0.896, 0.917, 0.903, 0.869, 0.916, 0.929, 0.864])

# topo = 0.10 — 3 seeds (aggregate from RESULTS.md)
sigma_010_mean, sigma_010_sd = 0.76, 0.15
valid_010_mean, valid_010_sd = 0.84, 0.01

# High-topo reference (single training run, topo = 0.80)
ref_sigma = 0.875
ref_valid = 0.660

# -----------------------------------------------------------------------
# Build arrays for error-bar plots
# -----------------------------------------------------------------------
topos = [0.10, 0.15, 0.20]
x_pos = np.array(topos)
x_ref = 0.80

sigma_means = np.array([sigma_010_mean, sigma_015.mean(), sigma_020.mean()])
sigma_sds = np.array([sigma_010_sd, sigma_015.std(ddof=1), sigma_020.std(ddof=1)])

valid_means = np.array([valid_010_mean, valid_015.mean(), valid_020.mean()])
valid_sds = np.array([valid_010_sd, valid_015.std(ddof=1), valid_020.std(ddof=1)])

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(
    "Somatotopic map quality vs MON topography strength\n"
    "(extract-mode; all seeds; ll_mon_topo = mon_ts_topo)",
    fontsize=12,
)

SEED_COLOR = "#2196F3"   # blue for individual seeds
MEAN_COLOR = "#0D47A1"   # dark blue for mean ± SD
REF_COLOR = "#9E9E9E"    # grey for high-topo reference

rng = np.random.default_rng(42)

# ---- helper to add individual seed scatter ----
def _scatter(ax, x_center, values, **kw):
    jitter = rng.uniform(-0.005, 0.005, len(values))
    ax.scatter(x_center + jitter, values, **kw)


# ============================================================
# LEFT panel — sigma_theta (lower = sharper map)
# ============================================================
ax1.axhline(ref_sigma, color=REF_COLOR, linestyle="--", linewidth=1.4,
            label=f"High-topo ref (topo=0.80, σ={ref_sigma})")
ax1.axhline(np.pi / 2, color="lightgray", linestyle=":", linewidth=0.8,
            label="π/2 (chance level)")

_scatter(ax1, 0.15, sigma_015, color=SEED_COLOR, alpha=0.4, s=22, zorder=2)
_scatter(ax1, 0.20, sigma_020, color=SEED_COLOR, alpha=0.4, s=22, zorder=2,
         label="Individual seeds")

ax1.errorbar(
    x_pos, sigma_means, yerr=sigma_sds,
    fmt="o", color=MEAN_COLOR, markersize=9,
    capsize=6, linewidth=2, zorder=4, label="Mean ± SD",
)

ax1.set_xlabel("topo  (ll_mon_topo = mon_ts_topo)", fontsize=10)
ax1.set_ylabel("σ_θ  (rad)", fontsize=10)
ax1.set_title("Map sharpness — σ_θ (lower = sharper)", fontsize=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(["0.10\n(3 seeds)", "0.15\n(10 seeds)", "0.20\n(10 seeds)"])
ax1.set_xlim(0.06, 0.25)
ax1.set_ylim(0, 1.1)
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(axis="y", alpha=0.3)

# ============================================================
# RIGHT panel — valid_fraction (higher = more reliable map)
# ============================================================
ax2.axhline(ref_valid, color=REF_COLOR, linestyle="--", linewidth=1.4,
            label=f"High-topo ref (topo=0.80, valid={ref_valid})")
ax2.axhline(0.60, color="lightgray", linestyle=":", linewidth=0.8,
            label="Threshold (0.60)")

_scatter(ax2, 0.15, valid_015, color=SEED_COLOR, alpha=0.4, s=22, zorder=2)
_scatter(ax2, 0.20, valid_020, color=SEED_COLOR, alpha=0.4, s=22, zorder=2,
         label="Individual seeds")

ax2.errorbar(
    x_pos, valid_means, yerr=valid_sds,
    fmt="o", color=MEAN_COLOR, markersize=9,
    capsize=6, linewidth=2, zorder=4, label="Mean ± SD",
)

ax2.set_xlabel("topo  (ll_mon_topo = mon_ts_topo)", fontsize=10)
ax2.set_ylabel("valid_fraction", fontsize=10)
ax2.set_title("Map reliability — valid_fraction (higher = better)", fontsize=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(["0.10\n(3 seeds)", "0.15\n(10 seeds)", "0.20\n(10 seeds)"])
ax2.set_xlim(0.06, 0.25)
ax2.set_ylim(0.40, 1.02)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(axis="y", alpha=0.3)

# ============================================================
# Save
# ============================================================
fig.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "Picture"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "topo_gradient_summary.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)
