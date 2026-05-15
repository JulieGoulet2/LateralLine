#!/usr/bin/env python3
"""
plots/chapter5_figures.py
-------------------------
Reproduce figures from chapter 5 of Iris Hydi's master thesis
(IrisHidiMaster.pdf) using the lateral line model.

Iris Hydi's original model: snake pit-organ system (infrared sensing).
This script reproduces the same figure types for the fish lateral line,
using the same 2D feedforward architecture and the same metrics.

Produces:
  Picture/ch5_fig51a_sigma_vs_dist_topo.png   — σ_θ vs D, different topo levels
  Picture/ch5_fig54_sharpening_vs_dist.png    — σ_LL/σ_TS sharpening vs D, different σ_noise
  Picture/ch5_fig55_variability_vs_dist.png   — Δ_TS/Δ_LL variability ratio vs D, different σ_noise

Reads JSON result files from Runs/distswp_<label>_seed<NNN>_d<DDD>/artifacts/seed_<NNN>_results.json.

Usage (from project root, using base anaconda python):
    python plots/chapter5_figures.py
"""

from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "Runs"
PIC_DIR = ROOT / "Picture"
PIC_DIR.mkdir(exist_ok=True)

# Body length used to normalise D/L.
BODY_LEN_CM = 4.0


def _collect_sweep(label: str) -> dict:
    """
    Collect all results for a given sweep label.

    Returns dict keyed by distance D:
        { D_cm: { seed: {result dict} } }
    """
    pattern = f"distswp_{label}_seed*_d*"
    out: dict[float, dict[int, dict]] = defaultdict(dict)
    for d in (RUNS_DIR.glob(pattern)):
        m = re.search(r"seed(\d+)_d(\d+)", d.name)
        if not m:
            continue
        seed = int(m.group(1))
        D_cm = int(m.group(2)) / 100.0
        json_path = d / "artifacts" / f"seed_{seed}_results.json"
        if not json_path.exists():
            continue
        try:
            out[D_cm][seed] = json.loads(json_path.read_text())
        except Exception as e:
            print(f"  ! skip {json_path}: {e}")
    return dict(out)


def _per_distance_arrays(sweep: dict, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (D_array, mean_array, sd_array) sorted by D."""
    Ds = sorted(sweep.keys())
    means, sds = [], []
    for D in Ds:
        vals = np.array([sweep[D][s][field] for s in sweep[D] if not np.isnan(sweep[D][s].get(field, np.nan))])
        if len(vals) == 0:
            means.append(np.nan); sds.append(np.nan)
        else:
            means.append(float(vals.mean()))
            sds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
    return np.array(Ds), np.array(means), np.array(sds)


# =============================================================
# Figure 5.1a — σ_θ vs D, curves for different topo levels
# =============================================================
def fig51a():
    topo_sweeps = {
        0.10: ("topo010", "#D32F2F"),  # red
        0.20: ("topo020", "#1976D2"),  # blue
        0.40: ("topo040", "#388E3C"),  # green
        0.80: ("topo080", "#7B1FA2"),  # purple
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for topo, (label, color) in topo_sweeps.items():
        sweep = _collect_sweep(label)
        if not sweep:
            print(f"  ! Fig 5.1a: no data for {label}")
            continue
        D, mean, sd = _per_distance_arrays(sweep, "sigma_theta_rad")
        D_over_L = D / BODY_LEN_CM
        ax.plot(D_over_L, mean, "-o", color=color, markersize=5,
                label=f"topo = {topo:.2f}", linewidth=1.8)
        ax.fill_between(D_over_L, mean - sd, mean + sd, color=color, alpha=0.18)

    ax.axhline(np.pi / 2, color="gray", linestyle=":", linewidth=0.8,
               label="π/2 (chance)")
    ax.axvline(0.8 / BODY_LEN_CM, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="training D = 0.8 cm")

    ax.set_xlabel("Stimulus distance D / L  (L = 4 cm)", fontsize=11)
    ax.set_ylabel("Somatotopic error  σ_θ  (rad)", fontsize=11)
    ax.set_title("Fig 5.1 — Map quality vs stimulus distance\n"
                 "Curves at different MON anatomical somatotopy (topo)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, np.pi / 2 + 0.3)
    ax.set_xlim(0, None)

    fig.tight_layout()
    out_path = PIC_DIR / "ch5_fig51a_sigma_vs_dist_topo.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# =============================================================
# Figure 5.4 — Sharpening σ_LL/σ_TS vs D, different σ_noise
# (Population-level sharpening; in our model individual-cell σ_w
#  is NOT sharper in TS due to vertical bands, but the population
#  somatotopic error IS sharper. We plot σ_θ_LL / σ_θ_TS.)
# =============================================================
def fig54():
    noise_sweeps = {
        0.0: ("topo020",   "#1976D2", "σ_noise = 0 Hz"),
        2.0: ("topo020n2", "#FBC02D", "σ_noise = 2 Hz"),
        5.0: ("topo020n5", "#D32F2F", "σ_noise = 5 Hz"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for noise_hz, (label, color, legend) in noise_sweeps.items():
        sweep = _collect_sweep(label)
        if not sweep:
            print(f"  ! Fig 5.4: no data for {label}")
            continue
        Ds = sorted(sweep.keys())
        ratios = []
        sds = []
        for D in Ds:
            vals = []
            for s in sweep[D]:
                r = sweep[D][s]
                if r["sigma_theta_rad"] > 0:
                    vals.append(r["sigma_theta_ll_rad"] / r["sigma_theta_rad"])
            if len(vals) == 0:
                ratios.append(np.nan); sds.append(np.nan)
            else:
                arr = np.array(vals)
                ratios.append(float(arr.mean()))
                sds.append(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)
        Ds = np.array(Ds)
        ratios = np.array(ratios)
        sds = np.array(sds)
        D_over_L = Ds / BODY_LEN_CM
        ax.plot(D_over_L, ratios, "-o", color=color, markersize=5,
                label=legend, linewidth=1.8)
        ax.fill_between(D_over_L, ratios - sds, ratios + sds, color=color, alpha=0.18)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8,
               label="ratio = 1 (no sharpening)")
    ax.axvline(0.8 / BODY_LEN_CM, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="training D = 0.8 cm")

    ax.set_xlabel("Stimulus distance D / L", fontsize=11)
    ax.set_ylabel("Population sharpening ratio  σ_θ^LL / σ_θ^TS", fontsize=11)
    ax.set_title("Fig 5.4 — Population-level sharpening vs distance\n"
                 "Curves at different test-phase noise σ_noise (topo = 0.20)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, None)

    fig.tight_layout()
    out_path = PIC_DIR / "ch5_fig54_sharpening_vs_dist.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# =============================================================
# Figure 5.5 — Trial variability ratio Δ_TS/Δ_LL vs D, different σ_noise
# =============================================================
def fig55():
    noise_sweeps = {
        0.0: ("topo020",   "#1976D2", "σ_noise = 0 Hz"),
        2.0: ("topo020n2", "#FBC02D", "σ_noise = 2 Hz"),
        5.0: ("topo020n5", "#D32F2F", "σ_noise = 5 Hz"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for noise_hz, (label, color, legend) in noise_sweeps.items():
        sweep = _collect_sweep(label)
        if not sweep:
            print(f"  ! Fig 5.5: no data for {label}")
            continue
        Ds = sorted(sweep.keys())
        ratios = []
        sds = []
        for D in Ds:
            vals = []
            for s in sweep[D]:
                r = sweep[D][s]
                if r["delta_trial_ll_rad"] > 1e-9:
                    vals.append(r["delta_trial_rad"] / r["delta_trial_ll_rad"])
            if len(vals) == 0:
                ratios.append(np.nan); sds.append(np.nan)
            else:
                arr = np.array(vals)
                ratios.append(float(arr.mean()))
                sds.append(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)
        Ds = np.array(Ds)
        ratios = np.array(ratios)
        sds = np.array(sds)
        D_over_L = Ds / BODY_LEN_CM
        ax.plot(D_over_L, ratios, "-o", color=color, markersize=5,
                label=legend, linewidth=1.8)
        ax.fill_between(D_over_L, ratios - sds, ratios + sds, color=color, alpha=0.18)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8,
               label="ratio = 1 (no improvement)")
    ax.axvline(0.8 / BODY_LEN_CM, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="training D = 0.8 cm")

    ax.set_xlabel("Stimulus distance D / L", fontsize=11)
    ax.set_ylabel("Trial-to-trial variability ratio  Δ_trial^TS / Δ_trial^LL", fontsize=11)
    ax.set_title("Fig 5.5 — Trial variability ratio vs distance\n"
                 "Curves at different test-phase noise σ_noise (topo = 0.20)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, None)

    fig.tight_layout()
    out_path = PIC_DIR / "ch5_fig55_variability_vs_dist.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# =============================================================
# Figure 5.1b — σ_θ vs D, curves for different MON neuron counts
# (Phase 2: SCALED gain only. Unscaled is shown separately in fig51b_comparison.)
# =============================================================
def fig51b():
    nmon_sweeps = {
        400:  ("nmon400_scaled",  "#D32F2F", "N_MON = 400"),
        800:  ("nmon800_scaled",  "#FBC02D", "N_MON = 800"),
        1600: ("nmon1600_scaled", "#388E3C", "N_MON = 1600"),
        3200: ("topo020",         "#1976D2", "N_MON = 3200"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for nmon, (label, color, legend) in nmon_sweeps.items():
        sweep = _collect_sweep(label)
        if not sweep:
            print(f"  ! Fig 5.1b: no data for {label}")
            continue
        D, mean, sd = _per_distance_arrays(sweep, "sigma_theta_rad")
        D_over_L = D / BODY_LEN_CM
        ax.plot(D_over_L, mean, "-o", color=color, markersize=5,
                label=legend, linewidth=1.8)
        ax.fill_between(D_over_L, mean - sd, mean + sd, color=color, alpha=0.18)

    ax.axhline(np.pi / 2, color="gray", linestyle=":", linewidth=0.8,
               label="π/2 (chance)")
    ax.axvline(0.8 / BODY_LEN_CM, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="training D = 0.8 cm")

    ax.set_xlabel("Stimulus distance D / L  (L = 4 cm)", fontsize=11)
    ax.set_ylabel("Somatotopic error  σ_θ  (rad)", fontsize=11)
    ax.set_title("Fig 5.1b — Map quality vs stimulus distance\n"
                 "Curves at different MON neuron count (SCALED gain, topo = 0.20)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, np.pi / 2 + 0.3)
    ax.set_xlim(0, None)

    fig.tight_layout()
    out_path = PIC_DIR / "ch5_fig51b_sigma_vs_dist_nmon.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# =============================================================
# Figure 5.1b' — UNSCALED vs SCALED comparison (motivation for gain scaling)
# =============================================================
def fig51b_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    nmon_pairs = [
        (400,  "nmon400",  "nmon400_scaled",  "#D32F2F"),
        (800,  "nmon800",  "nmon800_scaled",  "#FBC02D"),
        (1600, "nmon1600", "nmon1600_scaled", "#388E3C"),
    ]
    baseline = ("topo020", "#1976D2", "N_MON = 3200 (baseline gain=220)")

    for ax, (which, title) in zip(axes, [(1, "Unscaled gain = 220 mV"), (2, "Scaled gain (220 × 3200/N)")]):
        sweep = _collect_sweep(baseline[0])
        if sweep:
            D, mean, sd = _per_distance_arrays(sweep, "sigma_theta_rad")
            D_over_L = D / BODY_LEN_CM
            ax.plot(D_over_L, mean, "-o", color=baseline[1], markersize=5,
                    label=baseline[2], linewidth=1.8)
            ax.fill_between(D_over_L, mean - sd, mean + sd, color=baseline[1], alpha=0.18)

        for nmon, lbl_uns, lbl_sca, color in nmon_pairs:
            label = lbl_uns if which == 1 else lbl_sca
            sweep = _collect_sweep(label)
            if not sweep:
                continue
            D, mean, sd = _per_distance_arrays(sweep, "sigma_theta_rad")
            D_over_L = D / BODY_LEN_CM
            ax.plot(D_over_L, mean, "-o", color=color, markersize=5,
                    label=f"N_MON = {nmon}", linewidth=1.8)
            ax.fill_between(D_over_L, mean - sd, mean + sd, color=color, alpha=0.18)

        ax.axhline(np.pi / 2, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(0.8 / BODY_LEN_CM, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Stimulus distance D / L", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, None)

    axes[0].set_ylabel("Somatotopic error  σ_θ  (rad)", fontsize=11)
    axes[0].set_ylim(0, np.pi / 2 + 0.3)
    fig.suptitle("Fig 5.1b' — Effect of gain scaling on MON-size sensitivity\n"
                 "Left: fixed gain=220 (small N: TS silent). Right: gain=220×(3200/N) (recovers firing).",
                 fontsize=11)

    fig.tight_layout()
    out_path = PIC_DIR / "ch5_fig51b_comparison_unscaled_vs_scaled.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# =============================================================
# Figure 5.3 — σ_θ at training distance vs MON neuron count
# (For Iris Hydi this was at fixed observation period T; we show
#  the result at D = 0.8 cm averaged across seeds.)
# =============================================================
def fig53():
    # SCALED gain points (and the baseline at N=3200).
    points = []  # (nmon, label, mean, sd, n)
    for nmon, label in [(400, "nmon400_scaled"), (800, "nmon800_scaled"),
                        (1600, "nmon1600_scaled"), (3200, "topo020")]:
        sweep = _collect_sweep(label)
        if not sweep or 0.8 not in sweep:
            print(f"  ! Fig 5.3: no D=0.8 data for {label}")
            continue
        vals = np.array([sweep[0.8][s]["sigma_theta_rad"] for s in sweep[0.8]])
        points.append((nmon, vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0, len(vals)))

    if not points:
        print("  ! Fig 5.3: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    nmons = np.array([p[0] for p in points])
    means = np.array([p[1] for p in points])
    sds   = np.array([p[2] for p in points])
    ns    = [p[3] for p in points]

    ax.errorbar(nmons, means, yerr=sds, fmt="o-", color="#0D47A1",
                markersize=8, capsize=6, linewidth=1.8,
                label="SCALED gain (= 220 × 3200/N)")
    for x, y, n in zip(nmons, means, ns):
        ax.annotate(f"N={n} seeds", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#0D47A1")

    # Optionally overlay UNSCALED at the training distance.
    unscaled_points = []
    for nmon, label in [(400, "nmon400"), (800, "nmon800"), (1600, "nmon1600")]:
        sweep = _collect_sweep(label)
        if not sweep or 0.8 not in sweep:
            continue
        vals = np.array([sweep[0.8][s]["sigma_theta_rad"] for s in sweep[0.8]])
        unscaled_points.append((nmon, vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0))
    if unscaled_points:
        uns_x = np.array([p[0] for p in unscaled_points])
        uns_m = np.array([p[1] for p in unscaled_points])
        uns_s = np.array([p[2] for p in unscaled_points])
        ax.errorbar(uns_x, uns_m, yerr=uns_s, fmt="x--", color="#D32F2F",
                    markersize=8, capsize=6, linewidth=1.4, alpha=0.7,
                    label="UNSCALED gain = 220 (small N → TS silent)")

    ax.axhline(np.pi / 2, color="gray", linestyle=":", linewidth=0.8,
               label="π/2 (chance)")

    ax.set_xscale("log", base=2)
    ax.set_xticks([400, 800, 1600, 3200])
    ax.set_xticklabels(["400", "800", "1600", "3200"])
    ax.set_xlabel("MON neuron count  N_MON", fontsize=11)
    ax.set_ylabel("Somatotopic error  σ_θ  at training D = 0.8 cm  (rad)", fontsize=11)
    ax.set_title("Fig 5.3 — Effect of MON neuron count on map quality\n"
                 "Tested at training distance D = 0.8 cm, topo = 0.20",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, np.pi / 2 + 0.3)

    fig.tight_layout()
    out_path = PIC_DIR / "ch5_fig53_sigma_vs_nmon.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# =============================================================
# Main
# =============================================================
if __name__ == "__main__":
    print("Generating chapter 5 figures...\n")
    fig51a()
    fig51b()
    fig51b_comparison()
    fig53()
    fig54()
    fig55()
    print("\nNote: Figure 5.2 (different observation periods T) requires per-window")
    print("analysis; skipped in this initial pass.")
