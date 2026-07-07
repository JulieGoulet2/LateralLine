"""
Open questions Step 1 — quantify the "vertical bands" (multimodal per-TS-cell tuning).

ANALYSIS ONLY. No Brian2 simulation is run. We load the trained weights
(`ll_mon_w_mV`, `mon_ts_w` + connectivity) from the saved .npz snapshots and the
deterministic dipole forward model from stimulus.py, and reconstruct each TS cell's
tuning curve purely feedforward:

    r_LL(x)      : noiseless neuromast rate vector for a source at position x
    a_MON(x)[j]  : ReLU( Σ_i w_LL->MON[i,j] · r_LL(x)[i] − baseline_j )   (contrast/modulation)
    d_TS(x)[k]   : Σ_j w_MON->TS[j,k] · a_MON(x)[j]

(a) Multimodality: count peaks per TS tuning curve d_TS(·)[k] across x (find_peaks with
    a prominence threshold), pooled over all seeds.

(b) LL-geometry hypothesis: the source x -> neuromast-pattern map has an intrinsic
    correlation structure. Build C[a,b] = corr(r_LL(x_a), r_LL(x_b)) over the 100
    neuromasts. Off-diagonal ridges (distant x that are correlated) = intrinsic geometry,
    not a learning bug.

Usage:
    /Users/juliegoulet/anaconda3/bin/python plots/tuning_multimodality.py

Writes Picture/tuning_multimodality.png (also prints summary statistics).
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stimulus import hydrodynamic_velocity_parallel

# ---------------------------------------------------------------------------
# Fixed geometry / stimulus for the topo=0.20 baseline (ll_thesis mode).
# All values are StimulusParams defaults, NOT overridden by the baseline recipe.
# ---------------------------------------------------------------------------
N_LL = 100
LINE_LEN_CM = 4.0          # neuromasts span 0..4 cm
DISTANCE_CM = 0.8          # source distance (Y)
SPEED_CM_S = 5.0
SPHERE_R_CM = 0.5
R0_HZ = 40.0
RMAX_HZ = 200.0
A_PER_CM = 300.0

# Source sweep: test starts at x0 = -0.5 (direction +) and covers test_path_cm = 5 cm.
X_MIN_CM = -0.5
X_MAX_CM = 4.5
N_X = 200                  # source-position grid resolution

RUN_DIR = Path("Runs/llmon_topo020_seeds127_132/artifacts")
SEEDS = [127, 128, 129, 130, 131, 132]
EXAMPLE_SEED = 127         # option 1: single seed for the visual band-map panel

# Two-stage competitive thresholding (analysis-only stand-in for spiking + global
# inhibition). A GLOBAL CONSTANT threshold per layer reproduces the x-varying band
# density: dense stripes where drive is high (array ends), sparse gap where drive is
# low (x ~ 2 cm). The threshold is set to a target ACTIVE FRACTION of the layer.
MON_ACTIVE_FRAC = 0.15     # fraction of MON (cell x x-bin) entries kept active
TS_ACTIVE_FRAC = 0.15      # fraction of TS (cell x x-bin) entries kept active

# Peak-detection settings for the TS tuning curves.
SMOOTH_SIGMA_BINS = 1.0
PROMINENCE_FRAC = 0.20     # peak must rise >= 20% of the cell's dynamic range
MIN_SEP_FRAC = 0.04        # peaks must be >= 4% of the sweep apart
ACTIVE_FRAC_OF_MAX = 0.02  # cells whose peak activity < 2% of the global max are "silent"

# Correlation-ridge settings for part (b).
CORR_THRESH = 0.8          # |corr| above this = "similar LL pattern"
DISTANT_CM = 1.0           # x-positions this far apart count as "distant"


def ll_representation(x_grid):
    """r_LL(x): noiseless neuromast rate vector for each source x. Shape (N_X, N_LL)."""
    xi = np.linspace(0.0, LINE_LEN_CM, N_LL)
    yi = np.zeros_like(xi)
    R = np.zeros((x_grid.size, N_LL))
    for k, xs in enumerate(x_grid):
        v = hydrodynamic_velocity_parallel(
            xi, yi, xs, DISTANCE_CM, SPEED_CM_S, SPHERE_R_CM,
            eX=1.0, eY=0.0, sx=1.0, sy=0.0,
        )
        R[k] = np.clip(R0_HZ + A_PER_CM * v, 0.0, RMAX_HZ)
    return xi, R


def _global_threshold_relu(drive, active_frac):
    """ReLU above a single global threshold chosen so ~active_frac of entries survive.
    A constant threshold (not per-x) preserves the x-varying activity density that
    global feedback inhibition produces in the spiking network."""
    thr = np.quantile(drive, 1.0 - active_frac)
    return np.clip(drive - thr, 0.0, None)


def reconstruct_ts_tuning(npz_path, r_ll):
    """Two-stage competitive reconstruction of TS tuning. r_ll: (N_X, N_LL).
    Returns act_TS: (N_X, n_ts), rectified TS activity after MON+TS thresholding."""
    d = np.load(npz_path, allow_pickle=True)
    ll_i = d["ll_mon_i"].astype(int)     # LL index
    ll_j = d["ll_mon_j"].astype(int)     # MON index
    ll_w = d["ll_mon_w_mV"].astype(float)
    mt_i = d["mon_ts_i"].astype(int)     # MON index
    mt_j = d["mon_ts_j"].astype(int)     # TS index
    mt_w = d["mon_ts_w"].astype(float)
    n_mon = int(max(ll_j.max(), mt_i.max()) + 1)
    n_ts = int(mt_j.max() + 1)
    n_x = r_ll.shape[0]

    # Stage 1 — MON linear drive vs x: drive_MON[x, j] = Σ_i w[i,j] r_LL(x)[i]
    drive_mon = np.zeros((n_x, n_mon))
    for kx in range(n_x):
        contrib = r_ll[kx, ll_i] * ll_w
        drive_mon[kx] = np.bincount(ll_j, weights=contrib, minlength=n_mon)
    # Contrast: MON responds above its own median drive across x (efference-style), then
    # a global threshold + rectification mimics MON threshold + global inhibition.
    drive_mon -= np.median(drive_mon, axis=0, keepdims=True)
    a_mon = _global_threshold_relu(drive_mon, MON_ACTIVE_FRAC)

    # Stage 2 — TS drive from thresholded MON, then TS threshold + global inhibition.
    drive_ts = np.zeros((n_x, n_ts))
    for kx in range(n_x):
        contrib = a_mon[kx, mt_i] * mt_w
        drive_ts[kx] = np.bincount(mt_j, weights=contrib, minlength=n_ts)
    act_ts = _global_threshold_relu(drive_ts, TS_ACTIVE_FRAC)
    return act_ts


def count_peaks(curve, min_sep_bins):
    """Number of prominent peaks in one TS tuning curve; returns (n_peaks, peak_idx)."""
    rng = curve.max() - curve.min()
    if rng <= 0:
        return 0, np.array([], dtype=int)
    sm = gaussian_filter1d(curve, SMOOTH_SIGMA_BINS)
    prom = PROMINENCE_FRAC * (sm.max() - sm.min())
    if prom <= 0:
        return 0, np.array([], dtype=int)
    idx, _ = find_peaks(sm, prominence=prom, distance=max(1, int(min_sep_bins)))
    return int(idx.size), idx


def main():
    out_dir = Path("Picture")
    out_dir.mkdir(exist_ok=True)
    x_grid = np.linspace(X_MIN_CM, X_MAX_CM, N_X)
    min_sep_bins = MIN_SEP_FRAC * N_X

    _, r_ll = ll_representation(x_grid)

    # ---- (a) peak counts pooled across seeds ----
    all_peak_counts = []
    example_drive = None
    for seed in SEEDS:
        npz = RUN_DIR / f"latest_seed_{seed}.npz"
        if not npz.exists():
            print(f"  WARNING missing {npz}")
            continue
        drive_ts = reconstruct_ts_tuning(npz, r_ll)          # (N_X, n_ts)
        if seed == EXAMPLE_SEED:
            example_drive = drive_ts
        global_max = drive_ts.max()
        for k in range(drive_ts.shape[1]):
            curve = drive_ts[:, k]
            if curve.max() < ACTIVE_FRAC_OF_MAX * global_max:
                continue  # silent cell — no tuning to count
            n_pk, _ = count_peaks(curve, min_sep_bins)
            all_peak_counts.append(n_pk)
    all_peak_counts = np.array(all_peak_counts)

    n_cells = all_peak_counts.size
    mean_pk = all_peak_counts.mean()
    frac_multi = np.mean(all_peak_counts >= 2)
    print("=== (a) multimodality ===")
    print(f"  active TS cells pooled over {len(SEEDS)} seeds: {n_cells}")
    print(f"  mean peaks / cell: {mean_pk:.2f}")
    print(f"  fraction with >= 2 peaks: {frac_multi:.2%}")
    for v in range(1, 6):
        print(f"    exactly {v} peak(s): {np.mean(all_peak_counts == v):.2%}")

    # ---- (b) LL correlation structure across x ----
    # C[a,b] = corr(r_LL(x_a), r_LL(x_b)) over the 100 neuromasts.
    Rc = r_ll - r_ll.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Rc, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    Rn = Rc / norm
    C = Rn @ Rn.T                                            # (N_X, N_X)

    dx = np.abs(x_grid[:, None] - x_grid[None, :])
    distant = dx >= DISTANT_CM
    n_distant_corr = np.sum((C >= CORR_THRESH) & distant, axis=1)
    frac_x_with_distant_twin = np.mean(n_distant_corr > 0)
    print("=== (b) LL geometry ===")
    print(f"  fraction of source-x with a DISTANT (>= {DISTANT_CM} cm) x' at corr >= {CORR_THRESH}: "
          f"{frac_x_with_distant_twin:.2%}")
    print(f"  median # distant correlated x' per x: {np.median(n_distant_corr):.0f}")

    # ---- figure: 2x2 ----
    assert example_drive is not None, f"example seed {EXAMPLE_SEED} not found"
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (i) reconstructed TS-vs-x tuning map, example seed. Cells are re-ordered by
    # their strongest-peak position so the coarse map + multimodal off-diagonal
    # activation (secondary bands) are both visible.
    ax = axes[0, 0]
    gm = example_drive.max()
    active = [k for k in range(example_drive.shape[1])
              if example_drive[:, k].max() >= ACTIVE_FRAC_OF_MAX * gm]
    order = sorted(active, key=lambda k: np.argmax(example_drive[:, k]))
    im = ax.imshow(
        example_drive[:, order].T, aspect="auto", origin="lower",
        extent=(x_grid[0], x_grid[-1], 0, len(order)),
        cmap="viridis",
    )
    ax.set_xlabel("source x (cm)")
    ax.set_ylabel("TS cell (sorted by main peak)")
    ax.set_title(f"(i) Reconstructed TS tuning map — seed {EXAMPLE_SEED}\n"
                 "(off-diagonal activation = secondary peaks)")
    plt.colorbar(im, ax=ax, label="TS activity (a.u.)")

    # (ii) example per-cell tuning curves — pick genuinely MULTIMODAL cells.
    ax = axes[0, 1]
    peak_info = [(k, *count_peaks(example_drive[:, k], min_sep_bins)) for k in active]
    multi = [(k, n, idx) for (k, n, idx) in peak_info if n >= 2]
    multi.sort(key=lambda t: -example_drive[:, t[0]].max())
    picks = multi[:: max(1, len(multi) // 4)][:4] if multi else [(k, *count_peaks(example_drive[:, k], min_sep_bins)) for k in active[:4]]
    for k, n_pk, idx in picks:
        curve = example_drive[:, k]
        sm = gaussian_filter1d(curve, SMOOTH_SIGMA_BINS)
        line, = ax.plot(x_grid, sm, lw=1.6, label=f"TS {k} ({n_pk} peaks)")
        ax.plot(x_grid[idx], sm[idx], "v", color=line.get_color(), ms=9)
    ax.set_xlabel("source x (cm)")
    ax.set_ylabel("TS activity (a.u.)")
    ax.set_title("(ii) Example multimodal TS cells\n(▼ = detected peaks)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (iii) histogram of peaks per cell, pooled across seeds
    ax = axes[1, 0]
    bins = np.arange(0.5, all_peak_counts.max() + 1.5, 1.0)
    ax.hist(all_peak_counts, bins=bins, color="tab:purple", edgecolor="black", rwidth=0.9)
    ax.axvline(mean_pk, color="tab:red", ls="--", lw=2, label=f"mean = {mean_pk:.2f}")
    ax.set_xlabel("peaks per TS tuning curve")
    ax.set_ylabel("number of TS cells")
    ax.set_title(f"(iii) Multimodality across {len(SEEDS)} seeds\n"
                 f"{frac_multi:.0%} of active cells have >= 2 peaks")
    ax.set_xticks(np.arange(1, all_peak_counts.max() + 1))
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (iv) LL correlation matrix across source x
    ax = axes[1, 1]
    im = ax.imshow(
        C, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
        extent=(x_grid[0], x_grid[-1], x_grid[0], x_grid[-1]),
    )
    ax.set_xlabel("source x' (cm)")
    ax.set_ylabel("source x (cm)")
    ax.set_title("(iv) LL pattern correlation across x — bipolar geometry\n"
                 "(blue flanks = anti-correlation at ~1 cm; weak distant re-correlation)")
    plt.colorbar(im, ax=ax, label="corr of r_LL(x), r_LL(x')")

    fig.suptitle(
        "Multimodal per-TS-cell tuning (\"vertical bands\") is inherited from lateral-line geometry",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / "tuning_multimodality.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    return dict(
        n_cells=n_cells, mean_pk=mean_pk, frac_multi=frac_multi,
        frac_x_with_distant_twin=frac_x_with_distant_twin,
    )


if __name__ == "__main__":
    main()
