"""
Test Julie's hypothesis (2026-07-07): would training across a WIDE range of source
distances wash out the band-forming structure?

Physical idea: the LL activation pattern's spatial scale grows with source distance D
(far source -> lower, broader bump; the bipolar lobes spread further apart). If the
band-forming off-diagonal structure sits at a DIFFERENT separation for each D, then
averaging over a wide D range (what multi-distance training effectively imprints)
should smear it out, while the near-diagonal map core survives.

ANALYSIS ONLY (no Brian2). This is an input-statistics proxy: it inspects the LL
correlation structure C_D[x,x'] the STDP rule would integrate, NOT a retraining.
The earlier multi-distance PILOT used only D in [0.6, 1.2] cm (factor ~2) and left
the bands intact — this asks whether a WIDER range would behave differently.

Usage:
    /Users/juliegoulet/anaconda3/bin/python plots/tuning_multimodality_distance.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stimulus import hydrodynamic_velocity_parallel

N_LL = 100
LINE_LEN_CM = 4.0
SPEED_CM_S = 5.0
SPHERE_R_CM = 0.5
R0_HZ = 40.0
RMAX_HZ = 200.0
A_PER_CM = 300.0

X_MIN_CM, X_MAX_CM, N_X = -0.5, 4.5, 200

D_SHOW = [0.5, 0.8, 1.2, 2.0]          # distances shown individually
D_AVG_GRID = np.linspace(0.5, 2.0, 16)  # wide range STDP would see if trained on it

# Off-diagonal window used to quantify band-forming structure (beyond the local map core).
GHOST_MIN_CM = 0.8
GHOST_MAX_CM = 2.5


def ll_corr_at_distance(x_grid, D):
    """C_D[x,x'] = corr(r_LL(x), r_LL(x')) over the 100 neuromasts, at fixed distance D."""
    xi = np.linspace(0.0, LINE_LEN_CM, N_LL)
    yi = np.zeros_like(xi)
    R = np.zeros((x_grid.size, N_LL))
    for k, xs in enumerate(x_grid):
        v = hydrodynamic_velocity_parallel(
            xi, yi, xs, D, SPEED_CM_S, SPHERE_R_CM, eX=1.0, eY=0.0, sx=1.0, sy=0.0
        )
        R[k] = np.clip(R0_HZ + A_PER_CM * v, 0.0, RMAX_HZ)
    Rc = R - R.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Rc, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Rn = Rc / n
    return Rn @ Rn.T


def ghost_strength(C, dx):
    """Strength of band-forming off-diagonal structure in the ghost window.
    Measured two ways: (1) max POSITIVE correlation (creates secondary peaks),
    (2) RMS deviation of correlation (any structured co-modulation)."""
    win = (dx >= GHOST_MIN_CM) & (dx <= GHOST_MAX_CM)
    vals = C[win]
    return float(vals.max()), float(np.sqrt(np.mean(vals ** 2)))


def main():
    out_dir = Path("Picture")
    out_dir.mkdir(exist_ok=True)
    x = np.linspace(X_MIN_CM, X_MAX_CM, N_X)
    dx = np.abs(x[:, None] - x[None, :])

    # Per-distance correlation matrices.
    C_by_D = {D: ll_corr_at_distance(x, D) for D in D_SHOW}

    # Distance-averaged correlation over the wide grid (what wide-range training sees).
    C_avg = np.mean([ll_corr_at_distance(x, D) for D in D_AVG_GRID], axis=0)

    print("=== band-forming (ghost) structure in the %.1f-%.1f cm off-diagonal window ==="
          % (GHOST_MIN_CM, GHOST_MAX_CM))
    print(f"{'condition':>22} | {'max +corr':>9} | {'rms corr':>8}")
    single_rms = []
    for D in D_SHOW:
        mx, rms = ghost_strength(C_by_D[D], dx)
        single_rms.append(rms)
        print(f"{'single D=%.1f cm' % D:>22} | {mx:+9.3f} | {rms:8.3f}")
    mx_a, rms_a = ghost_strength(C_avg, dx)
    print(f"{'mean single-D':>22} | {'':>9} | {np.mean(single_rms):8.3f}")
    print(f"{'wide-range averaged':>22} | {mx_a:+9.3f} | {rms_a:8.3f}")
    reduction = 100.0 * (1.0 - rms_a / np.mean(single_rms))
    print(f"\nghost RMS reduction from distance-averaging: {reduction:.0f}%")

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    ext = (x[0], x[-1], x[0], x[-1])
    for ax, D in zip(axes.flat[:4], D_SHOW):
        im = ax.imshow(C_by_D[D], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=ext)
        ax.set_title(f"single distance D = {D:.1f} cm")
        ax.set_xlabel("x' (cm)"); ax.set_ylabel("x (cm)")
    # averaged
    ax = axes.flat[4]
    im = ax.imshow(C_avg, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=ext)
    ax.set_title(f"wide-range averaged\nD in [{D_AVG_GRID[0]:.1f}, {D_AVG_GRID[-1]:.1f}] cm")
    ax.set_xlabel("x' (cm)"); ax.set_ylabel("x (cm)")
    plt.colorbar(im, ax=list(axes.flat[:5]), label="corr of r_LL(x), r_LL(x')",
                 fraction=0.025, pad=0.02)

    # correlation vs |dx| for each D + averaged, to show the ghost lobe shifting with D
    ax = axes.flat[5]
    edges = np.arange(0, 3.01, 0.1)
    ctr = (edges[:-1] + edges[1:]) / 2
    for D in D_SHOW:
        prof = [C_by_D[D][(dx >= a) & (dx < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(ctr, prof, lw=1.4, label=f"D={D:.1f}")
    prof_a = [C_avg[(dx >= a) & (dx < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
    ax.plot(ctr, prof_a, "k-", lw=2.6, label="wide avg")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvspan(GHOST_MIN_CM, GHOST_MAX_CM, color="0.85", zorder=0, label="ghost window")
    ax.set_xlabel("|x - x'| (cm)"); ax.set_ylabel("mean correlation")
    ax.set_title("correlation vs separation\n(ghost lobe shifts with D -> averaging smears it)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        f"Does wide-range distance training wash out the bands?  "
        f"(ghost RMS reduced {reduction:.0f}% by averaging)",
        fontsize=13, fontweight="bold",
    )
    out = out_dir / "tuning_multimodality_distance.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
