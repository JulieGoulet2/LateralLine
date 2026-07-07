"""
Test Julie's hypothesis (2026-07-07): would training on a RICHER, more realistic
stimulus ensemble (varying direction, distance, speed, object size) wash out the
band-forming structure — i.e. are the bands an artifact of our simplified
single-condition training?

ANALYSIS ONLY (no Brian2). Input-statistics proxy: we inspect the position-position
similarity structure the STDP rule would integrate over the training ensemble.

Band-forming metric: cosine similarity of the RECTIFIED LL pattern (only the
actively-driving neuromasts, above a global threshold). Rectification is what
neurons actually do, and it breaks the sign symmetry — so DIRECTION (which flips the
bipolar leading/trailing lobe) can genuinely matter here, unlike raw correlation.

Key expectations to verify numerically (not assume):
  - speed U and size R scale the field amplitude -> near-zero effect (shape unchanged);
  - distance D changes spatial width -> modest effect (already ~9% for raw corr);
  - direction flips the bipolar lobe -> the physically interesting test.

Usage:
    /Users/juliegoulet/anaconda3/bin/python plots/tuning_multimodality_ensemble.py
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
R0_HZ, RMAX_HZ, A_PER_CM = 40.0, 200.0, 300.0
X_MIN_CM, X_MAX_CM, N_X = -0.5, 4.5, 200

BASE = dict(D=0.8, direction=1.0, U=5.0, R=0.5)
RECT_KEEP_FRAC = 0.30       # keep the top 30% most-driven neuromasts as "active"
GHOST_MIN_CM, GHOST_MAX_CM = 0.8, 2.5


def ll_patterns(x_grid, D, direction, U, R):
    """Rectified LL activation pattern for each source position. Shape (N_X, N_LL)."""
    xi = np.linspace(0.0, LINE_LEN_CM, N_LL)
    yi = np.zeros_like(xi)
    eX = 1.0 if direction >= 0 else -1.0
    P = np.zeros((x_grid.size, N_LL))
    for k, xs in enumerate(x_grid):
        v = hydrodynamic_velocity_parallel(xi, yi, xs, D, U, R, eX=eX, eY=0.0, sx=1.0, sy=0.0)
        P[k] = np.clip(R0_HZ + A_PER_CM * v, 0.0, RMAX_HZ)
    return P


def rectified_similarity(P):
    """Cosine similarity of rectified patterns (band-forming overlap of active neuromasts)."""
    thr = np.quantile(P, 1.0 - RECT_KEEP_FRAC)
    Pr = np.clip(P - thr, 0.0, None)
    n = np.linalg.norm(Pr, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Pn = Pr / n
    return Pn @ Pn.T


def ensemble_similarity(x_grid, conditions):
    """Average band-forming similarity over a list of (D, direction, U, R) conditions."""
    mats = [rectified_similarity(ll_patterns(x_grid, **c)) for c in conditions]
    return np.mean(mats, axis=0)


def ghost_rms(S, dx):
    win = (dx >= GHOST_MIN_CM) & (dx <= GHOST_MAX_CM)
    return float(np.sqrt(np.mean(S[win] ** 2)))


def main():
    out_dir = Path("Picture"); out_dir.mkdir(exist_ok=True)
    x = np.linspace(X_MIN_CM, X_MAX_CM, N_X)
    dx = np.abs(x[:, None] - x[None, :])

    # Ensembles (each is a list of conditions to average over).
    base = [dict(BASE)]
    dir_ens = [dict(BASE, direction=+1.0), dict(BASE, direction=-1.0)]
    dist_ens = [dict(BASE, D=d) for d in np.linspace(0.5, 2.0, 8)]
    speed_ens = [dict(BASE, U=u) for u in np.linspace(3.0, 8.0, 6)]
    size_ens = [dict(BASE, R=r) for r in np.linspace(0.3, 0.7, 5)]
    rich_ens = [dict(D=d, direction=s, U=BASE["U"], R=r)
                for d in np.linspace(0.5, 2.0, 5) for s in (+1.0, -1.0)
                for r in (0.3, 0.5, 0.7)]

    ensembles = {
        "baseline (single)": base,
        "+ direction (fwd+back)": dir_ens,
        "+ distance (0.5-2.0)": dist_ens,
        "+ speed (3-8)": speed_ens,
        "+ size (0.3-0.7)": size_ens,
        "ALL rich (dir+dist+size)": rich_ens,
    }
    S = {k: ensemble_similarity(x, v) for k, v in ensembles.items()}
    g0 = ghost_rms(S["baseline (single)"], dx)

    print("=== band-forming (rectified) ghost structure, %.1f-%.1f cm window ==="
          % (GHOST_MIN_CM, GHOST_MAX_CM))
    print(f"{'ensemble':>28} | {'ghost RMS':>9} | {'reduction vs baseline':>22}")
    results = {}
    for k in ensembles:
        g = ghost_rms(S[k], dx)
        red = 100.0 * (1.0 - g / g0)
        results[k] = (g, red)
        print(f"{k:>28} | {g:9.3f} | {red:21.0f}%")

    # ---- figure: similarity maps + reduction bar chart ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    show = ["baseline (single)", "+ direction (fwd+back)", "+ distance (0.5-2.0)",
            "ALL rich (dir+dist+size)"]
    ext = (x[0], x[-1], x[0], x[-1])
    for ax, k in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], show):
        im = ax.imshow(S[k], origin="lower", cmap="magma", vmin=0, vmax=1, extent=ext)
        ax.set_title(k, fontsize=10)
        ax.set_xlabel("x' (cm)"); ax.set_ylabel("x (cm)")
    plt.colorbar(im, ax=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
                 label="rectified pattern overlap", fraction=0.025, pad=0.02)

    # bar chart of ghost reduction (top-right)
    ax = axes[0, 2]
    labels = ["direction", "distance", "speed", "size", "ALL rich"]
    keys = ["+ direction (fwd+back)", "+ distance (0.5-2.0)", "+ speed (3-8)",
            "+ size (0.3-0.7)", "ALL rich (dir+dist+size)"]
    reds = [results[k][1] for k in keys]
    colors = ["tab:red" if r > 25 else "tab:gray" for r in reds]
    ax.barh(labels, reds, color=colors)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("ghost reduction vs baseline (%)")
    ax.set_title("Which variation washes out the bands?", fontsize=10)
    for i, r in enumerate(reds):
        ax.text(r + (1 if r >= 0 else -1), i, f"{r:.0f}%", va="center",
                ha="left" if r >= 0 else "right", fontsize=9)
    ax.grid(alpha=0.3, axis="x")

    # correlation-vs-separation for baseline vs direction vs rich (bottom-right)
    ax = axes[1, 2]
    edges = np.arange(0, 3.01, 0.1); ctr = (edges[:-1] + edges[1:]) / 2
    for k, style in [("baseline (single)", "-"), ("+ direction (fwd+back)", "-"),
                     ("ALL rich (dir+dist+size)", "-")]:
        prof = [S[k][(dx >= a) & (dx < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(ctr, prof, style, lw=2, label=k)
    ax.axvspan(GHOST_MIN_CM, GHOST_MAX_CM, color="0.9", zorder=0)
    ax.set_xlabel("|x - x'| (cm)"); ax.set_ylabel("mean overlap")
    ax.set_title("overlap vs separation", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    best = max(labels, key=lambda L: results[keys[labels.index(L)]][1])
    fig.suptitle(
        f"Do richer training stimuli wash out the bands?  "
        f"Best single factor: {best} ({max(reds):.0f}% ghost reduction)",
        fontsize=13, fontweight="bold",
    )
    out = out_dir / "tuning_multimodality_ensemble.png"
    fig.savefig(out, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
