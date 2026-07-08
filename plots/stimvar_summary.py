"""
Step 2 (Q4) summary: does the learned somatotopic map hold under varied TEST stimuli?

Reads extract-mode results produced by run_stimvar_extract.sh:
    Runs/stimvar_<label>_seed<SEED>_<cond>/artifacts/seed_<SEED>_results.json

Plots map quality (sigma_theta, lower = better) and valid_fraction per stimulus
condition, pooled over seeds, with the baseline as reference.

Usage:
    /Users/juliegoulet/anaconda3/bin/python plots/stimvar_summary.py
"""

from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = "topo020"
SEEDS = [127, 128, 129, 130, 131, 132]

# (cond_key, display label, group) — ordered for the figure.
CONDS = [
    ("size_small", "small\n(r=0.3)", "size"),
    ("base",       "BASE\n(r=0.5, 5cm/s, fwd)", "base"),
    ("size_big",   "big\n(r=0.7)", "size"),
    ("speed_slow", "slow\n(2.5cm/s)", "speed"),
    ("speed_fast", "fast\n(10cm/s)", "speed"),
    ("dir_back",   "backward", "direction"),
]
GROUP_COLOR = {"base": "tab:gray", "size": "tab:blue",
               "speed": "tab:green", "direction": "tab:red"}


def load(cond, seed):
    p = Path(f"Runs/stimvar_{LABEL}_seed{seed}_{cond}/artifacts/seed_{seed}_results.json")
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    out_dir = Path("Picture"); out_dir.mkdir(exist_ok=True)

    sig = {c: [] for c, _, _ in CONDS}
    val = {c: [] for c, _, _ in CONDS}
    for c, _, _ in CONDS:
        for s in SEEDS:
            r = load(c, s)
            if r is None:
                continue
            sig[c].append(r["sigma_theta_rad"])
            val[c].append(r["valid_fraction"])

    labels = [lbl for _, lbl, _ in CONDS]
    colors = [GROUP_COLOR[g] for _, _, g in CONDS]
    keys = [c for c, _, _ in CONDS]
    sig_mean = [np.mean(sig[c]) if sig[c] else np.nan for c in keys]
    sig_sd = [np.std(sig[c]) if sig[c] else np.nan for c in keys]
    val_mean = [np.mean(val[c]) if val[c] else np.nan for c in keys]
    val_sd = [np.std(val[c]) if val[c] else np.nan for c in keys]
    base_sig = sig_mean[keys.index("base")]

    print(f"{'condition':>12} | {'n':>2} | {'sigma_theta (mean±SD)':>22} | {'valid_fraction':>16}")
    for c, lbl, _ in CONDS:
        n = len(sig[c])
        sm = np.mean(sig[c]) if sig[c] else float("nan")
        ss = np.std(sig[c]) if sig[c] else float("nan")
        vm = np.mean(val[c]) if val[c] else float("nan")
        vs = np.std(val[c]) if val[c] else float("nan")
        print(f"{c:>12} | {n:>2} | {sm:8.3f} ± {ss:5.3f}       | {vm:6.3f} ± {vs:5.3f}")

    x = np.arange(len(CONDS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # sigma_theta (lower = sharper map)
    ax1.bar(x, sig_mean, yerr=sig_sd, color=colors, edgecolor="black", capsize=4, alpha=0.85)
    for i, c in enumerate(keys):
        ax1.plot(np.full(len(sig[c]), x[i]), sig[c], "o", color="black", ms=4, alpha=0.5)
    ax1.axhline(base_sig, color="tab:gray", ls="--", lw=1.5, label=f"baseline = {base_sig:.2f}")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel(r"$\sigma_\theta$ (rad) — lower = sharper map")
    ax1.set_title("Map sharpness vs test stimulus")
    ax1.legend(); ax1.grid(alpha=0.3, axis="y")

    # valid_fraction (higher = more decodable positions)
    ax2.bar(x, val_mean, yerr=val_sd, color=colors, edgecolor="black", capsize=4, alpha=0.85)
    for i, c in enumerate(keys):
        ax2.plot(np.full(len(val[c]), x[i]), val[c], "o", color="black", ms=4, alpha=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("valid_fraction — higher = better")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Decodable fraction vs test stimulus")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Step 2 (Q4): does the map hold under varied test stimuli?  "
        f"(topo=0.20, {len(SEEDS)} seeds, extract-mode)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "stimvar_summary.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
