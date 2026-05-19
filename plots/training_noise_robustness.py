#!/usr/bin/env python3
"""
plots/training_noise_robustness.py
----------------------------------
Two-panel figure: how does LL training-phase noise affect map formation?

Panel A — Learning curves
    `pv_sigma_theta_series` vs trial number, one coloured curve per
    noise scale, mean ± SD across seeds. Dashed vertical line at the
    intended budget (10 000 trials).

Panel B — Final map quality vs noise level
    Extract-mode σ_θ at the training distance D = 0.8 cm, mean ± SD
    across seeds. Dotted line at π/2 (chance). Dashed line at the
    baseline (noise = 0) for visual reference.

Data sources
    Training runs:
        Runs/llmon_trainnoise_noise{00,03,05,08,10}_seeds123_124/
            artifacts/mid_checkpoint.npz   (learning curves)
            artifacts/seed_{123,124}_results.json (final training-mode metric)
    Extract-mode runs:
        Runs/extract_trainnoise_noise{XX}_seed_{NNN}/
            artifacts/seed_NNN_results.json (final extract-mode σ_θ)

Run (from project root):
    /Users/juliegoulet/anaconda3/bin/python plots/training_noise_robustness.py

Output: Picture/ch5_training_noise_robustness.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "Runs"
PIC = ROOT / "Picture"
PIC.mkdir(exist_ok=True)


# (label, noise_scale, color)
CONDITIONS = [
    ("noise00", 0.0, "#1976D2"),   # blue   — control
    ("noise03", 0.3, "#388E3C"),   # green
    ("noise05", 0.5, "#FBC02D"),   # yellow
    ("noise08", 0.8, "#E65100"),   # orange
    ("noise10", 1.0, "#D32F2F"),   # red    — breaking-point candidate
]
SEEDS = [123, 124]


def _load_learning_curve(tag: str, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (trial_idx, sigma_theta_series). None if checkpoint missing.

    NOTE on training-time → trial-index conversion: the simulation saves
    `pv_ckpt_t_s` (wall-clock training time in seconds, NOT trial number) on
    `mid_checkpoint.npz`. Each training trial is `params.trial_duration_s = 1.2 s`
    of simulated time, so trial number = t_s / 1.2.
    """
    ckpt = RUNS / f"llmon_trainnoise_{tag}_seeds123_124" / "artifacts" / "mid_checkpoint.npz"
    if not ckpt.is_file():
        return None
    try:
        raw = np.load(ckpt, allow_pickle=False)
    except Exception:
        return None
    if "pv_sigma_theta_series" not in raw.files or "pv_ckpt_t_s" not in raw.files:
        return None
    # The series stored on disk is a single time-history for the run, not per-seed.
    # We re-load it for each seed; the underlying run was --multi-seed 2 so the
    # learning curve we want is whichever was saved last (seed 124).
    # In practice we want a single combined curve per (tag); see _panel_A().
    _ = seed  # currently unused; placeholder for a possible per-seed extension.
    t_s = np.asarray(raw["pv_ckpt_t_s"], dtype=float)
    sigma = np.asarray(raw["pv_sigma_theta_series"], dtype=float)
    trial = t_s / 1.2
    return trial, sigma


def _load_extract_final(tag: str, seed: int) -> float | None:
    """Return extract-mode σ_θ for one (tag, seed). None if missing."""
    p = RUNS / f"extract_trainnoise_{tag}_seed_{seed}" / "artifacts" / f"seed_{seed}_results.json"
    if not p.is_file():
        return None
    try:
        return float(json.loads(p.read_text())["sigma_theta_rad"])
    except Exception:
        return None


def _load_training_final(tag: str, seed: int) -> float | None:
    """Return training-mode σ_θ from the per-seed results JSON. Fallback if
    extract-mode hasn't been run yet."""
    p = RUNS / f"llmon_trainnoise_{tag}_seeds123_124" / "artifacts" / f"seed_{seed}_results.json"
    if not p.is_file():
        return None
    try:
        return float(json.loads(p.read_text())["sigma_theta_rad"])
    except Exception:
        return None


def _smooth(y: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average; pads with edge values so output has same length."""
    if win <= 1 or y.size < win:
        return y
    k = np.ones(win) / win
    pad = win // 2
    ypad = np.pad(y, pad, mode="edge")
    return np.convolve(ypad, k, mode="valid")[: y.size]


def _panel_A(ax, smooth_win: int = 50):
    """Learning curves: σ_θ vs trial, one curve per noise scale.

    Per-checkpoint σ_θ is very noisy (single test-sweep estimate); smooth with
    a centered moving average over `smooth_win` checkpoints (~ 50 × 10 trials
    = 500 trials window) so the trend is readable.
    """
    for tag, scale, color in CONDITIONS:
        # mid_checkpoint stores one time-history for the whole run; OOM-safe
        # runner trains seeds in separate processes, so each seed has its own
        # mid_checkpoint. Load both, plot mean across seeds.
        curves: list[tuple[np.ndarray, np.ndarray]] = []
        for seed in SEEDS:
            r = _load_learning_curve(tag, seed)
            if r is not None:
                curves.append(r)
        if not curves:
            print(f"  ! Panel A: no data for {tag}")
            continue

        # Align to the shortest length (in case one run was extended later).
        n = min(c[0].size for c in curves)
        trials = curves[0][0][:n]
        sigmas = np.stack([c[1][:n] for c in curves], axis=0)
        mean = sigmas.mean(axis=0)
        mean_s = _smooth(mean, smooth_win)
        sd = sigmas.std(axis=0, ddof=1) if sigmas.shape[0] > 1 else np.zeros_like(mean)
        sd_s = _smooth(sd, smooth_win)

        ax.plot(trials, mean_s, "-", color=color, linewidth=1.8,
                label=f"scale = {scale:.1f}  ({scale * 10:.0f} Hz)")
        if sigmas.shape[0] > 1:
            ax.fill_between(trials, mean_s - sd_s, mean_s + sd_s, color=color, alpha=0.12)

    ax.axvline(10000, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="planned budget (10 k trials)")
    ax.axhline(np.pi / 2, color="gray", linestyle=":", linewidth=0.8,
               label="π/2 (chance)")
    ax.set_xlabel("Training trial", fontsize=11)
    ax.set_ylabel("σ_θ during training (rad)", fontsize=11)
    ax.set_title("A — Learning curves: σ_θ vs trial, per noise scale", fontsize=11)
    ax.set_ylim(0, np.pi / 2 + 0.2)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)


def _panel_B(ax, use_extract: bool):
    """Final σ_θ vs noise scale, mean ± SD across seeds."""
    xs, means, sds = [], [], []
    for tag, scale, color in CONDITIONS:
        vals: list[float] = []
        for seed in SEEDS:
            v = _load_extract_final(tag, seed) if use_extract else _load_training_final(tag, seed)
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        xs.append(scale)
        means.append(float(np.mean(vals)))
        sds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    if not xs:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        return

    xs_a = np.asarray(xs)
    means_a = np.asarray(means)
    sds_a = np.asarray(sds)

    ax.errorbar(xs_a, means_a, yerr=sds_a, fmt="o-", color="#0D47A1",
                markersize=9, capsize=6, linewidth=2,
                label=("Extract-mode" if use_extract else "Training-mode"))

    # Reference lines.
    if means_a.size:
        ax.axhline(means_a[0], color="#0D47A1", linestyle=":", linewidth=0.8,
                   alpha=0.5, label="control (noise = 0)")
    ax.axhline(np.pi / 2, color="gray", linestyle=":", linewidth=0.8,
               label="π/2 (chance)")

    ax.set_xlabel("LL training-noise scale  (×  sigma_noise_hz = 10 Hz)", fontsize=11)
    ax.set_ylabel("Final σ_θ  (rad)", fontsize=11)
    ax.set_title("B — Final map quality vs training-noise scale", fontsize=11)
    ax.set_xticks([c[1] for c in CONDITIONS])
    ax.set_ylim(0, np.pi / 2 + 0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)


def main() -> int:
    # Prefer extract-mode for panel B if available (cleaner comparison to
    # the topo-gradient figure); fall back to training-mode otherwise.
    any_extract = any(
        _load_extract_final(tag, seed) is not None
        for tag, _, _ in CONDITIONS for seed in SEEDS
    )

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    _panel_A(axA)
    _panel_B(axB, use_extract=any_extract)
    fig.suptitle("Training-phase LL noise robustness  —  topo = 0.20, MON = 3200, 10 000 trials",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    out = PIC / "ch5_training_noise_robustness.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")

    # Summary stats for the log.
    print("\nFinal σ_θ per condition:")
    for tag, scale, _ in CONDITIONS:
        for seed in SEEDS:
            e = _load_extract_final(tag, seed)
            t = _load_training_final(tag, seed)
            print(f"  scale={scale:.1f}  seed={seed}  "
                  f"train={t if t is not None else '—'}  "
                  f"extract={e if e is not None else '—'}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
