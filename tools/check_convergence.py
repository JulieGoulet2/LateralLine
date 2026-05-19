#!/usr/bin/env python3
"""
tools/check_convergence.py
--------------------------
Decide whether a training run has converged.

Two tests on the saved `mid_checkpoint.npz`:

  1. PLATEAU
     Compute Δ_σ_θ = mean(σ_θ over last 3 PV checkpoints)
                   − mean(σ_θ over checkpoints 4-6 before that).
     If Δ_σ_θ > +0.02 rad (still meaningfully *decreasing* in the last
     ~1500 trials), the run has NOT plateaued.

  2. STABILIZATION
     The same heuristic used at end-of-training by
     `ll_stdp_brian2.estimate_stabilization_time()` — first checkpoint
     where mean |Δw| < 0.5 % of weight range for 4 consecutive
     checkpoints. We require this checkpoint to exist (i.e. not None /
     not the very last checkpoint of the run).

A run is CONVERGED iff both tests pass.

Usage:
    python tools/check_convergence.py Runs/llmon_trainnoise_*/artifacts/mid_checkpoint.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# Convergence thresholds (kept in one place so a follow-up tweak is easy).
DSIGMA_PLATEAU_RAD = 0.02
STABILIZATION_REL = 0.005  # 0.5 %
STABILIZATION_RUN = 4      # consecutive checkpoints below threshold


def _classify(ckpt_path: Path) -> tuple[str, dict]:
    """Return (VERDICT, details) for one mid_checkpoint.npz."""
    raw = np.load(ckpt_path, allow_pickle=False)

    pv = raw["pv_sigma_theta_series"] if "pv_sigma_theta_series" in raw.files else None
    wda = raw["w_mean_abs_delta_series"] if "w_mean_abs_delta_series" in raw.files else None

    if pv is None or wda is None:
        return "UNKNOWN", {"reason": "missing series in checkpoint"}

    pv = np.asarray(pv, dtype=float)
    wda = np.asarray(wda, dtype=float)

    details = {"n_pv_ckpt": int(pv.size), "n_w_ckpt": int(wda.size)}

    # Plateau test on σ_θ.
    plateau_ok = True
    if pv.size < 6:
        plateau_ok = False
        details["plateau"] = f"only {pv.size} PV checkpoints; need ≥ 6"
    else:
        recent = float(np.mean(pv[-3:]))
        prior = float(np.mean(pv[-6:-3]))
        # If σ_θ is still going DOWN, prior > recent → delta > 0.
        delta = prior - recent
        details["plateau_delta_rad"] = delta
        details["plateau_recent_sigma"] = recent
        details["plateau_prior_sigma"] = prior
        plateau_ok = delta <= DSIGMA_PLATEAU_RAD

    # Stabilization test on |Δw|.
    if wda.size < STABILIZATION_RUN + 1:
        stab_ok = False
        details["stabilization"] = f"only {wda.size} weight checkpoints; need ≥ {STABILIZATION_RUN + 1}"
    else:
        # 0.5 % of weight range — same convention as estimate_stabilization_time().
        # The range is approximated from observed |Δw| max; if we have a wmax
        # field in the checkpoint we could use it, but a relative threshold on
        # the recent baseline is sufficient and robust.
        thresh = float(STABILIZATION_REL * np.max(wda))
        # First index where the next RUN values are all below thresh.
        idx_stab = None
        for i in range(0, wda.size - STABILIZATION_RUN + 1):
            if np.all(wda[i:i + STABILIZATION_RUN] < thresh):
                idx_stab = i
                break
        if idx_stab is None or idx_stab >= wda.size - STABILIZATION_RUN:
            stab_ok = False
            details["stabilization"] = "never stable over 4 consecutive checkpoints (or only at the very last)"
        else:
            stab_ok = True
            details["stabilization_first_idx"] = int(idx_stab)

    verdict = "CONVERGED" if (plateau_ok and stab_ok) else "NOT_CONVERGED"
    details["plateau_ok"] = plateau_ok
    details["stabilization_ok"] = stab_ok
    return verdict, details


def main(paths: list[str]) -> int:
    any_not_converged = False
    for p in paths:
        path = Path(p)
        if not path.is_file():
            print(f"[skip] {path}: not a file")
            continue
        try:
            verdict, det = _classify(path)
        except Exception as e:
            print(f"[error] {path}: {e}")
            any_not_converged = True
            continue

        run = path.parent.parent.name
        if verdict == "CONVERGED":
            print(f"  CONVERGED  {run}  "
                  f"σ_θ_recent={det.get('plateau_recent_sigma', float('nan')):.3f} rad  "
                  f"plateau_Δ={det.get('plateau_delta_rad', float('nan')):+.3f}  "
                  f"stab_at_ckpt={det.get('stabilization_first_idx', -1)}")
        else:
            any_not_converged = True
            print(f"  NOT_CONVERGED  {run}")
            for k, v in det.items():
                print(f"      {k}: {v}")

    # Non-zero exit if any run failed — useful for shell scripts.
    return 1 if any_not_converged else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
