from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from params import NetworkParams


def _eval_window_cm(params: NetworkParams) -> tuple[float, float] | None:
    emin, emax = params.eval_x_min_cm, params.eval_x_max_cm
    if emin is None and emax is None:
        return None
    if emin is None or emax is None:
        raise ValueError("eval_x_min_cm and eval_x_max_cm must both be set or both None")
    if float(emax) <= float(emin):
        raise ValueError("eval_x_max_cm must be greater than eval_x_min_cm")
    return (float(emin), float(emax))


def _test_x_local_bins(
    x_test: np.ndarray,
    p: NetworkParams,
    n_pos_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Linear test positions -> local x for binning [0, span].
    Returns (x_local, ok_time, x_edges, x_centers, xlabel_tag).
    """
    w = _eval_window_cm(p)
    n_t = int(x_test.size)
    if w is not None:
        emin, emax = w
        span = max(float(emax - emin), 1e-12)
        ok = (x_test >= emin) & (x_test <= emax)
        x_local = x_test - emin
        x_edges = np.linspace(0.0, span, n_pos_bins + 1)
        tag = f"eval window [{emin:.3f},{emax:.3f}] cm, local x"
    else:
        ok = np.ones(n_t, dtype=bool)
        xmin = float(np.min(x_test))
        span = max(float(np.ptp(x_test)), 1e-12)
        x_local = x_test - xmin
        x_edges = np.linspace(0.0, span, n_pos_bins + 1)
        tag = "linear x (full test path)"
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    return x_local, ok, x_edges, x_centers, tag
