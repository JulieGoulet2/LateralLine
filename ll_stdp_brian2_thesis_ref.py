import argparse
from pathlib import Path

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

from ll_stdp_brian2 import NetworkParams, apply_model_mode, run_spatial_two_stage_model
from stimulus import StimulusParams


def build_reference_params(
    seed: int,
    n_training_trials: int | None,
    use_ll_mon_stdp: bool,
    bg_rate_ts_hz: float | None,
    bg_w_ts_mV: float | None,
    mon_ts_gain_mV: float | None,
    ts_local_inh_peak_mV: float | None,
    use_ts_feedback_inh: bool,
    ts_feedback_inh_mV: float | None,
    distance_cm: float | None,
    training_fixed_distance: bool | None,
) -> NetworkParams:
    """
    Strict thesis-like reference configuration:
    - keep MON->TS STDP params fixed to thesis values
    - disable debug interventions
    - allow only background + inhibition + geometry overrides (as needed)
    """
    p = apply_model_mode(NetworkParams(), "ll_thesis")

    # Strict MON->TS STDP values from thesis table (to avoid drift).
    p.mon_ts_apre = 0.02
    p.mon_ts_apost = -0.021
    p.mon_ts_wmax = 0.045
    p.mon_ts_w_init = 0.020
    p.mon_ts_w_jitter = 0.005

    # Reference diagnostic geometry.
    p.silence_ts_edges = False
    p.training_x_min_cm = 0.0
    p.training_x_max_cm = 4.0
    p.test_path_cm = 4.0

    p.seed = int(seed)
    if n_training_trials is not None:
        p.n_training_trials = max(1, int(n_training_trials))

    p.ll_mon_use_stdp = bool(use_ll_mon_stdp)

    if bg_rate_ts_hz is not None:
        p.bg_rate_ts_hz = float(max(0.0, bg_rate_ts_hz))
    if bg_w_ts_mV is not None:
        p.bg_w_ts_mV = float(max(0.0, bg_w_ts_mV))
    if mon_ts_gain_mV is not None:
        p.mon_ts_gain_mV = float(max(0.0, mon_ts_gain_mV))
    if ts_local_inh_peak_mV is not None:
        p.ts_local_inh_peak_mV = float(max(0.0, ts_local_inh_peak_mV))

    if use_ts_feedback_inh:
        p.use_ts_feedback_inh = True
    if ts_feedback_inh_mV is not None:
        # In ll_stdp_brian2 this maps to global inhibition strength onto TS.
        p.global_inh_to_ts_mV = float(max(0.0, ts_feedback_inh_mV))

    if distance_cm is not None:
        p.distance_cm = float(max(0.0, distance_cm))
        if training_fixed_distance:
            p.training_fixed_distance = True
            p.training_distance_min_cm = float(max(0.0, distance_cm))
            p.training_distance_max_cm = float(max(0.0, distance_cm))

    return p


def save_ts_spikes_vs_x_test(result: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    sp_ts = result["sp_ts"]
    test_sim = result["test_sim"]

    train_duration_s = float(result["train_duration_s"])
    test_duration_s = float(result["test_duration_s"])
    t_end = train_duration_s + test_duration_s

    dt = float(p.dt_s)
    x_test = np.asarray(test_sim["X_cm"], dtype=float)

    ts_t_abs = np.asarray(sp_ts.t / b2.second, dtype=float)
    ts_i = np.asarray(sp_ts.i, dtype=int)

    m = (ts_t_abs >= train_duration_s) & (ts_t_abs < t_end)
    if not np.any(m):
        # Keep file creation predictable: make an empty plot.
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
        ax.set_xlabel("stimulus position x (cm) during test (in [0,4])")
        ax.set_ylabel("TS index")
        ax.set_title("TS spikes vs x during test window (no spikes)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    ts_t_rel = ts_t_abs[m] - train_duration_s
    ts_i = ts_i[m]

    # Map each spike time to the nearest test x sample using dt.
    k = np.floor(ts_t_rel / dt).astype(int)
    valid = (k >= 0) & (k < x_test.size) & (ts_i >= 0) & (ts_i < p.n_ts)
    if not np.any(valid):
        return

    x_sp = x_test[k[valid]]
    ts_i = ts_i[valid]

    in_range = (x_sp >= 0.0) & (x_sp <= float(StimulusParams().lateral_line_length_cm))
    x_sp = x_sp[in_range]
    ts_i = ts_i[in_range]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    ax.scatter(x_sp, ts_i, s=2.0, alpha=0.4, color="tab:green")
    ax.set_xlabel("stimulus position x (cm) during test (in [0,4])")
    ax.set_ylabel("TS index")
    ax.set_title("TS spikes vs x during test window")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_pv_theta_scatter(result: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    theta_true = np.asarray(result.get("pv_theta_true", []), dtype=float)
    theta_hat = np.asarray(result.get("pv_theta_hat", []), dtype=float)
    if theta_true.size == 0 or theta_hat.size == 0:
        return

    m = np.isfinite(theta_true) & np.isfinite(theta_hat)
    if not np.any(m):
        return

    theta_true = theta_true[m]
    theta_hat = theta_hat[m]

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.3))
    ax.scatter(theta_true, theta_hat, s=2.0, alpha=0.35, color="tab:blue")
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], color="tab:orange", lw=2.0)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_xlabel("theta_true (rad)")
    ax.set_ylabel("theta_hat (rad)")
    ax.set_title("PV decode during test")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Clean thesis-reference run for LL->MON->TS model.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-training-trials", type=int, default=None)
    parser.add_argument("--use-ll-mon-stdp", action="store_true")

    # Inhibition/background knobs (as used in your previous workflow).
    parser.add_argument("--bg-rate-ts-hz", type=float, default=None)
    parser.add_argument("--bg-w-ts-mv", type=float, default=None)
    parser.add_argument("--mon-ts-gain-mv", type=float, default=None)
    parser.add_argument("--ts-local-inh-peak-mv", type=float, default=None)
    parser.add_argument("--use-ts-feedback-inh", action="store_true")
    parser.add_argument("--ts-feedback-inh-mv", type=float, default=None)

    # Geometry override.
    parser.add_argument("--distance-cm", type=float, default=None)
    parser.add_argument("--training-fixed-distance", action="store_true")

    args = parser.parse_args()

    params = build_reference_params(
        seed=args.seed,
        n_training_trials=args.n_training_trials,
        use_ll_mon_stdp=bool(args.use_ll_mon_stdp),
        bg_rate_ts_hz=args.bg_rate_ts_hz,
        bg_w_ts_mV=args.bg_w_ts_mv,
        mon_ts_gain_mV=args.mon_ts_gain_mv,
        ts_local_inh_peak_mV=args.ts_local_inh_peak_mv,
        use_ts_feedback_inh=bool(args.use_ts_feedback_inh),
        ts_feedback_inh_mV=args.ts_feedback_inh_mv,
        distance_cm=args.distance_cm,
        training_fixed_distance=bool(args.training_fixed_distance),
    )

    result = run_spatial_two_stage_model(params)

    out_dir = Path("Picture")
    save_ts_spikes_vs_x_test(result, out_dir / "thesis_ref_ts_spikes_vs_x_test.png")
    save_pv_theta_scatter(result, out_dir / "thesis_ref_pv_theta_scatter.png")

    print("Reference run complete.")
    ts_t = np.asarray(result["sp_ts"].t / b2.second, dtype=float)
    t0 = float(result["train_duration_s"])
    t1 = float(result["train_duration_s"] + result["test_duration_s"])
    n_ts_test = int(np.sum((ts_t >= t0) & (ts_t < t1)))

    print(
        f"Spikes: LL={result['sp_ll'].num_spikes}, MON={result['sp_mon'].num_spikes}, TS={result['sp_ts'].num_spikes}"
    )
    print(f"TS spikes during test window: {n_ts_test}")
    print(
        "PV map quality: "
        f"sigma_theta={result['pv_sigma_theta']:.4f} rad, "
        f"delta_trial={result['pv_delta_trial']:.4f} rad, "
        f"valid_fraction={result['pv_valid_fraction']:.3f}"
    )
    print("Saved:")
    print(" - Picture/thesis_ref_ts_spikes_vs_x_test.png")
    print(" - Picture/thesis_ref_pv_theta_scatter.png")


if __name__ == "__main__":
    main()

