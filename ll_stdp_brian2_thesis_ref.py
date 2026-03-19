import argparse
from pathlib import Path

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

from ll_stdp_brian2 import NetworkParams, apply_model_mode, run_spatial_two_stage_model


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
    Build a strict reference configuration for thesis-style checks.
    Keeps only core settings and disables debug hacks.
    """
    p = apply_model_mode(NetworkParams(), "ll_thesis")

    # Strict MON->TS STDP values from thesis table.
    p.mon_ts_apre = 0.02
    p.mon_ts_apost = -0.021
    p.mon_ts_wmax = 0.045
    p.mon_ts_w_init = 0.020
    p.mon_ts_w_jitter = 0.005

    # Keep baseline architecture behavior (no debug interventions).
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
        p.global_inh_to_ts_mV = float(max(0.0, ts_feedback_inh_mV))

    if distance_cm is not None:
        p.distance_cm = float(max(0.0, distance_cm))
        # If training is fixed-distance, align the band to the same value.
        if training_fixed_distance is True:
            p.training_fixed_distance = True
            p.training_distance_min_cm = float(max(0.0, distance_cm))
            p.training_distance_max_cm = float(max(0.0, distance_cm))
        elif training_fixed_distance is False:
            p.training_fixed_distance = False
    return p


def save_ts_spikes_vs_x_test(result: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sp_ts = result["sp_ts"]
    tsim = result["test_sim"]
    t_train = float(result["train_duration_s"])
    t_end = float(result["total_duration_s"])
    dt = float(result["params"].dt_s)

    ts_t = np.asarray(sp_ts.t / b2.second, dtype=float)
    ts_i = np.asarray(sp_ts.i, dtype=int)
    m = (ts_t >= t_train) & (ts_t < t_end)
    if not np.any(m):
        return

    ts_t = ts_t[m] - t_train
    ts_i = ts_i[m]
    x_test = np.asarray(tsim["X_cm"], dtype=float)
    k = np.floor(ts_t / dt).astype(int)
    valid = (k >= 0) & (k < x_test.size)
    if not np.any(valid):
        return

    x_sp = x_test[k[valid]]
    ts_i = ts_i[valid]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    ax.scatter(x_sp, ts_i, s=2.0, alpha=0.4, color="tab:green")
    ax.set_xlabel("x during test (cm)")
    ax.set_ylabel("TS index")
    ax.set_title("TS spikes vs x during test")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_pv_theta_scatter(result: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    theta_true = np.asarray(result["pv_theta_true"], dtype=float)
    theta_hat = np.asarray(result["pv_theta_hat"], dtype=float)
    m = np.isfinite(theta_true) & np.isfinite(theta_hat)
    if not np.any(m):
        return

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
    ax.scatter(theta_true[m], theta_hat[m], s=2.0, alpha=0.35, color="tab:blue")
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
    parser.add_argument("--bg-rate-ts-hz", type=float, default=None)
    parser.add_argument("--bg-w-ts-mv", type=float, default=None)
    parser.add_argument("--mon-ts-gain-mv", type=float, default=None)
    parser.add_argument("--ts-local-inh-peak-mv", type=float, default=None)
    parser.add_argument("--use-ts-feedback-inh", action="store_true")
    parser.add_argument("--ts-feedback-inh-mv", type=float, default=None)
    parser.add_argument("--distance-cm", type=float, default=None, help="Override stimulus distance cm (test and optionally training).")
    parser.add_argument(
        "--training-fixed-distance",
        action="store_true",
        help="If set with --distance-cm, use fixed-distance training at the same distance.",
    )
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
        training_fixed_distance=True if args.training_fixed_distance else None,
    )
    result = run_spatial_two_stage_model(params)

    out_dir = Path("Picture")
    save_ts_spikes_vs_x_test(result, out_dir / "thesis_ref_ts_spikes_vs_x_test.png")
    save_pv_theta_scatter(result, out_dir / "thesis_ref_pv_theta_scatter.png")

    ts_t = np.asarray(result["sp_ts"].t / b2.second, dtype=float)
    t0 = float(result["train_duration_s"])
    t1 = float(result["train_duration_s"] + result["test_duration_s"])
    n_ts_test = int(np.sum((ts_t >= t0) & (ts_t < t1)))

    n_mon_ts = len(result["mon_to_ts_i"])
    tot_s = float(result["total_duration_s"])
    mon_rate = result["sp_mon"].num_spikes / max(1e-12, result["params"].n_mon * tot_s)
    ts_rate = result["pr_ts"].smooth_rate(width=20 * b2.ms) / b2.Hz

    print("Reference run complete.")
    print(f"Spikes: LL={result['sp_ll'].num_spikes}, MON={result['sp_mon'].num_spikes}, TS={result['sp_ts'].num_spikes}")
    print(f"TS spikes during test window: {n_ts_test}")
    print(f"MON->TS synapses: {n_mon_ts}, MON rate per neuron: {mon_rate:.4f} Hz")
    print(f"Max TS population rate (20 ms smooth): {float(np.max(ts_rate)):.2f} Hz")
    print(
        f"PV map quality: sigma_theta={result['pv_sigma_theta']:.4f} rad, "
        f"delta_trial={result['pv_delta_trial']:.4f} rad, valid_fraction={result['pv_valid_fraction']:.3f}"
    )
    print("Saved:")
    print(" - Picture/thesis_ref_ts_spikes_vs_x_test.png")
    print(" - Picture/thesis_ref_pv_theta_scatter.png")


if __name__ == "__main__":
    main()
