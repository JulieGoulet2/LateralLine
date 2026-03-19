import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2
import argparse

from ll_stdp_brian2 import NetworkParams, apply_model_mode, run_spatial_two_stage_model


def main():
    ap = argparse.ArgumentParser(description="Diagnose TS tuning during test sweep.")
    ap.add_argument("--mode", default="ll_thesis", choices=["ll_thesis", "ll_fast"])
    ap.add_argument("--ll-rate-mode", default="modulation", choices=["raw", "modulation"])
    ap.add_argument("--bg-rate-ts-hz", type=float, default=None)
    ap.add_argument("--bg-w-ts-mv", type=float, default=None)
    ap.add_argument("--ts-local-inh-peak-mv", type=float, default=None)
    ap.add_argument("--use-ts-feedback-inh", action="store_true")
    ap.add_argument("--ts-feedback-inh-mv", type=float, default=None)
    ap.add_argument("--examples", type=int, default=16)
    args = ap.parse_args()

    # Use preset params, but FREEZE MON->TS STDP so we just measure tuning.
    params = apply_model_mode(NetworkParams(), args.mode)
    params.ll_rate_mode = args.ll_rate_mode
    params.ll_mon_use_stdp = False  # LL->MON fixed (biology)
    params.mon_ts_apre = 0.0        # freeze MON->TS learning
    params.mon_ts_apost = 0.0

    # Turn OFF all TS inhibition for the diagnostic.
    params.use_ts_feedback_inh = False
    params.global_inh_to_ts_mV = 0.0
    params.ts_to_global_inh_p = 0.0
    params.ts_local_inh_peak_mV = 0.0
    if args.bg_rate_ts_hz is not None:
        params.bg_rate_ts_hz = float(max(0.0, args.bg_rate_ts_hz))
    if args.bg_w_ts_mv is not None:
        params.bg_w_ts_mV = float(max(0.0, args.bg_w_ts_mv))
    if args.ts_local_inh_peak_mv is not None:
        params.ts_local_inh_peak_mV = float(max(0.0, args.ts_local_inh_peak_mv))
    if args.use_ts_feedback_inh:
        params.use_ts_feedback_inh = True
    if args.ts_feedback_inh_mv is not None:
        params.global_inh_to_ts_mV = float(max(0.0, args.ts_feedback_inh_mv))

    print("Running model once to collect TS spikes...")
    result = run_spatial_two_stage_model(params)

    sp_ts = result["sp_ts"]
    tsim = result["test_sim"]
    t_train = result["train_duration_s"]

    # Test trajectory (0–4 cm path).
    t_test = np.asarray(tsim["t_s"], dtype=float)
    x_test = np.asarray(tsim["X_cm"], dtype=float)

    # TS spikes in absolute time.
    ts_t_abs = np.asarray(sp_ts.t / b2.second, dtype=float)
    ts_i = np.asarray(sp_ts.i, dtype=int)

    # Keep ONLY spikes during the test window: [t_train, t_end].
    t_end = result["total_duration_s"]
    mtest = (ts_t_abs >= t_train) & (ts_t_abs <= t_end)
    ts_t = ts_t_abs[mtest] - t_train
    ts_i = ts_i[mtest]

    if ts_t.size == 0:
        print("No TS spikes in test window – TS is effectively silent during the 4 cm sweep.")
        return

    p = result["params"]
    dt = float(max(p.dt_s, 1e-4))
    n_t = t_test.size

    # Build instantaneous TS rates on the same time grid as test_sim.
    rates = np.zeros((n_t, p.n_ts), dtype=float)
    k = np.floor(ts_t / dt).astype(int)
    valid = (k >= 0) & (k < n_t) & (ts_i >= 0) & (ts_i < p.n_ts)
    if np.any(valid):
        np.add.at(rates, (k[valid], ts_i[valid]), 1.0 / dt)

    # Bin by position along the 4 cm test path.
    n_pos_bins = min(40, n_t)
    x_edges = np.linspace(x_test.min(), x_test.max(), n_pos_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    tuning = np.zeros((p.n_ts, n_pos_bins), dtype=float)
    for b in range(n_pos_bins):
        mb = (x_test >= x_edges[b]) & (x_test < x_edges[b + 1])
        if not np.any(mb):
            continue
        tuning[:, b] = rates[mb, :].mean(axis=0)

    # Plot tuning for a subset of TS neurons.
    ts_indices = np.linspace(0, p.n_ts - 1, num=min(int(args.examples), p.n_ts), dtype=int)
    fig, axes = plt.subplots(len(ts_indices), 1, figsize=(8, 2.0 * len(ts_indices)), sharex=True)
    if len(ts_indices) == 1:
        axes = [axes]
    for ax, j in zip(axes, ts_indices):
        ax.plot(x_centers, tuning[j, :], "-o", ms=3)
        ax.set_ylabel(f"TS {j}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("position x during test (cm)")
    fig.suptitle(f"TS tuning curves vs x (test sweep, STDP frozen, mode={args.mode})", y=1.02)
    fig.tight_layout()
    out_path = "Picture/diagnose_ts_tuning.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print("Saved TS tuning figure:", out_path)


if __name__ == "__main__":
    main()