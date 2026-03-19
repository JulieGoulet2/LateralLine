import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2
import argparse

from ll_stdp_brian2 import NetworkParams, apply_model_mode, run_spatial_two_stage_model


def main():
    ap = argparse.ArgumentParser(description="Diagnose TS tuning during TRAINING.")
    ap.add_argument("--mode", default="ll_thesis", choices=["ll_thesis", "ll_fast"])
    ap.add_argument("--ll-rate-mode", default="modulation", choices=["raw", "modulation"])
    ap.add_argument("--bg-rate-ts-hz", type=float, default=None)
    ap.add_argument("--bg-w-ts-mv", type=float, default=None)
    ap.add_argument("--ts-local-inh-peak-mv", type=float, default=None)
    ap.add_argument("--disable-ts-inh", action="store_true", help="Disable TS lateral+feedback inhibition.")
    ap.add_argument("--vth-mv", type=float, default=None, help="Override neuron threshold (mV) for diagnostics.")
    args = ap.parse_args()

    # Use preset params, but don't let MON->TS STDP change during this diagnostic run.
    params = apply_model_mode(NetworkParams(), args.mode)
    params.ll_rate_mode = args.ll_rate_mode
    params.ll_mon_use_stdp = False
    params.mon_ts_apre = 0.0
    params.mon_ts_apost = 0.0

    # TS excitability overrides for the diagnostic (TRAINING).
    if args.disable_ts_inh:
        params.use_ts_feedback_inh = False
        params.global_inh_to_ts_mV = 0.0
        params.ts_to_global_inh_p = 0.0
        params.ts_local_inh_peak_mV = 0.0
    if args.ts_local_inh_peak_mv is not None:
        params.ts_local_inh_peak_mV = float(max(0.0, args.ts_local_inh_peak_mv))
    if args.bg_rate_ts_hz is not None:
        params.bg_rate_ts_hz = float(max(0.0, args.bg_rate_ts_hz))
    if args.bg_w_ts_mv is not None:
        params.bg_w_ts_mV = float(max(0.0, args.bg_w_ts_mv))
    if args.vth_mv is not None:
        params.vth_mV = float(args.vth_mv)

    print(
        "TS diag params:",
        f"bg_rate_ts_hz={params.bg_rate_ts_hz}, bg_w_ts_mV={params.bg_w_ts_mV}, "
        f"ts_local_inh_peak_mV={params.ts_local_inh_peak_mV}, "
        f"use_ts_feedback_inh={params.use_ts_feedback_inh}, global_inh_to_ts_mV={params.global_inh_to_ts_mV}, "
        f"vth_mV={params.vth_mV}"
    )

    print("Running model once to collect TS spikes during TRAINING...")
    result = run_spatial_two_stage_model(params)

    sp_ts = result["sp_ts"]
    train_x_cm = result["train_x_cm"]       # x(t) during training
    train_duration_s = result["train_duration_s"]

    # TS spikes during training only.
    ts_t = np.asarray(sp_ts.t / b2.second, dtype=float)
    ts_i = np.asarray(sp_ts.i, dtype=int)
    mtrain = (ts_t >= 0.0) & (ts_t < train_duration_s)
    ts_t = ts_t[mtrain]
    ts_i = ts_i[mtrain]

    if ts_t.size == 0:
        print("No TS spikes during training.")
        return

    p = result["params"]
    dt = float(max(p.dt_s, 1e-6))

    # Bin by training position x. Compute occupancy time per bin, then TS spike counts per bin.
    x = np.asarray(train_x_cm, dtype=float)
    n_pos_bins = 40
    x_edges = np.linspace(x.min(), x.max(), n_pos_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Occupancy (seconds) for each x bin during training.
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_pos_bins - 1)
    occ_steps = np.bincount(x_bin, minlength=n_pos_bins).astype(float)
    occ_s = occ_steps * dt
    occ_s = np.maximum(occ_s, 1e-12)

    # Map each TS spike to the x-bin at its spike time.
    t_idx = np.clip((ts_t / dt).astype(int), 0, x.size - 1)
    spike_x_bin = x_bin[t_idx]

    # Spike counts per (ts_i, x_bin).
    counts = np.zeros((p.n_ts, n_pos_bins), dtype=float)
    valid = (ts_i >= 0) & (ts_i < p.n_ts)
    if np.any(valid):
        np.add.at(counts, (ts_i[valid], spike_x_bin[valid]), 1.0)

    # Convert to rate (Hz) by dividing by occupancy time in that bin.
    tuning = counts / occ_s[None, :]

    # Quick sanity: how many TS neurons actually spiked during training?
    spike_counts_per_ts = np.bincount(ts_i[valid], minlength=p.n_ts)
    active = int(np.sum(spike_counts_per_ts > 0))
    print(f"TS active neurons during training: {active}/{p.n_ts} (showing 16 examples)")

    # Plot tuning for a subset of TS neurons.
    ts_indices = np.linspace(0, p.n_ts - 1, 16, dtype=int)
    fig, axes = plt.subplots(len(ts_indices), 1, figsize=(8, 2.0 * len(ts_indices)), sharex=True)
    if len(ts_indices) == 1:
        axes = [axes]
    for ax, j in zip(axes, ts_indices):
        ax.plot(x_centers, tuning[j, :], "-o", ms=3)
        ax.set_ylabel(f"TS {j}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("position x during TRAINING (cm)")
    fig.suptitle("TS tuning curves vs x during training (MON->TS STDP frozen)", y=1.02)
    fig.tight_layout()
    out_path = "Picture/diagnose_ts_tuning_train.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print("Saved TS training-tuning figure:", out_path)


if __name__ == "__main__":
    main()