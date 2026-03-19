from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stimulus import StimulusParams, simulate_lateral_line


def _run_one(
    seed: int,
    distance_cm: float,
    direction: float,
    speed_cm_s: float,
    n_neuromasts: int,
    dt_s: float,
    duration_s: float,
):
    return simulate_lateral_line(
        duration_s=duration_s,
        dt_s=dt_s,
        n_neuromasts=n_neuromasts,
        seed=seed,
        params=StimulusParams(),
        fixed_distance_cm=distance_cm,
        direction=direction,
        fixed_speed_cm_s=speed_cm_s,
    )


def _metrics(sim: dict):
    rates = sim["rates_hz"]
    return {
        "mean_rate_hz": float(np.mean(rates)),
        "peak_rate_hz": float(np.max(rates)),
        "std_rate_hz": float(np.std(rates)),
    }


def _save_distance_validation(out_dir: Path, distances_cm: list[float], speed_cm_s: float):
    """
    Stage A check:
    As distance increases, the stimulus modulation should weaken.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    rows = []

    for k, d in enumerate(distances_cm):
        sim = _run_one(
            seed=100 + k,
            distance_cm=d,
            direction=1.0,
            speed_cm_s=speed_cm_s,
            n_neuromasts=100,
            dt_s=0.001,
            duration_s=1.2,
        )
        mean_over_time = np.mean(sim["rates_hz"], axis=1)
        ax.plot(sim["t_s"], mean_over_time, lw=1.5, label=f"Y={d:.1f} cm")

        m = _metrics(sim)
        rows.append((d, m["mean_rate_hz"], m["peak_rate_hz"], m["std_rate_hz"]))

    ax.set_title(f"Stage A: mean LL rate vs time (U={speed_cm_s:.1f} cm/s, head->tail)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean LL rate (Hz)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stageA_distance_validation.png", dpi=180)
    plt.close(fig)

    csv_path = out_dir / "stageA_distance_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("distance_cm,mean_rate_hz,peak_rate_hz,std_rate_hz\n")
        for d, m, p, s in rows:
            f.write(f"{d:.3f},{m:.6f},{p:.6f},{s:.6f}\n")


def _save_distance_decodability(
    out_dir: Path, distances_cm: list[float], speed_cm_s: float, n_seeds: int = 12
):
    """
    Stage A.2:
    Quantify how stimulus decodability falls with distance.
    """
    params = StimulusParams()
    rows = []
    for k, d in enumerate(distances_cm):
        peak_mod_vals = []
        rms_mod_vals = []
        peak_snr_vals = []
        rms_snr_vals = []

        for s in range(n_seeds):
            sim = _run_one(
                seed=900 + 100 * k + s,
                distance_cm=d,
                direction=1.0,
                speed_cm_s=speed_cm_s,
                n_neuromasts=100,
                dt_s=0.001,
                duration_s=1.2,
            )
            rates = sim["rates_hz"]
            # Mean across neuromasts each time step.
            mean_t = np.mean(rates, axis=1)
            # Modulation above spontaneous baseline.
            peak_mod_hz = float(max(0.0, np.max(mean_t) - params.r0_hz))
            rms_mod_hz = float(np.sqrt(np.mean((mean_t - params.r0_hz) ** 2)))
            snr_peak = peak_mod_hz / max(1e-9, params.sigma_noise_hz)
            snr_rms = rms_mod_hz / max(1e-9, params.sigma_noise_hz)

            peak_mod_vals.append(peak_mod_hz)
            rms_mod_vals.append(rms_mod_hz)
            peak_snr_vals.append(snr_peak)
            rms_snr_vals.append(snr_rms)

        rows.append(
            (
                d,
                float(np.mean(peak_mod_vals)),
                float(np.mean(rms_mod_vals)),
                float(np.mean(peak_snr_vals)),
                float(np.mean(rms_snr_vals)),
                float(np.std(peak_mod_vals)),
                float(np.std(rms_mod_vals)),
                float(np.std(peak_snr_vals)),
                float(np.std(rms_snr_vals)),
            )
        )

    d_arr = np.array([r[0] for r in rows], dtype=float)
    peak_arr = np.array([r[1] for r in rows], dtype=float)
    rms_arr = np.array([r[2] for r in rows], dtype=float)
    snr_peak_arr = np.array([r[3] for r in rows], dtype=float)
    snr_rms_arr = np.array([r[4] for r in rows], dtype=float)
    peak_std = np.array([r[5] for r in rows], dtype=float)
    rms_std = np.array([r[6] for r in rows], dtype=float)
    snr_peak_std = np.array([r[7] for r in rows], dtype=float)
    snr_rms_std = np.array([r[8] for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(d_arr, peak_arr, "o-", lw=1.8, label="Peak modulation")
    ax[0].plot(d_arr, rms_arr, "s-", lw=1.8, label="RMS modulation")
    ax[0].fill_between(d_arr, peak_arr - peak_std, peak_arr + peak_std, alpha=0.15)
    ax[0].fill_between(d_arr, rms_arr - rms_std, rms_arr + rms_std, alpha=0.15)
    ax[0].set_title("Distance vs LL modulation")
    ax[0].set_xlabel("Sphere distance Y (cm)")
    ax[0].set_ylabel("Modulation above r0 (Hz)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(d_arr, snr_peak_arr, "o-", lw=1.8, label="Peak SNR")
    ax[1].plot(d_arr, snr_rms_arr, "s-", lw=1.8, label="RMS SNR")
    ax[1].fill_between(d_arr, snr_peak_arr - snr_peak_std, snr_peak_arr + snr_peak_std, alpha=0.15)
    ax[1].fill_between(d_arr, snr_rms_arr - snr_rms_std, snr_rms_arr + snr_rms_std, alpha=0.15)
    ax[1].axhline(1.0, color="tab:red", ls="--", lw=1.2, label="SNR = 1")
    ax[1].set_title("Distance vs effective SNR")
    ax[1].set_xlabel("Sphere distance Y (cm)")
    ax[1].set_ylabel("SNR (modulation / noise sigma)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "stageA_distance_decodability.png", dpi=180)
    plt.close(fig)

    csv_path = out_dir / "stageA_distance_decodability.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "distance_cm,peak_mod_hz_mean,rms_mod_hz_mean,peak_snr_mean,rms_snr_mean,"
            "peak_mod_hz_std,rms_mod_hz_std,peak_snr_std,rms_snr_std\n"
        )
        for d, pm, rm, sp, sr, pstd, rstd, spstd, srstd in rows:
            f.write(f"{d:.3f},{pm:.6f},{rm:.6f},{sp:.6f},{sr:.6f},{pstd:.6f},{rstd:.6f},{spstd:.6f},{srstd:.6f}\n")


def _save_group_examples(out_dir: Path):
    distances_cm = [0.7, 1.5, 2.3]
    directions = [1.0, -1.0]
    speed_cm_s = 10.0

    for i, d in enumerate(distances_cm):
        for j, direction in enumerate(directions):
            sim = _run_one(
                seed=500 + 10 * i + j,
                distance_cm=d,
                direction=direction,
                speed_cm_s=speed_cm_s,
                n_neuromasts=30,
                dt_s=0.001,
                duration_s=1.2,
            )

            t = sim["t_s"]
            xi = sim["xi_cm"]
            X = sim["X_cm"]
            Y = sim["Y_cm"]
            rates = sim["rates_hz"]

            fig, ax = plt.subplots(3, 1, figsize=(10, 9))
            dir_txt = "head->tail" if direction > 0 else "tail->head"

            ax[0].plot(xi, np.zeros_like(xi), "k.", ms=3, label="Neuromasts")
            ax[0].plot(X, Y, lw=2, color="tab:blue", label="Sphere path")
            ax[0].scatter([X[0]], [Y[0]], color="tab:green", s=40, label="Start")
            ax[0].scatter([X[-1]], [Y[-1]], color="tab:red", s=40, label="End")
            ax[0].set_title(f"Stage A geometry | Y={d:.1f} cm | {dir_txt} | U={speed_cm_s:.1f} cm/s")
            ax[0].set_xlabel("x (cm)")
            ax[0].set_ylabel("y (cm)")
            ax[0].grid(alpha=0.3)
            ax[0].legend(fontsize=8)

            ax[1].plot(t, np.mean(rates, axis=1), color="tab:purple")
            ax[1].set_xlabel("Time (s)")
            ax[1].set_ylabel("Mean rate (Hz)")
            ax[1].grid(alpha=0.3)

            im = ax[2].imshow(
                rates.T,
                aspect="auto",
                origin="lower",
                extent=[t[0], t[-1], xi[0], xi[-1]],
                vmin=0.0,
                vmax=200.0,
                cmap="viridis",
            )
            ax[2].set_xlabel("Time (s)")
            ax[2].set_ylabel("Neuromast x (cm)")
            fig.colorbar(im, ax=ax[2], label="Rate (Hz)")

            fig.tight_layout()
            name = f"stageA_example_y_{d:.1f}_{'ht' if direction > 0 else 'th'}.png"
            fig.savefig(out_dir / name, dpi=180)
            plt.close(fig)


def main():
    out_dir = Path("Picture/StageA")
    out_dir.mkdir(parents=True, exist_ok=True)

    dvals = [0.6, 1.0, 1.5, 2.0, 2.5]
    _save_distance_validation(out_dir, distances_cm=dvals, speed_cm_s=10.0)
    _save_distance_decodability(out_dir, distances_cm=dvals, speed_cm_s=10.0)
    _save_group_examples(out_dir)

    print(f"Stage A outputs saved in: {out_dir}")
    print(f"- {out_dir / 'stageA_distance_validation.png'}")
    print(f"- {out_dir / 'stageA_distance_metrics.csv'}")
    print(f"- {out_dir / 'stageA_distance_decodability.png'}")
    print(f"- {out_dir / 'stageA_distance_decodability.csv'}")
    print("- stageA_example_*.png")


if __name__ == "__main__":
    main()
