from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class StimulusParams:
    # Geometry and kinematics (cm, s)
    sphere_radius_cm: float = 0.5
    lateral_line_length_cm: float = 4.0
    mu_distance_cm: float = 1.5
    sigma_distance_cm: float = 0.6
    mu_speed_cm_s: float = 5.0
    sigma_speed_cm_s: float = 1.0

    # Firing rate model (Hz)
    r0_hz: float = 40.0
    rmax_hz: float = 200.0
    # Table gives A = 30000 m^-1; in cm-units that is 300 cm^-1.
    A_per_cm: float = 300.0

    # Correlated noise model (Hz, s, cm)
    sigma_noise_hz: float = 10.0
    tau_noise_s: float = 0.1
    l_noise_cm: float = 1.0


def hydrodynamic_velocity_parallel(
    xi_cm: np.ndarray,
    yi_cm: np.ndarray,
    X_cm: float,
    Y_cm: float,
    U_cm_s: float,
    R_cm: float,
    eX: float = 1.0,
    eY: float = 0.0,
    sx: float = 1.0,
    sy: float = 0.0,
) -> np.ndarray:
    """
    Compute v_parallel at each neuromast.
    eX,eY: stimulus direction unit vector.
    sx,sy: neuromast sensitivity axis unit vector.
    """
    dx = xi_cm - X_cm
    dy = yi_cm - Y_cm
    rho2 = dx * dx + dy * dy
    rho2 = np.maximum(rho2, 1e-9)
    denom = np.power(rho2, 2.5)

    common = -0.5 * U_cm_s * (R_cm**3) / denom
    dphi_dxi = common * ((2.0 * dx * dx - dy * dy) * eX + 3.0 * dx * dy * eY)
    dphi_dyi = common * (3.0 * dx * dy * eX - (dx * dx - 2.0 * dy * dy) * eY)
    return dphi_dxi * sx + dphi_dyi * sy


def simulate_lateral_line(
    duration_s: float = 1.2,
    dt_s: float = 0.001,
    n_neuromasts: int = 20,
    seed: int = 0,
    params: StimulusParams = StimulusParams(),
    fixed_distance_cm: float | None = None,
    direction: float | None = None,
    fixed_speed_cm_s: float | None = None,
    x0_cm: float | None = None,
):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt_s)
    n_t = t.size

    # 1D arrangement along fish body; y = 0 for all superficial neuromasts.
    xi = np.linspace(0.0, params.lateral_line_length_cm, n_neuromasts)
    yi = np.zeros_like(xi)

    # Draw one stimulus trajectory for this trial.
    if fixed_speed_cm_s is None:
        U = max(0.1, rng.normal(params.mu_speed_cm_s, params.sigma_speed_cm_s))
    else:
        U = max(0.1, float(fixed_speed_cm_s))

    if fixed_distance_cm is None:
        D = rng.normal(params.mu_distance_cm, params.sigma_distance_cm)
    else:
        D = float(fixed_distance_cm)

    if direction is None:
        direction = 1.0 if rng.random() < 0.5 else -1.0
    else:
        direction = 1.0 if direction >= 0 else -1.0
    eX, eY = direction, 0.0
    if x0_cm is None:
        X0 = -1.0 if direction > 0 else params.lateral_line_length_cm + 1.0
    else:
        X0 = float(x0_cm)
    X = X0 + U * eX * t
    Y = np.full_like(t, D)

    # Spatial correlation matrix R_ij and its Cholesky factor.
    dist = xi[:, None] - xi[None, :]
    R = np.exp(-0.5 * (dist / params.l_noise_cm) ** 2)
    L = np.linalg.cholesky(R + 1e-12 * np.eye(n_neuromasts))

    # Correlated OU noise process eta_i(t).
    eta = np.zeros(n_neuromasts)
    eta_t = np.zeros((n_t, n_neuromasts))
    noise_gain = np.sqrt(2.0 * (params.sigma_noise_hz**2) * dt_s / params.tau_noise_s)

    # Output rates.
    v_drive = np.zeros((n_t, n_neuromasts))
    rates = np.zeros((n_t, n_neuromasts))

    for k in range(n_t):
        z = rng.standard_normal(n_neuromasts)
        xi_corr = L @ z
        eta += dt_s * (-eta / params.tau_noise_s) + noise_gain * xi_corr
        eta_t[k] = eta

        v = hydrodynamic_velocity_parallel(
            xi, yi, X[k], Y[k], U, params.sphere_radius_cm, eX=eX, eY=eY, sx=1.0, sy=0.0
        )
        v_drive[k] = v
        rates[k] = np.clip(params.r0_hz + params.A_per_cm * v + eta, 0.0, params.rmax_hz)

    return {
        "t_s": t,
        "xi_cm": xi,
        "X_cm": X,
        "Y_cm": Y,
        "U_cm_s": U,
        "D_cm": D,
        "direction": direction,
        "v_cm_s": v_drive,
        "eta_hz": eta_t,
        "rates_hz": rates,
    }


def plot_simulation(
    sim, save_path: Path | None = None, title: str | None = None, show: bool = True
):
    t = sim["t_s"]
    xi = sim["xi_cm"]
    rates = sim["rates_hz"]
    X = sim["X_cm"]
    Y = sim["Y_cm"]
    y0 = float(Y[0])
    direction_text = "head->tail (+x)" if sim["direction"] > 0 else "tail->head (-x)"

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)

    # Geometry panel to make the physical setup explicit.
    axes[0].plot(xi, np.zeros_like(xi), "ko", ms=3, label="Neuromasts (y=0)")
    axes[0].plot(X, Y, color="tab:blue", lw=2, label="Sphere trajectory")
    axes[0].scatter([X[0]], [Y[0]], color="tab:green", s=50, zorder=5, label="Start")
    axes[0].scatter([X[-1]], [Y[-1]], color="tab:red", s=50, zorder=5, label="End")
    arrow_dx = 0.7 if sim["direction"] > 0 else -0.7
    axes[0].arrow(
        X[0],
        y0,
        arrow_dx,
        0.0,
        width=0.01,
        head_width=0.12,
        head_length=0.15,
        color="tab:blue",
        length_includes_head=True,
    )
    axes[0].text(
        0.02,
        0.95,
        f"Start distance Y = {y0:.2f} cm\nDirection: {direction_text}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    axes[0].set_xlim(min(X.min(), -1.2), max(X.max(), xi[-1] + 1.2))
    y_span = max(0.8, abs(y0) + 0.6)
    axes[0].set_ylim(-0.6, y_span)
    axes[0].set_xlabel("x (cm)")
    axes[0].set_ylabel("y (cm)")
    axes[0].set_title(title if title is not None else "Hydrodynamic stimulus and lateral-line firing")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, X, lw=1.8)
    axes[1].set_ylabel("Sphere X (cm)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)

    mean_rate = rates.mean(axis=1)
    axes[2].plot(t, mean_rate, lw=1.8)
    axes[2].set_ylabel("Mean rate (Hz)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.3)

    im = axes[2].imshow(
        rates.T,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], xi[0], xi[-1]],
        vmin=0.0,
        vmax=200.0,
        cmap="viridis",
    )
    axes[3].set_ylabel("Neuromast position x (cm)")
    axes[3].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[3], label="Rate (Hz)")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_group_simulations(
    output_dir: str = "Picture",
    distances_cm: tuple[float, ...] = (0.7, 1.5, 2.3),
    directions: tuple[float, ...] = (1.0, -1.0),
    base_seed: int = 100,
    fixed_speed_cm_s: float | None = 10.0,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    params = StimulusParams()
    idx = 0
    for d in distances_cm:
        for direction in directions:
            sim = simulate_lateral_line(
                duration_s=1.2,
                dt_s=0.001,
                n_neuromasts=30,
                seed=base_seed + idx,
                params=params,
                fixed_distance_cm=d,
                direction=direction,
                fixed_speed_cm_s=fixed_speed_cm_s,
            )
            direction_name = "head_to_tail" if direction > 0 else "tail_to_head"
            speed_tag = (
                f"_u_{fixed_speed_cm_s:.1f}cm_s" if fixed_speed_cm_s is not None else "_u_random"
            )
            filename = f"stimulus_y_{d:.1f}cm_{direction_name}{speed_tag}.png"
            speed_text = (
                f"U={fixed_speed_cm_s:.1f} cm/s" if fixed_speed_cm_s is not None else "U sampled"
            )
            title = (
                f"Lateral-line response: Y={d:.1f} cm, "
                f"{direction_name.replace('_', ' ')}, {speed_text}"
            )
            plot_simulation(sim, save_path=out / filename, title=title, show=False)
            print(f"Saved: {out / filename}")
            idx += 1


if __name__ == "__main__":
    save_group_simulations()
    print("Finished group simulation. Check the Picture/ folder for annotated figures.")
