"""
Network parameter dataclass and named presets for the lateral-line STDP model.

Pure Python — no Brian2, no numpy, no plotting. Lives separately from
`ll_stdp_brian2.py` so test files, plot helpers, and external callers can
load configuration without pulling in the full simulation stack.

Every NetworkParams field has an inline comment that explains its role and
points to the corresponding CLI flag in `ll_stdp_brian2.py:main()`.
"""
from dataclasses import dataclass, replace


@dataclass
class NetworkParams:
    # --- Network architecture ---
    n_ll: int = 100    # number of LL afferent neurons (one per neuromast receptor)  --n-ll
    n_mon: int = 3200  # number of MON neurons (expansion / intermediate layer)      --n-mon
    n_ts: int = 300    # number of TS neurons (map output layer)                      --n-ts

    # --- Simulation timestep and trial structure ---
    dt_s: float = 0.001                          # Brian2 integration timestep (seconds)                          --dt-s
    trial_duration_s: float = 1.2                # duration of one training trial (seconds)                       --trial-duration-s
    n_training_trials: int = 200                 # total number of training trials                                 --n-training-trials
    checkpoint_trials: int = 10                  # record weight statistics every N trials                        --checkpoint-trials
    pv_eval_every_checkpoints: int = 2           # run PV map-quality metric every N checkpoints                  --pv-eval-every-checkpoints
    checkpoint_save_every_n_checkpoints: int = 10  # flush mid_checkpoint.npz to disk every N checkpoints        --checkpoint-save-every-n

    training_position_hold_s: float = 0.05       # how long each snapshot position is held during training (s)    --training-position-hold-s
    training_noise_scale_early: float = 0.0      # LL noise std during early training (fraction of sigma_noise)  --training-noise-early
    training_noise_scale_late: float = 0.0       # LL noise std during late training (fraction of sigma_noise)   --training-noise-late
    training_noise_switch_fraction: float = 0.5  # training fraction where noise transitions early→late (0..1)   --training-noise-switch
    training_ordered_sweeps: bool = True          # True: sweep x forward/backward; False: random shuffle         --training-ordered-sweeps / --no-training-ordered-sweeps
    training_fixed_distance: bool = True          # fix sphere distance at mu_distance_cm (no distance jitter)    --training-fixed-distance / --no-training-fixed-distance
    training_bidirectional: bool = False          # alternate sphere direction each snapshot                       --training-bidirectional / --no-training-bidirectional
    training_distance_min_cm: float = 0.8         # minimum sphere distance during training (cm)                  --training-distance-min-cm
    training_distance_max_cm: float = 1.8         # maximum sphere distance during training (cm)                  --training-distance-max-cm

    # --- Stimulus and test path ---
    speed_cm_s: float = 5.0       # sphere speed (cm/s); test duration = test_path_cm / speed_cm_s   --speed-cm-s
    distance_cm: float = 1.5      # sphere distance from lateral line during the test sweep (cm)       --distance-cm
    direction: float = 1.0        # sphere travel direction (+1 = forward along x, -1 = backward)     --direction
    ll_rate_mode: str = "raw"     # "raw": use LL rates from model; "modulation": subtract r0         --ll-rate-mode
    ll_rate_baseline_subtract_hz: float = 0.0  # additional baseline subtracted from LL rates before Poisson drive (Hz)  --ll-baseline-subtract-hz
    ll_rate_gain: float = 1.0                  # multiplicative gain on LL rates before Poisson drive                     --ll-rate-gain
    ll_body_length_cm: float = 4.0             # physical length of the lateral line / fish body axis (cm)                --ll-body-length-cm
    test_path_cm: float = 4.0                  # length of the continuous test sweep along x (cm)                        --test-path-cm
    eval_x_min_cm: float | None = None         # lower bound of PV evaluation window on x axis (cm); None = full sweep   --eval-x-min-cm
    eval_x_max_cm: float | None = None         # upper bound of PV evaluation window (cm); must be set with eval_x_min_cm --eval-x-max-cm

    seed: int = 123  # random seed for connectivity, weight init, and training sequence  --seed-start

    # --- LIF neuron constants (apply to MON and TS) ---
    vth_mV: float = -54.0    # spike threshold (mV)              --vth-mv
    vreset_mV: float = -60.0  # reset potential after spike (mV)  --vreset-mv
    el_mV: float = -74.0     # leak / resting potential (mV)     --el-mv
    tau_ref_ms: float = 2.0  # absolute refractory period (ms)   --tau-ref-ms
    tau_m_ms: float = 10.0   # membrane time constant (ms)       --tau-m-ms
    tau_s_ms: float = 2.0    # synaptic PSP decay time constant (ms)  --tau-s-ms

    # --- LL→MON connectivity ---
    # Mixed random + weakly topographic. topography_strength=0 → fully random; =1 → all Gaussian.
    p_ll_to_mon: float = 0.03                     # connection probability when topography_strength == 0          --p-ll-to-mon
    ll_to_mon_in_degree: int = 3                  # LL inputs per MON neuron when topography_strength > 0         --ll-mon-in-degree
    ll_to_mon_sigma: float = 15.0                 # spread of topographic Gaussian in LL-index units              --ll-mon-sigma
    ll_to_mon_topography_strength: float = 0.01   # fraction of connections that follow somatotopy (0=random)     --ll-mon-topo
    ll_mon_w_mean_mV: float = 7.0                 # mean LL→MON fixed weight (mV)                                 --ll-mon-w-mean-mv
    ll_mon_w_jitter_mV: float = 3.0               # uniform jitter on fixed LL→MON weights (mV)                   --ll-mon-w-jitter-mv

    # --- LL→MON STDP (optional plastic expansion) ---
    ll_mon_use_stdp: bool = False             # enable multiplicative STDP on LL→MON synapses            --use-ll-mon-stdp
    ll_mon_apre: float = 0.01                # pre-synaptic trace increment (LTP amplitude)              --ll-mon-apre
    ll_mon_apost: float = -0.0105            # post-synaptic trace increment (LTD amplitude, negative)   --ll-mon-apost
    ll_mon_wmax_mV: float = 20.0             # maximum LL→MON weight when STDP is enabled (mV)           --ll-mon-wmax-mv
    ll_mon_w_init_mV: float = 10.0           # initial mean LL→MON weight when STDP is enabled (mV)      --ll-mon-w-init-mv
    ll_mon_w_jitter_stdp_mV: float = 2.0     # initial weight jitter when STDP is enabled (mV)           --ll-mon-w-jitter-stdp-mv
    ll_mon_homeo_eta: float = 0.0            # homeostatic scaling rate for incoming LL→MON weights       --ll-mon-homeo-eta
    ll_mon_homeo_every_trials: int = 10      # apply LL→MON homeostasis every N trials                   --ll-mon-homeo-every-trials

    # --- MON→TS connectivity ---
    # Mixed random + weakly topographic. topography_strength=0 → fully random; =1 → all Gaussian.
    mon_to_ts_out_degree: int = 16                  # TS targets per MON neuron                                     --mon-ts-out-degree
    mon_to_ts_sigma: float = 120.0                  # spread of topographic Gaussian in TS-index units              --mon-ts-sigma
    mon_to_ts_topography_strength: float = 0.05     # fraction of MON→TS connections that follow somatotopy         --mon-ts-topo

    # --- MON→TS STDP ---
    mon_ts_apre: float = 0.02        # pre-synaptic trace increment (LTP amplitude)            --mon-ts-apre
    mon_ts_apost: float = -0.021     # post-synaptic trace increment (LTD amplitude, negative)  --mon-ts-apost
    mon_ts_wmax: float = 0.045       # maximum dimensionless MON→TS weight                      --mon-ts-wmax
    mon_ts_w_init: float = 0.020     # initial mean MON→TS weight                               --mon-ts-w-init
    mon_ts_w_jitter: float = 0.010   # uniform jitter on initial MON→TS weights                 --mon-ts-w-jitter
    mon_ts_homeo_eta: float = 0.0    # homeostatic scaling rate for incoming MON→TS weights      --mon-ts-homeo-eta
    mon_ts_homeo_every_trials: int = 10  # apply MON→TS homeostasis every N trials              --mon-ts-homeo-every-trials
    mon_ts_gain_mV: float = 30.0     # EPSP size per unit weight: ge_post += mon_ts_gain * w * mV  --mon-ts-gain-mv

    keep_mon_ts_stdp_during_test: bool = False  # if True, STDP stays active during the test phase (diagnostic)   --keep-mon-ts-stdp-during-test
    test_using_held_snapshots: bool = False      # if True, test uses snapshot positions instead of continuous sweep  --test-using-held-snapshots

    # --- Inhibition and background noise ---
    mon_to_global_inh_p: float = 0.06             # MON→inh-cell connection probability                         --mon-global-inh-p
    mon_to_global_inh_drive_mV: float = 0.5       # EPSP increment in MON inh-cell per spike (mV)              --mon-global-inh-drive-mv
    global_inh_to_mon_mV: float = 1.4             # IPSP delivered by MON inh-cell to each MON neuron (mV)     --mon-global-inh-mv

    ts_lateral_radius: int = 14                    # TS lateral inhibition radius in neuron-index units (toroidal ring)  --ts-lateral-radius
    ts_local_inh_peak_mV: float = 1.0             # peak TS lateral inhibition weight at distance 0 (mV)               --ts-local-inh-peak-mv

    use_ts_feedback_inh: bool = False              # enable TS global feedback inhibition via dedicated inh cell   --use-ts-feedback-inh
    ts_to_global_inh_p: float = 0.12              # TS→inh-cell connection probability                            --ts-feedback-p
    ts_to_global_inh_drive_mV: float = 0.35       # EPSP increment in TS inh-cell per spike (mV)                  --ts-feedback-drive-mv
    global_inh_to_ts_mV: float = 0.8              # IPSP delivered by TS inh-cell to each TS neuron (mV)          --ts-feedback-inh-mv

    bg_rate_mon_hz: float = 22.0    # MON background Poisson input rate (Hz)  --bg-rate-mon-hz
    bg_rate_ts_hz: float = 12.0     # TS background Poisson input rate (Hz)   --bg-rate-ts-hz
    bg_w_mon_mV: float = 1.5        # EPSP weight of MON background synapse (mV)  --bg-w-mon-mv
    bg_w_ts_mV: float = 0.60        # EPSP weight of TS background synapse (mV)   --bg-w-ts-mv


def apply_model_mode(params: NetworkParams, mode: str) -> NetworkParams:
    """
    Apply named parameter presets.

    Modes:
      - ll_thesis: strict lateral-line setting from thesis context.
      - ll_fast: reduced sizes for quick checks.
    """
    mode = mode.strip().lower()

    if mode == "ll_thesis":
        return replace(
            params,
            # Lateral-line architecture from thesis context.
            n_ll=100,
            n_mon=3200,
            n_ts=300,
            # Long training with fixed network dynamics.
            n_training_trials=1000,
            trial_duration_s=1.2,
            checkpoint_trials=5,
            mon_ts_apre=0.01,
            mon_ts_apost=-0.006,
            mon_ts_wmax=0.028,
            mon_ts_w_init=0.020,
            mon_ts_w_jitter=0.005,
            ll_mon_w_mean_mV=11.5,
            ll_to_mon_in_degree=10,
            ll_to_mon_topography_strength=0.08,
            mon_to_ts_topography_strength=0.08,
            mon_to_ts_out_degree=16,
            mon_to_ts_sigma=10.0,
            mon_ts_gain_mV=72.0,
            mon_to_global_inh_p=0.03,
            global_inh_to_mon_mV=1.15,
            ts_lateral_radius=18,
            ts_local_inh_peak_mV=0.9,
            use_ts_feedback_inh=True,
            ts_to_global_inh_p=0.08,
            ts_to_global_inh_drive_mV=0.25,
            global_inh_to_ts_mV=0.2,
            bg_rate_mon_hz=22.0,
            bg_rate_ts_hz=20.0,
            bg_w_ts_mV=0.85,
            # Start with no training noise to verify map mechanism.
            training_noise_scale_early=0.0,
            training_noise_scale_late=0.0,
            training_bidirectional=False,
            # Train in a near-field distance band for better SNR.
            training_fixed_distance=False,
            training_distance_min_cm=0.8,
            training_distance_max_cm=0.8,
            # Thesis-consistent stimulus speed.
            speed_cm_s=5.0,
            test_path_cm=5.0,
            distance_cm=0.8,
        )

    if mode == "ll_fast":
        return replace(
            params,
            n_mon=800,
            n_ts=80,
            n_training_trials=40,
            trial_duration_s=0.8,
            checkpoint_trials=10,
        )

    raise ValueError(f"Unknown mode '{mode}'. Use 'll_thesis' or 'll_fast'.")
