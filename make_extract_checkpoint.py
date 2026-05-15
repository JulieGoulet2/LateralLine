"""
Create a minimal resume checkpoint from a saved latest_seed_NNN.npz artifact,
so we can run extract-mode (test-only) without re-training.

Usage:
    python make_extract_checkpoint.py <seed> <source_run_name> <dest_run_name>

Example:
    python make_extract_checkpoint.py 127 llmon_topo020_seeds127_132 llmon_topo020_extract_seed127
    python make_extract_checkpoint.py 128 llmon_topo020_seeds127_132 llmon_topo020_extract_seed128
"""

import sys
from pathlib import Path
import numpy as np

seed = int(sys.argv[1])
src_run = sys.argv[2]
dst_run = sys.argv[3]

root = Path(__file__).parent
src_weights = root / "Runs" / src_run / "artifacts" / f"latest_seed_{seed}.npz"
dst_artifacts = root / "Runs" / dst_run / "artifacts"
dst_artifacts.mkdir(parents=True, exist_ok=True)

w = np.load(src_weights, allow_pickle=False)

# savez_compressed (vs savez): ~3x smaller on disk and reduces I/O time;
# load is still mmap'd lazily by np.load downstream.
np.savez_compressed(
    dst_artifacts / "mid_checkpoint.npz",
    trial_idx=np.array(9999, dtype=int),
    mon_ts_w=w["mon_ts_w"],
    ll_mon_w_mV=w["ll_mon_w_mV"],
)
print(f"Checkpoint written to {dst_artifacts / 'mid_checkpoint.npz'}")
print(f"  mon_ts_w shape: {w['mon_ts_w'].shape}, ll_mon_w_mV shape: {w['ll_mon_w_mV'].shape}")
