"""
Builds a small cached set of demo windows from MaFaulDa, so the live
demo (simulate_stream.py) does not reload and re-parse all 382 source
CSV files on every run.

Run once:
    python build_demo_cache.py

Re-run only if the demo window selection needs to change.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load_mafaulda_by_deployment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "demo_windows.npz")


def main():
    print("loading full MaFaulDa dataset (slow, one-time)...")
    windows, labels, groups, sample_rate = load_mafaulda_by_deployment(DATA_ROOT, window_s=1.0)

    normal_windows = windows[labels == 0]
    imbalance_windows = windows[labels == 1]

    calibration_windows = normal_windows[:10]
    normal_eval_windows = normal_windows[10:20]
    imbalance_eval_windows = imbalance_windows[:20]

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(
        CACHE_PATH,
        calibration_windows=calibration_windows,
        normal_eval_windows=normal_eval_windows,
        imbalance_eval_windows=imbalance_eval_windows,
        sample_rate=sample_rate,
    )
    print(f"saved demo cache -> {CACHE_PATH}")


if __name__ == "__main__":
    main()