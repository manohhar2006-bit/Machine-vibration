"""
Builds a cached set of demo vibration windows (demo_windows.npz) so that
simulate_stream.py can stream realistic vibration data into the Flask API
without needing external downloads or live hardware.

If raw MaFaulDa CSV files exist in data/raw, it extracts windows from them.
Otherwise, it automatically generates realistic physical vibration windows
(normal baseline, normal evaluation, and imbalance fault) using the
hardware-calibrated vibration generator at 800 Hz.
"""
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import (
    MAFAULDA_SAMPLE_RATE,
    SAMPLE_RATE_SYNTHETIC,
    generate_synthetic_dataset,
    load_mafaulda_by_deployment,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "demo_windows.npz")


def main():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    has_raw = os.path.exists(DATA_ROOT) and len(glob.glob(os.path.join(DATA_ROOT, "normal", "*.csv"))) > 0

    if has_raw:
        print("Loading full MaFaulDa dataset from data/raw...")
        windows, labels, groups, sample_rate = load_mafaulda_by_deployment(DATA_ROOT, window_s=1.0)
        normal_windows = windows[labels == 0]
        imbalance_windows = windows[labels == 1]

        calibration_windows = normal_windows[:10]
        normal_eval_windows = normal_windows[10:20]
        imbalance_eval_windows = imbalance_windows[:20]
    else:
        print("Raw MaFaulDa files not detected in data/raw.")
        print("Synthesizing realistic vibration demo windows (800 Hz, 256 samples/window)...")
        sample_rate = SAMPLE_RATE_SYNTHETIC
        # Generate 15 normal calibration windows, 15 normal evaluation windows, 25 imbalance windows
        # Deployment shift 0 (nominal motor speed ~30 Hz / 1800 RPM)
        windows, labels, _ = generate_synthetic_dataset(
            n_windows_per_class=50, window_s=0.32, sample_rate=sample_rate, deployment_shift=0.0, seed=123
        )
        norm = windows[labels == 0]
        imb = windows[labels == 1]

        calibration_windows = norm[:10]
        normal_eval_windows = norm[10:25]
        imbalance_eval_windows = imb[:25]

    np.savez(
        CACHE_PATH,
        calibration_windows=calibration_windows,
        normal_eval_windows=normal_eval_windows,
        imbalance_eval_windows=imbalance_eval_windows,
        sample_rate=float(sample_rate),
    )
    print(f"Successfully generated and saved demo cache -> {CACHE_PATH}")
    print(f"  Calibration windows: {len(calibration_windows)}")
    print(f"  Normal evaluation:   {len(normal_eval_windows)}")
    print(f"  Imbalance fault:     {len(imbalance_eval_windows)}")
    print(f"  Sample rate:         {sample_rate} Hz")


if __name__ == "__main__":
    main()