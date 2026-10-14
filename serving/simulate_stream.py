"""
Streams cached MaFaulDa vibration windows to the Flask API's /ingest
endpoint, standing in for the Arduino + serial_reader.py path while the
physical rig is not yet assembled. Uses real accelerometer data, not
synthetic, so the demo reflects the actual trained pipeline end to end.

Sends normal windows first (for calibration), then normal windows the
calibrator has not seen, then imbalance windows -- the dashboard's
classification is expected to stay NORMAL through the first two groups
and flip to FAULTY on the third.

Requires the cache built by build_demo_cache.py (run once beforehand).

Usage: python simulate_stream.py [--host http://127.0.0.1:5000] [--interval 1.0]
"""
import argparse
import os
import time

import numpy as np
import requests

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "demo_windows.npz")


def send(host, window, sample_rate):
    payload = {"window": window.tolist(), "sample_rate": float(sample_rate)}
    try:
        res = requests.post(f"{host}/ingest", json=payload, timeout=5)
        return res.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def main(host, interval_s):
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"{CACHE_PATH} not found. Run build_demo_cache.py once before using this script."
        )

    cache = np.load(CACHE_PATH)
    calibration_windows = cache["calibration_windows"]
    normal_eval_windows = cache["normal_eval_windows"]
    imbalance_eval_windows = cache["imbalance_eval_windows"]
    sample_rate = cache["sample_rate"]

    print("starting calibration")
    requests.post(f"{host}/calibrate/start", timeout=5)
    for window in calibration_windows:
        send(host, window, sample_rate)
        time.sleep(interval_s)
    result = requests.post(f"{host}/calibrate/stop", timeout=5).json()
    print(f"calibration: {result}")

    sequence = list(normal_eval_windows) + list(imbalance_eval_windows)
    for i, window in enumerate(sequence):
        result = send(host, window, sample_rate)
        print(f"[{i}] prediction={result.get('prediction')} probability={result.get('probability')}")
        time.sleep(interval_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:5000")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    main(args.host, args.interval)