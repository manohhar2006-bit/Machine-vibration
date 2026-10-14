"""
Streams cached vibration windows to the Flask API's /ingest endpoint,
standing in for the Arduino + serial_reader.py path while the physical rig
is not connected.

Procedure:
1. Verifies connectivity to the Flask API.
2. Initiates calibration mode (/calibrate/start).
3. Streams healthy normal windows to fit per-deployment baseline.
4. Finalizes calibration (/calibrate/stop).
5. Streams unseen normal windows (verdict stays NORMAL).
6. Streams imbalance fault windows (verdict flips to FAULTY -- imbalance).

Usage:
    python simulate_stream.py [--host http://127.0.0.1:5000] [--interval 0.5]
"""
import argparse
import os
import sys
import time

import numpy as np
import requests

CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "demo_windows.npz"))


def send(host, window, sample_rate, rpm=1800.0):
    payload = {
        "window": window.tolist() if hasattr(window, "tolist") else list(window),
        "sample_rate": float(sample_rate),
        "rpm": float(rpm),
    }
    try:
        res = requests.post(f"{host.rstrip('/')}/ingest", json=payload, timeout=5)
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": f"HTTP {res.status_code}: {res.text[:80]}"}
    except requests.RequestException as e:
        return {"error": str(e)}


def main(host, interval_s):
    # Ensure cache exists, auto-generating if necessary
    if not os.path.exists(CACHE_PATH):
        print(f"Demo cache {CACHE_PATH} not found. Automatically generating...")
        build_script = os.path.join(os.path.dirname(__file__), "build_demo_cache.py")
        import subprocess
        subprocess.run([sys.executable, build_script], check=True)

    cache = np.load(CACHE_PATH)
    calibration_windows = cache["calibration_windows"]
    normal_eval_windows = cache["normal_eval_windows"]
    imbalance_eval_windows = cache["imbalance_eval_windows"]
    sample_rate = float(cache["sample_rate"])

    print(f"Connecting to Flask API at {host}...")
    try:
        res = requests.get(f"{host.rstrip('/')}/health", timeout=3)
        if res.status_code == 200:
            health = res.json()
            print(f"Connected successfully. Server health: {health}")
        else:
            print(f"Warning: Server returned HTTP {res.status_code}")
    except requests.RequestException:
        print(f"\n[Error] Could not connect to Flask backend at {host}.")
        print("Please start the backend server first in a separate terminal:")
        print("    python serving/app.py")
        sys.exit(1)

    print("\n--- 1. Starting Baseline Calibration ---")
    try:
        requests.post(f"{host.rstrip('/')}/calibrate/start", timeout=5)
    except requests.RequestException as e:
        print(f"Failed to start calibration: {e}")
        sys.exit(1)

    for i, window in enumerate(calibration_windows):
        send(host, window, sample_rate, rpm=1795.0 + (i % 3))
        print(f"  Calibrating window {i + 1}/{len(calibration_windows)}...")
        time.sleep(interval_s)

    try:
        calib_res = requests.post(f"{host.rstrip('/')}/calibrate/stop", timeout=5).json()
        print(f"Calibration completed: {calib_res}")
    except requests.RequestException as e:
        print(f"Failed to finalize calibration: {e}")
        sys.exit(1)

    print("\n--- 2. Streaming Unseen Normal Windows (Expected: NORMAL) ---")
    for i, window in enumerate(normal_eval_windows):
        result = send(host, window, sample_rate, rpm=1802.0)
        pred = result.get("prediction", "N/A")
        prob = result.get("probability")
        conf = f"{prob * 100:.1f}%" if prob is not None else "--"
        print(f"  [Normal {i + 1:02d}] Prediction: {pred:<10} Fault Confidence: {conf}")
        time.sleep(interval_s)

    print("\n--- 3. Streaming Imbalance Fault Windows (Expected: FAULTY) ---")
    for i, window in enumerate(imbalance_eval_windows):
        result = send(host, window, sample_rate, rpm=1760.0 - (i % 5))
        pred = result.get("prediction", "N/A")
        prob = result.get("probability")
        conf = f"{prob * 100:.1f}%" if prob is not None else "--"
        print(f"  [Fault {i + 1:02d}]  Prediction: {pred:<10} Fault Confidence: {conf}")
        time.sleep(interval_s)

    print("\nSimulation complete. Open http://127.0.0.1:5000 in your browser to view results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate vibration stream into Flask backend.")
    parser.add_argument("--host", default="http://127.0.0.1:5000", help="Flask server URL")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval between windows in seconds")
    args = parser.parse_args()
    main(args.host, args.interval)