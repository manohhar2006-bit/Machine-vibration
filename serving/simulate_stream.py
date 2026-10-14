"""
Streams cached vibration windows to the Flask API's /ingest endpoint,
standing in for the Arduino + serial_reader.py path while the physical rig
is not connected.

Procedure:
1. Puts backend into calibration mode (/calibrate/start).
2. Streams healthy normal windows to fit per-deployment baseline.
3. Finalizes calibration (/calibrate/stop).
4. Streams unseen normal windows (verdict stays NORMAL).
5. Streams imbalance fault windows (verdict flips to FAULTY -- imbalance).

Usage:
    python simulate_stream.py [--host http://127.0.0.1:5000] [--interval 0.5]
"""
import argparse
import os
import time

import numpy as np
import requests

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "demo_windows.npz")


def send(host, window, sample_rate, rpm=1800.0):
    payload = {
        "window": window.tolist(),
        "sample_rate": float(sample_rate),
        "rpm": float(rpm),
    }
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
    sample_rate = float(cache["sample_rate"])

    print(f"Connecting to {host}...")
    try:
        health = requests.get(f"{host}/health", timeout=3).json()
        print(f"Server health: {health}")
    except Exception as e:
        print(f"Warning: Could not contact {host} ({e}). Ensure 'python serving/app.py' is running.")

    print("\n--- 1. Starting Baseline Calibration ---")
    requests.post(f"{host}/calibrate/start", timeout=5)
    for i, window in enumerate(calibration_windows):
        send(host, window, sample_rate, rpm=1795.0 + (i % 3))
        print(f"  Calibrating window {i + 1}/{len(calibration_windows)}...")
        time.sleep(interval_s)

    calib_res = requests.post(f"{host}/calibrate/stop", timeout=5).json()
    print(f"Calibration completed: {calib_res}")

    print("\n--- 2. Streaming Unseen Normal Windows (Expected: NORMAL) ---")
    for i, window in enumerate(normal_eval_windows):
        result = send(host, window, sample_rate, rpm=1802.0)
        pred = result.get("prediction", "N/A")
        prob = result.get("probability")
        conf = f"{prob * 100:.1f}%" if prob is not None else "--"
        print(f"  [Normal {i + 1:02d}] Prediction: {pred:<10} Confidence: {conf}")
        time.sleep(interval_s)

    print("\n--- 3. Streaming Imbalance Fault Windows (Expected: FAULTY) ---")
    for i, window in enumerate(imbalance_eval_windows):
        result = send(host, window, sample_rate, rpm=1760.0 - (i % 5))
        pred = result.get("prediction", "N/A")
        prob = result.get("probability")
        conf = f"{prob * 100:.1f}%" if prob is not None else "--"
        print(f"  [Fault {i + 1:02d}]  Prediction: {pred:<10} Confidence: {conf}")
        time.sleep(interval_s)

    print("\nSimulation complete. Dashboard should show transitions and final status.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate vibration stream into Flask backend.")
    parser.add_argument("--host", default="http://127.0.0.1:5000", help="Flask server URL")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval between windows in seconds")
    args = parser.parse_args()
    main(args.host, args.interval)