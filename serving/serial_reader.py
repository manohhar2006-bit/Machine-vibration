"""
Reads buffered vibration windows from the Arduino over USB serial and
forwards each one to the Flask API's /ingest endpoint. Each serial
line is expected in the format produced by firmware/vibration_monitor.ino:

    <rpm>,<sample_1>,<sample_2>,...,<sample_N>

Usage: python serial_reader.py --port /dev/ttyUSB0 [--host http://127.0.0.1:5000]
"""
import argparse

import requests
import serial

SAMPLE_RATE = 4000  # Hz, must match SAMPLE_INTERVAL_US in the firmware


def main(port, baud, host):
    ser = serial.Serial(port, baud, timeout=5)
    print(f"reading from {port} at {baud} baud, forwarding to {host}/ingest")

    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        try:
            values = [float(v) for v in line.split(",")]
        except ValueError:
            print(f"skipping malformed line: {line[:50]}")
            continue

        rpm, window = values[0], values[1:]
        if len(window) < 10:
            continue

        payload = {"window": window, "sample_rate": SAMPLE_RATE, "rpm": rpm}
        try:
            res = requests.post(f"{host}/ingest", json=payload, timeout=5)
            result = res.json()
            print(f"rpm={rpm:.1f} prediction={result.get('prediction')}")
        except requests.RequestException as e:
            print(f"failed to reach {host}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    main(args.port, args.baud, args.host)