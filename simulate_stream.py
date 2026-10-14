"""
Root entrypoint wrapper for serving/simulate_stream.py.
Allows running 'python simulate_stream.py' directly from the repository root.
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.join(BASE_DIR, "serving")
sys.path.insert(0, SERVING_DIR)

import simulate_stream

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate vibration stream into Flask backend.")
    parser.add_argument("--host", default="http://127.0.0.1:5000", help="Flask server URL")
    parser.add_argument("--interval", type=float, default=0.5, help="Interval between windows in seconds")
    args = parser.parse_args()
    simulate_stream.main(args.host, args.interval)
