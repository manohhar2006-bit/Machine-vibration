"""
Root entrypoint wrapper for serving/serial_reader.py.
Allows running 'python serial_reader.py' directly from the repository root.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.join(BASE_DIR, "serving")
sys.path.insert(0, SERVING_DIR)

import serial_reader

if __name__ == "__main__":
    serial_reader.main()
