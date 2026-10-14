"""
Root entrypoint wrapper for src/train.py.
Allows running 'python train.py' directly from the repository root.
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vibration classifier.")
    parser.add_argument("--synthetic", action="store_true", help="Force use of synthetic vibration dataset.")
    args = parser.parse_args()
    train.train_pipeline(use_synthetic=args.synthetic)
