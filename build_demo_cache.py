"""
Root entrypoint wrapper for serving/build_demo_cache.py.
Allows running 'python build_demo_cache.py' directly from the repository root.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.join(BASE_DIR, "serving")
sys.path.insert(0, SERVING_DIR)

import build_demo_cache

if __name__ == "__main__":
    build_demo_cache.main()
