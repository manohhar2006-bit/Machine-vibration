"""
Root entrypoint wrapper for serving/app.py.
Allows running 'python app.py' directly from the repository root.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.join(BASE_DIR, "serving")
sys.path.insert(0, SERVING_DIR)

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
