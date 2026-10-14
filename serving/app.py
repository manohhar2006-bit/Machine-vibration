"""
Flask API serving the trained classifier and the dashboard.

Runtime state (latest window, features, classification, calibration
status) is held in a single in-memory dict, since this serves one
physical rig at a time.

Endpoints:
  GET  /                 dashboard page
  GET  /health            liveness check, used by Render
  POST /ingest             receive a vibration window, classify it,
                           update latest state
  POST /calibrate/start     begin accumulating incoming windows as the
                           calibration set for a new deployment
  POST /calibrate/stop       fit and save a BaselineCalibrator on the
                           accumulated windows
  GET  /latest             current state, polled by the dashboard
"""
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.fft import rfft, rfftfreq

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from calibration import BaselineCalibrator
from features import DEFAULT_MAX_FREQ, extract_features, features_to_vector
from model import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_store", "classifier.joblib")
BASELINE_PATH = os.path.join(BASE_DIR, "model_store", "baseline.json")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "dashboard", "templates"),
    static_folder=os.path.join(BASE_DIR, "..", "dashboard", "static"),
)

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
calibrator = BaselineCalibrator.load(BASELINE_PATH) if os.path.exists(BASELINE_PATH) else BaselineCalibrator()

state = {
    "latest_window": [],
    "latest_fft": [],
    "sample_rate": None,
    "rpm": None,
    "features": None,
    "prediction": None,
    "probability": None,
    "calibrated": calibrator.is_fitted,
    "calibrating": False,
    "calibration_buffer": [],
    "last_update": None,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/ingest", methods=["POST"])
def ingest():
    payload = request.get_json(force=True)
    window = payload.get("window")
    sample_rate = payload.get("sample_rate", 4000)
    rpm = payload.get("rpm")

    if not window:
        return jsonify({"error": "missing 'window'"}), 400

    feat = extract_features(window, sample_rate)
    vector = features_to_vector(feat)

    if state["calibrating"]:
        state["calibration_buffer"].append(vector.tolist())

    prediction = None
    probability = None
    if calibrator.is_fitted and model is not None:
        normalized = calibrator.transform(vector.reshape(1, -1))
        pred = model.predict(normalized)[0]
        proba = model.predict_proba(normalized)[0][1]
        prediction = "imbalance" if pred == 1 else "normal"
        probability = float(proba)

    w = np.asarray(window, dtype=float)
    spectrum = np.abs(rfft(w - w.mean()))
    freqs = rfftfreq(len(w), d=1.0 / sample_rate)
    band_mask = freqs <= DEFAULT_MAX_FREQ

    state.update({
        "latest_window": w[-500:].tolist(),
        "latest_fft": {"freqs": freqs[band_mask].tolist(), "magnitude": spectrum[band_mask].tolist()},
        "sample_rate": sample_rate,
        "rpm": rpm,
        "features": feat,
        "prediction": prediction,
        "probability": probability,
        "last_update": time.time(),
    })

    return jsonify({"prediction": prediction, "probability": probability, "features": feat})


@app.route("/calibrate/start", methods=["POST"])
def calibrate_start():
    state["calibrating"] = True
    state["calibration_buffer"] = []
    return jsonify({"status": "calibrating"})


@app.route("/calibrate/stop", methods=["POST"])
def calibrate_stop():
    global calibrator

    buffer = np.array(state["calibration_buffer"])
    if len(buffer) < 3:
        return jsonify({"error": "not enough windows collected during calibration"}), 400

    calibrator = BaselineCalibrator().fit(buffer)
    calibrator.save(BASELINE_PATH)

    state["calibrating"] = False
    state["calibrated"] = True
    return jsonify({"status": "calibrated", "n_windows": len(buffer)})


@app.route("/latest")
def latest():
    return jsonify(state)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)