"""
Flask API serving the predictive vibration classifier and web dashboard.

Endpoints:
  GET  /                  Main dashboard page
  GET  /health            Liveness and model status check
  POST /ingest            Ingests one vibration window, extracts features, performs inference
  POST /calibrate/start   Initiates healthy baseline collection
  POST /calibrate/stop    Fits and persists BaselineCalibrator on collected windows
  GET  /latest            Provides real-time state, charts, and connection diagnostics
  POST /motor/control     Receives motor commands (START, STOP, SPEED) from UI
"""
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

# Import core signal processing and ML modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from calibration import BaselineCalibrator
from features import (
    DEFAULT_MAX_FREQ,
    extract_features,
    features_to_vector,
    validate_window,
)
from model import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_store", "classifier.joblib")
BASELINE_PATH = os.path.join(BASE_DIR, "model_store", "baseline.json")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "dashboard", "templates"),
    static_folder=os.path.join(BASE_DIR, "..", "dashboard", "static"),
)
# Limit max payload to 2MB to prevent memory exhaustion
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

calibrator = None
if os.path.exists(BASELINE_PATH):
    try:
        calibrator = BaselineCalibrator.load(BASELINE_PATH)
    except Exception as e:
        print(f"Warning: Failed to load baseline ({e}), initializing uncalibrated.")
        calibrator = BaselineCalibrator()
else:
    calibrator = BaselineCalibrator()

state = {
    "latest_window": [],
    "latest_fft": {"freqs": [], "magnitude": []},
    "sample_rate": 800.0,
    "rpm": None,
    "features": None,
    "prediction": None,
    "probability": None,
    "calibrated": calibrator.is_fitted if calibrator else False,
    "calibrating": False,
    "calibration_buffer": [],
    "last_update": None,
    "error_message": None,
    "motor_state": "STOPPED",
    "motor_speed": 0,
    "pending_motor_command": None,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "calibrated": calibrator.is_fitted if calibrator else False,
        "timestamp": time.time(),
    })


@app.route("/ingest", methods=["POST"])
def ingest():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    raw_window = payload.get("window")
    if raw_window is None:
        return jsonify({"error": "Missing 'window' field in payload"}), 400

    # Validate window
    try:
        window_arr = validate_window(raw_window)
    except ValueError as ve:
        return jsonify({"error": f"Invalid vibration window: {ve}"}), 400

    # Validate sample rate
    try:
        sample_rate = float(payload.get("sample_rate", 800.0))
        if sample_rate <= 0 or not np.isfinite(sample_rate):
            return jsonify({"error": "Sample rate must be a finite positive number"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Sample rate must be numeric"}), 400

    # Validate optional RPM
    rpm = payload.get("rpm")
    if rpm is not None:
        try:
            rpm = float(rpm)
            if not np.isfinite(rpm) or rpm < 0 or rpm > 50000:
                rpm = None
        except (ValueError, TypeError):
            rpm = None

    # Feature extraction (single-pass FFT, no redundant calculations)
    try:
        feat = extract_features(
            window_arr, sample_rate, max_freq=DEFAULT_MAX_FREQ, return_spectrum=True
        )
        vector = features_to_vector(feat)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {e}"}), 400

    # If calibrating, accumulate healthy feature vector
    if state["calibrating"]:
        state["calibration_buffer"].append(vector.tolist())

    # Classification
    prediction = None
    probability = None
    if calibrator is not None and calibrator.is_fitted and model is not None:
        try:
            # Baseline standardization using feature scale floors
            normalized = calibrator.transform(vector.reshape(1, -1))
            if np.all(np.isfinite(normalized)):
                pred = model.predict(normalized)[0]
                proba = model.predict_proba(normalized)[0][1]
                prediction = "imbalance" if pred == 1 else "normal"
                probability = float(proba)
            else:
                state["error_message"] = "Standardization produced non-finite values."
        except Exception as e:
            state["error_message"] = f"Inference failed: {e}"

    # Extract display arrays (downsample waveform to at most 500 points for browser performance)
    display_len = min(len(window_arr), 500)
    display_window = window_arr[-display_len:].round(4).tolist()

    fft_freqs = feat.get("freqs", np.array([])).round(1).tolist()
    fft_mag = feat.get("magnitude", np.array([])).round(4).tolist()

    clean_features = {
        "rms": round(feat["rms"], 4),
        "dominant_freq": round(feat["dominant_freq"], 2),
        "spectral_energy": round(feat["spectral_energy"], 2),
        "kurtosis": round(feat["kurtosis"], 4),
    }

    state.update({
        "latest_window": display_window,
        "latest_fft": {"freqs": fft_freqs, "magnitude": fft_mag},
        "sample_rate": sample_rate,
        "rpm": round(rpm, 1) if rpm is not None else None,
        "features": clean_features,
        "prediction": prediction,
        "probability": probability,
        "last_update": time.time(),
        "error_message": None,
    })

    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "features": clean_features,
        "rpm": state["rpm"],
    })


@app.route("/calibrate/start", methods=["POST"])
def calibrate_start():
    state["calibrating"] = True
    state["calibration_buffer"] = []
    return jsonify({"status": "calibrating", "collected": 0})


@app.route("/calibrate/stop", methods=["POST"])
def calibrate_stop():
    global calibrator

    buffer = np.array(state["calibration_buffer"], dtype=float)
    if len(buffer) < 3:
        state["calibrating"] = False
        return jsonify({
            "error": f"Insufficient calibration windows ({len(buffer)}). Minimum 3 required."
        }), 400

    try:
        new_calibrator = BaselineCalibrator(min_windows=3).fit(buffer)
        new_calibrator.save(BASELINE_PATH)
        calibrator = new_calibrator

        state["calibrating"] = False
        state["calibrated"] = True
        return jsonify({
            "status": "calibrated",
            "n_windows": len(buffer),
            "baseline_mean": calibrator.mean_.round(3).tolist(),
            "effective_scale": calibrator.effective_scale_.round(3).tolist(),
        })
    except Exception as e:
        state["calibrating"] = False
        return jsonify({"error": f"Calibration failed: {e}"}), 400


@app.route("/motor/control", methods=["POST"])
def motor_control():
    """Receives motor control commands from UI or external integration."""
    payload = request.get_json(silent=True) or {}
    action = payload.get("action", "").upper()

    if action == "START":
        speed = int(payload.get("speed", 180))
        speed = max(0, min(255, speed))
        state["motor_state"] = "RUNNING"
        state["motor_speed"] = speed
        state["pending_motor_command"] = f"MOTOR START {speed}"
        return jsonify({"status": "ok", "motor_state": "RUNNING", "speed": speed})

    elif action == "STOP":
        state["motor_state"] = "STOPPED"
        state["motor_speed"] = 0
        state["pending_motor_command"] = "MOTOR STOP"
        return jsonify({"status": "ok", "motor_state": "STOPPED", "speed": 0})

    elif action == "SPEED":
        speed = int(payload.get("speed", state["motor_speed"]))
        speed = max(0, min(255, speed))
        state["motor_speed"] = speed
        state["pending_motor_command"] = f"MOTOR SPEED {speed}"
        return jsonify({"status": "ok", "speed": speed})

    return jsonify({"error": f"Unknown motor action: '{action}'"}), 400


@app.route("/motor/pending_command", methods=["GET"])
def get_pending_motor_command():
    """Polled by serial_reader.py to forward commands to Arduino hardware."""
    cmd = state.get("pending_motor_command")
    state["pending_motor_command"] = None
    return jsonify({"command": cmd})


@app.route("/latest")
def latest():
    now = time.time()
    last_up = state.get("last_update")
    is_stale = (last_up is None) or ((now - last_up) > 3.0)

    response_data = dict(state)
    response_data["is_stale"] = is_stale
    response_data["calibrated"] = calibrator.is_fitted if calibrator else False
    response_data["server_time"] = now

    return jsonify(response_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Bind to 127.0.0.1 for local security, debug=False to avoid arbitrary code execution
    app.run(host="127.0.0.1", port=port, debug=False)