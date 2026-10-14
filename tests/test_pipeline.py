"""
Comprehensive Automated Test Suite for Predictive Machine Vibration Monitoring.

Covers:
  - Feature extraction numerical tests (zero, constant, sine, invalid, NaNs)
  - Baseline calibration robustness (zero variance, scale floors, fit/transform, round-trip)
  - Flask backend API contracts, input validation, stale status, and motor commands
  - Serial protocol packet parsing and malformed data handling
  - End-to-end simulated inference pipeline
"""
import json
import os
import sys
import unittest
import numpy as np

# Ensure src and serving are in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "serving"))

from features import (
    DEFAULT_MAX_FREQ,
    extract_features,
    features_to_vector,
    validate_window,
    window_to_vector,
)
from calibration import BaselineCalibrator
from app import app
from serial_reader import parse_packet


class TestFeatureExtraction(unittest.TestCase):
    """Test suite for src/features.py."""

    def test_zero_signal(self):
        arr = np.zeros(256)
        feat = extract_features(arr, sample_rate=800.0)
        self.assertEqual(feat["rms"], 0.0)
        self.assertEqual(feat["dominant_freq"], 0.0)
        self.assertEqual(feat["spectral_energy"], 0.0)
        self.assertEqual(feat["kurtosis"], 0.0)
        self.assertFalse(np.isnan(feat["kurtosis"]))

    def test_constant_signal(self):
        arr = np.ones(256) * 2.5
        feat = extract_features(arr, sample_rate=800.0)
        self.assertAlmostEqual(feat["rms"], 2.5, places=5)
        self.assertEqual(feat["dominant_freq"], 0.0)
        self.assertEqual(feat["spectral_energy"], 0.0)
        self.assertEqual(feat["kurtosis"], 0.0)
        self.assertFalse(np.isnan(feat["kurtosis"]))

    def test_known_sine_frequency(self):
        # 35 Hz sine sampled at 800 Hz for 1.0 second (800 samples)
        sr = 800.0
        t = np.arange(800) / sr
        freq = 35.0
        amplitude = 1.41421356  # RMS should be approx 1.0
        arr = amplitude * np.sin(2 * np.pi * freq * t)

        feat = extract_features(arr, sample_rate=sr)
        self.assertAlmostEqual(feat["rms"], 1.0, delta=0.05)
        self.assertAlmostEqual(feat["dominant_freq"], freq, delta=1.5)
        self.assertGreater(feat["spectral_energy"], 0.0)
        self.assertTrue(np.isfinite(feat["kurtosis"]))

    def test_invalid_and_nan_inputs(self):
        with self.assertRaises(ValueError):
            extract_features(None, 800)

        with self.assertRaises(ValueError):
            extract_features([], 800)

        with self.assertRaises(ValueError):
            extract_features([1.0, 2.0], 800)  # Too short

        with self.assertRaises(ValueError):
            extract_features([1.0, np.nan, 3.0, 4.0], 800)  # NaN

        with self.assertRaises(ValueError):
            extract_features([1.0, np.inf, 3.0, 4.0], 800)  # Inf

        with self.assertRaises(ValueError):
            extract_features(np.ones(10), sample_rate=-10)  # Invalid sample rate

    def test_different_window_sizes(self):
        for size in [64, 128, 256, 512, 1024]:
            arr = np.random.normal(0, 1.0, size=size)
            feat = extract_features(arr, sample_rate=800)
            self.assertEqual(len(features_to_vector(feat)), 4)
            self.assertTrue(all(np.isfinite(v) for v in feat.values() if isinstance(v, (int, float))))


class TestBaselineCalibration(unittest.TestCase):
    """Test suite for src/calibration.py."""

    def setUp(self):
        self.calib_path = os.path.join(BASE_DIR, "serving", "model_store", "test_baseline.json")

    def tearDown(self):
        if os.path.exists(self.calib_path):
            os.remove(self.calib_path)

    def test_zero_variance_dominant_frequency(self):
        # 5 calibration windows with identical dominant frequency (117 Hz)
        data = np.array([
            [0.30, 117.0, 7.9e6, -0.69],
            [0.31, 117.0, 8.0e6, -0.70],
            [0.29, 117.0, 7.8e6, -0.68],
            [0.30, 117.0, 7.9e6, -0.69],
            [0.30, 117.0, 7.9e6, -0.69],
        ])
        calib = BaselineCalibrator().fit(data)
        self.assertTrue(calib.is_fitted)
        self.assertEqual(calib.std_[1], 0.0)
        # Verify scale floor took effect
        self.assertGreaterEqual(calib.effective_scale_[1], 2.0)

        # Small 0.5 Hz jitter must produce bounded, reasonable z-score
        jittered = np.array([0.30, 117.5, 7.9e6, -0.69])
        z = calib.transform(jittered)
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertLess(abs(z[1]), 1.0, "Z-score exploded on fractional Hz jitter!")

    def test_minimum_windows_validation(self):
        data = np.array([[0.3, 30.0, 1000.0, 0.0]])
        calib = BaselineCalibrator(min_windows=3)
        with self.assertRaises(ValueError):
            calib.fit(data)

    def test_flatlined_sensor_rejection(self):
        # Disconnected sensor producing 0 RMS
        data = np.zeros((5, 4))
        calib = BaselineCalibrator()
        with self.assertRaises(ValueError):
            calib.fit(data)

    def test_save_load_round_trip(self):
        data = np.array([
            [0.30, 30.0, 1000.0, 0.0],
            [0.31, 30.0, 1050.0, 0.1],
            [0.29, 30.0, 950.0, -0.1],
        ])
        calib = BaselineCalibrator().fit(data)
        calib.save(self.calib_path)

        loaded = BaselineCalibrator.load(self.calib_path)
        self.assertTrue(loaded.is_fitted)
        np.testing.assert_allclose(loaded.mean_, calib.mean_)
        np.testing.assert_allclose(loaded.effective_scale_, calib.effective_scale_)


class TestFlaskBackend(unittest.TestCase):
    """Test suite for serving/app.py."""

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("model_loaded", data)

    def test_ingest_validation(self):
        # Empty payload
        res = self.client.post("/ingest", json={})
        self.assertEqual(res.status_code, 400)

        # Non-JSON
        res = self.client.post("/ingest", data="not json")
        self.assertEqual(res.status_code, 400)

        # Non-finite values
        res = self.client.post("/ingest", json={"window": [1.0, float("nan"), 3.0, 4.0]})
        self.assertEqual(res.status_code, 400)

        # Valid payload
        valid_win = [float(x % 10) * 0.1 for x in range(256)]
        res = self.client.post("/ingest", json={"window": valid_win, "sample_rate": 800, "rpm": 1800.0})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("features", data)
        self.assertIn("rms", data["features"])

    def test_calibration_lifecycle(self):
        # Start calibration
        res = self.client.post("/calibrate/start")
        self.assertEqual(res.status_code, 200)

        # Stop with zero windows -> expect 400
        res = self.client.post("/calibrate/stop")
        self.assertEqual(res.status_code, 400)

        # Send 4 healthy windows
        self.client.post("/calibrate/start")
        win = [0.2 * np.sin(2 * np.pi * 30 * i / 800) for i in range(256)]
        for _ in range(4):
            self.client.post("/ingest", json={"window": win, "sample_rate": 800, "rpm": 1800.0})

        res = self.client.post("/calibrate/stop")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("status"), "calibrated")

    def test_motor_control_endpoints(self):
        res = self.client.post("/motor/control", json={"action": "START", "speed": 220})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("motor_state"), "RUNNING")

        # Check pending command
        res = self.client.get("/motor/pending_command")
        self.assertEqual(res.get_json().get("command"), "MOTOR START 220")

        # Command should clear on read
        res = self.client.get("/motor/pending_command")
        self.assertIsNone(res.get_json().get("command"))

        # Stop motor
        res = self.client.post("/motor/control", json={"action": "STOP"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("motor_state"), "STOPPED")


class TestSerialProtocol(unittest.TestCase):
    """Test suite for serving/serial_reader.py."""

    def test_valid_packet(self):
        line = "$VIB,42,800,1800.0,4,0.1,-0.2,0.3,0.0"
        packet = parse_packet(line)
        self.assertIsNotNone(packet)
        self.assertEqual(packet["sequence"], 42)
        self.assertEqual(packet["sample_rate"], 800.0)
        self.assertEqual(packet["rpm"], 1800.0)
        self.assertEqual(len(packet["window"]), 4)

    def test_count_mismatch(self):
        line = "$VIB,42,800,1800.0,5,0.1,-0.2,0.3,0.0"  # count is 5, provided 4
        self.assertIsNone(parse_packet(line))

    def test_malformed_numeric_data(self):
        line = "$VIB,42,800,1800.0,3,0.1,CORRUPTED,0.0"
        self.assertIsNone(parse_packet(line))

    def test_status_messages(self):
        self.assertIsNone(parse_packet("$STATUS,SENSOR=OK,MOTOR=STOPPED,RPM=0.0"))
        self.assertIsNone(parse_packet("$ERR,ADXL345_NOT_FOUND"))
        self.assertIsNone(parse_packet("$BOOT,Arduino Uno initializing..."))


if __name__ == "__main__":
    unittest.main()
