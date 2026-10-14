"""
Adaptive baseline calibration for predictive vibration monitoring.

Per-deployment calibration records an initial baseline of healthy operation
for the specific motor and mounting foundation currently in service.
Subsequent operational feature vectors are standardized against this baseline:

    z = (x - mean) / effective_std

where effective_std incorporates feature-specific scale floors:
    effective_std = max(std, scale_floor, relative_floor * |mean|)

This mathematically guarantees that zero-variance features (such as dominant
frequency under steady-state motor speed) do not cause division-by-epsilon
numerical blowups (e.g., multipliers of 10^8) that trigger false alarms on
sub-Hertz frequency jitter.
"""
import json
import os
import numpy as np

# Feature order: ["rms", "dominant_freq", "spectral_energy", "kurtosis"]
# Physically grounded scale floors:
# - RMS: 0.01 g (accounts for sensor noise floor at +/-2g range)
# - Dominant Frequency: 2.0 Hz (accounts for FFT bin resolution & speed wander)
# - Spectral Energy: 100.0 (accounts for baseline noise power)
# - Kurtosis: 0.1 (accounts for statistical sample variance in kurtosis)
DEFAULT_SCALE_FLOORS = [0.01, 2.0, 100.0, 0.1]
DEFAULT_RELATIVE_FLOOR = 0.05  # 5% of mean magnitude minimum scale
DEFAULT_MIN_WINDOWS = 3


class BaselineCalibrator:
    """Computes and applies per-deployment healthy baseline standardization."""

    def __init__(self, scale_floors=None, relative_floor: float = DEFAULT_RELATIVE_FLOOR,
                 min_windows: int = DEFAULT_MIN_WINDOWS):
        self.scale_floors_config = list(scale_floors) if scale_floors is not None else DEFAULT_SCALE_FLOORS
        self.relative_floor = float(relative_floor)
        self.min_windows = int(min_windows)

        self.mean_ = None
        self.std_ = None
        self.effective_scale_ = None
        self.n_samples_ = 0

    def validate_calibration_data(self, calibration_features: np.ndarray) -> np.ndarray:
        """Validates that calibration features are finite, non-empty, and sufficient."""
        if calibration_features is None:
            raise ValueError("Calibration features cannot be None.")

        arr = np.asarray(calibration_features, dtype=float)

        if arr.ndim != 2:
            raise ValueError(f"Calibration features must be a 2D array (n_windows, n_features), got shape {arr.shape}.")

        if arr.shape[0] < self.min_windows:
            raise ValueError(
                f"Insufficient calibration windows: got {arr.shape[0]}, "
                f"minimum required is {self.min_windows}."
            )

        if arr.shape[1] != 4:
            raise ValueError(
                f"Expected 4 features (rms, dominant_freq, spectral_energy, kurtosis), got {arr.shape[1]}."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError("Calibration features contain NaN or Inf values.")

        # Check for completely flatlined (disconnected) sensor across all windows
        rms_vals = arr[:, 0]
        if np.all(rms_vals < 1e-6):
            raise ValueError("Calibration rejected: sensor appears disconnected (RMS ~ 0 across all windows).")

        return arr

    def fit(self, calibration_features: np.ndarray) -> "BaselineCalibrator":
        """Fits baseline statistics from healthy calibration feature windows.

        Parameters:
            calibration_features: (n_windows, n_features) matrix of feature vectors.
        """
        arr = self.validate_calibration_data(calibration_features)

        self.n_samples_ = int(arr.shape[0])
        self.mean_ = np.mean(arr, axis=0)
        self.std_ = np.std(arr, axis=0)

        # Compute robust effective scale floors to prevent zero-variance blowup
        base_floors = np.array(self.scale_floors_config, dtype=float)
        rel_floors = self.relative_floor * np.abs(self.mean_)
        floors = np.maximum(base_floors, rel_floors)

        # effective_scale_ is strictly positive and never smaller than the scale floor
        self.effective_scale_ = np.maximum(self.std_, floors)

        return self

    def transform(self, feature_vectors) -> np.ndarray:
        """Standardizes feature vectors relative to the fitted baseline.

        Returns:
            Standardized z-scores clipped to [-50, 50] to prevent downstream numerical overflow.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Calibrator has not been fitted yet. Record a healthy calibration "
                "window for this deployment before transforming features."
            )

        arr = np.asarray(feature_vectors, dtype=float)
        single = (arr.ndim == 1)
        if single:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != len(self.mean_):
            raise ValueError(f"Feature dimension mismatch: expected {len(self.mean_)}, got {arr.shape[1]}.")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Input feature vector contains NaN or Inf.")

        # Standardize using robust effective scale
        z = (arr - self.mean_) / self.effective_scale_

        # Defend against extreme outliers causing numerical overflow in MLP
        z = np.clip(z, -50.0, 50.0)

        return z[0] if single else z

    def fit_transform(self, calibration_features: np.ndarray) -> np.ndarray:
        """Fit baseline on healthy features and transform them."""
        return self.fit(calibration_features).transform(calibration_features)

    @property
    def is_fitted(self) -> bool:
        """Returns True if calibrator has been fitted with valid baseline statistics."""
        return self.mean_ is not None and self.effective_scale_ is not None

    def to_dict(self) -> dict:
        """Serializes calibrator state to JSON-compatible dictionary."""
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "effective_scale": self.effective_scale_.tolist() if self.effective_scale_ is not None else None,
            "scale_floors": self.scale_floors_config,
            "relative_floor": self.relative_floor,
            "min_windows": self.min_windows,
            "n_samples": self.n_samples_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineCalibrator":
        """Instantiates calibrator from state dictionary."""
        c = cls(
            scale_floors=d.get("scale_floors", DEFAULT_SCALE_FLOORS),
            relative_floor=d.get("relative_floor", DEFAULT_RELATIVE_FLOOR),
            min_windows=d.get("min_windows", DEFAULT_MIN_WINDOWS),
        )
        if d.get("mean") is not None and d.get("std") is not None:
            c.mean_ = np.array(d["mean"], dtype=float)
            c.std_ = np.array(d["std"], dtype=float)
            c.n_samples_ = int(d.get("n_samples", 0))

            if d.get("effective_scale") is not None:
                c.effective_scale_ = np.array(d["effective_scale"], dtype=float)
            else:
                # Backward-compatibility fallback: recompute effective scale with floors
                base_floors = np.array(c.scale_floors_config, dtype=float)
                rel_floors = c.relative_floor * np.abs(c.mean_)
                c.effective_scale_ = np.maximum(c.std_, np.maximum(base_floors, rel_floors))

        return c

    def save(self, path: str) -> None:
        """Safely saves calibrator to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        # Atomic rename on Windows/POSIX
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> "BaselineCalibrator":
        """Safely loads calibrator from JSON file, with comprehensive validation."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baseline file does not exist: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid baseline JSON file {path}: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"Invalid baseline structure in {path}, expected dictionary.")

        return cls.from_dict(data)