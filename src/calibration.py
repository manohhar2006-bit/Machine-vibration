"""
Adaptive baseline calibration.

The classifier this feeds is trained once, offline, on baseline-relative
(z-scored) features, and its weights never change after that (see
model.py / train.py). What changes per machine, per deployment, is only
the baseline stored here: a short calibration window is recorded on
whatever machine the rig is currently mounted on, this class computes
that machine's own mean/std per feature, and every subsequent reading is
expressed relative to that baseline rather than as a raw value.

Recalibrating for a different motor -- or the same motor after
reassembly -- means calling fit() again on a fresh calibration window.
Nothing about the trained network changes; only the reference point does.
"""
import json
import numpy as np


class BaselineCalibrator:
    def __init__(self, eps: float = 1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, calibration_features: np.ndarray) -> "BaselineCalibrator":
        """calibration_features: (n_windows, n_features) array collected
        during the calibration window (e.g. the first 30-60s on a machine,
        assumed to represent its normal operating condition)."""
        calibration_features = np.asarray(calibration_features, dtype=float)
        self.mean_ = calibration_features.mean(axis=0)
        self.std_ = calibration_features.std(axis=0)
        return self

    def transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError(
                "Calibrator has not been fit yet -- record a calibration "
                "window for this deployment before scoring anything."
            )
        return (feature_vectors - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, calibration_features: np.ndarray) -> np.ndarray:
        return self.fit(calibration_features).transform(calibration_features)

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineCalibrator":
        c = cls(eps=d.get("eps", 1e-8))
        if d.get("mean") is not None:
            c.mean_ = np.array(d["mean"])
            c.std_ = np.array(d["std"])
        return c

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BaselineCalibrator":
        with open(path) as f:
            return cls.from_dict(json.load(f))