"""
Feature extraction from a vibration window.

Imported by both the offline training pipeline (train.py, ablation.py)
and the live inference path (serving/app.py), so training-time and
deployment-time features are computed by identical code.
"""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis

FEATURE_ORDER = ["rms", "dominant_freq", "spectral_energy", "kurtosis"]


DEFAULT_MAX_FREQ = 500.0  # Hz -- upper bound for frequency-domain features


def extract_features(window: np.ndarray, sample_rate: float,
                      max_freq: float = DEFAULT_MAX_FREQ) -> dict:
    """Compute RMS, dominant frequency, spectral energy, and kurtosis
    for one buffered vibration window (a 1-D array of accelerometer
    samples sampled at a fixed interval).

    Frequency-domain features (dominant_freq, spectral_energy) are
    computed only over frequencies up to max_freq. This project's
    rotor speeds fall in the 12-61 Hz range, and the associated fault
    signature occurs at that frequency and its low harmonics; frequency
    content above a few hundred Hz is unrelated to rotor imbalance and,
    if included, can dominate the spectrum and mask the fault signal.
    """
    window = np.asarray(window, dtype=float)

    rms = float(np.sqrt(np.mean(window ** 2)))

    spectrum = np.abs(rfft(window - window.mean()))
    freqs = rfftfreq(len(window), d=1.0 / sample_rate)

    band_mask = freqs <= max_freq
    band_spectrum = spectrum[band_mask]
    band_freqs = freqs[band_mask]

    dominant_freq = float(band_freqs[np.argmax(band_spectrum)]) if len(band_spectrum) else 0.0
    spectral_energy = float(np.sum(band_spectrum ** 2))

    kurt = float(kurtosis(window, fisher=True, bias=False))

    return {
        "rms": rms,
        "dominant_freq": dominant_freq,
        "spectral_energy": spectral_energy,
        "kurtosis": kurt,
    }


def features_to_vector(feat: dict) -> np.ndarray:
    """Convert a feature dict to a fixed-order numpy vector, so the
    classifier always receives features in the same column order."""
    return np.array([feat[k] for k in FEATURE_ORDER], dtype=float)


def window_to_vector(window: np.ndarray, sample_rate: float,
                      max_freq: float = DEFAULT_MAX_FREQ) -> np.ndarray:
    return features_to_vector(extract_features(window, sample_rate, max_freq=max_freq))