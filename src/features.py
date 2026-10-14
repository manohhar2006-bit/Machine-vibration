"""
Feature extraction from a vibration window.

Extracts 4 time- and frequency-domain features from 1-D vibration windows:
  1. RMS (Root Mean Square) -- physical magnitude/energy in time domain.
  2. Dominant Frequency -- primary oscillation frequency within (0 Hz, max_freq] (excludes DC).
  3. Spectral Energy -- total power within the relevant sub-band (0 Hz, max_freq].
  4. Kurtosis -- Fisher excess kurtosis (peakedness/impulsiveness of vibration).

Imported by offline training (train.py, ablation.py), simulation (simulate_stream.py),
and live inference (serving/app.py) to guarantee identical feature extraction.
"""
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis

FEATURE_ORDER = ["rms", "dominant_freq", "spectral_energy", "kurtosis"]

DEFAULT_MAX_FREQ = 500.0  # Hz -- upper bound for machinery vibration monitoring


def validate_window(window) -> np.ndarray:
    """Validates that a window is numeric, non-empty, 1-dimensional, and finite.

    Raises:
        ValueError: If input is None, empty, non-1D, or contains NaN / Inf.
    """
    if window is None:
        raise ValueError("Window cannot be None.")

    arr = np.asarray(window, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"Window must be 1-dimensional, got {arr.ndim}D array with shape {arr.shape}.")

    if len(arr) < 4:
        raise ValueError(f"Window too short ({len(arr)} samples). Minimum 4 samples required.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Window contains non-finite values (NaN or Inf).")

    return arr


def compute_spectrum(window: np.ndarray, sample_rate: float,
                     apply_hann: bool = True):
    """Computes real FFT with mean subtraction and Hann windowing.

    Returns:
        freqs (np.ndarray): Frequency bins in Hz.
        spectrum (np.ndarray): Single-sided FFT magnitude.
    """
    n = len(window)
    # Remove DC bias (mean subtraction)
    detrended = window - np.mean(window)

    if apply_hann and n >= 4:
        # Hann window reduces spectral leakage from non-integer cycle boundaries
        win = np.hanning(n)
        windowed = detrended * win
    else:
        windowed = detrended

    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(n, d=1.0 / sample_rate)
    return freqs, spectrum


def extract_features(window, sample_rate: float,
                     max_freq: float = DEFAULT_MAX_FREQ,
                     return_spectrum: bool = False) -> dict:
    """Compute RMS, dominant frequency, spectral energy, and kurtosis
    for one buffered vibration window.

    Definitions and Units:
      - RMS (g or m/s^2): Root mean square value of raw acceleration.
            RMS = sqrt( (1/N) * sum(x_i^2) )
      - Dominant Frequency (Hz): Peak spectral component in (0 Hz, max_freq].
            Excludes 0 Hz (DC) component to avoid picking sensor offset.
      - Spectral Energy (amplitude^2): Sum of squared FFT magnitudes within (0 Hz, max_freq].
            E_spectral = sum_{0 < f_k <= max_freq} |X(f_k)|^2
      - Kurtosis (dimensionless): Fisher excess kurtosis (normal distribution = 0.0).
            Returns 0.0 for constant or zero-variance signals to avoid NaN.

    Parameters:
        window: 1D array-like containing accelerometer samples.
        sample_rate: Sampling frequency in Hz (> 0).
        max_freq: Upper frequency limit for band analysis (default 500 Hz).
        return_spectrum: If True, includes frequency and magnitude arrays in return dict.

    Returns:
        dict: Containing 'rms', 'dominant_freq', 'spectral_energy', 'kurtosis'.
    """
    arr = validate_window(window)

    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    # 1. RMS
    rms = float(np.sqrt(np.mean(arr ** 2)))

    # Detect flatline / constant / zero-variance signal
    arr_std = float(np.std(arr))
    if arr_std < 1e-12:
        # Constant or zero signal: perfectly stationary, no oscillatory energy, kurtosis=0
        feat = {
            "rms": rms,
            "dominant_freq": 0.0,
            "spectral_energy": 0.0,
            "kurtosis": 0.0,
        }
        if return_spectrum:
            freqs = rfftfreq(len(arr), d=1.0 / sample_rate)
            mask = freqs <= max_freq
            feat["freqs"] = freqs[mask]
            feat["magnitude"] = np.zeros(np.count_nonzero(mask), dtype=float)
        return feat

    # 2. FFT Spectrum (with Hann window and mean subtraction)
    freqs, spectrum = compute_spectrum(arr, sample_rate, apply_hann=True)

    # Band mask: exclude DC (f > 0) up to max_freq
    band_mask = (freqs > 0.0) & (freqs <= max_freq)
    band_spectrum = spectrum[band_mask]
    band_freqs = freqs[band_mask]

    # 3. Dominant Frequency
    if len(band_spectrum) > 0 and np.max(band_spectrum) > 1e-12:
        dominant_freq = float(band_freqs[np.argmax(band_spectrum)])
    else:
        dominant_freq = 0.0

    # 4. Spectral Energy
    if len(band_spectrum) > 0:
        spectral_energy = float(np.sum(band_spectrum ** 2))
    else:
        spectral_energy = 0.0

    # 5. Fisher Excess Kurtosis (0 for normal distribution)
    try:
        kurt = float(kurtosis(arr, fisher=True, bias=False))
        if not np.isfinite(kurt):
            kurt = 0.0
    except Exception:
        kurt = 0.0

    feat = {
        "rms": rms,
        "dominant_freq": dominant_freq,
        "spectral_energy": spectral_energy,
        "kurtosis": kurt,
    }

    if return_spectrum:
        disp_mask = freqs <= max_freq
        feat["freqs"] = freqs[disp_mask]
        feat["magnitude"] = spectrum[disp_mask]

    return feat


def features_to_vector(feat: dict) -> np.ndarray:
    """Convert a feature dict to a fixed-order numpy vector, so the
    classifier always receives features in the exact same column order."""
    return np.array([feat[k] for k in FEATURE_ORDER], dtype=float)


def window_to_vector(window, sample_rate: float,
                     max_freq: float = DEFAULT_MAX_FREQ) -> np.ndarray:
    """Extract features and return as ordered 1D numpy vector."""
    return features_to_vector(extract_features(window, sample_rate, max_freq=max_freq))