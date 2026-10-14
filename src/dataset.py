"""
Dataset loading and synthesis for Predictive Machine Vibration Monitoring.

Two sources, consistent output shape (windows, labels, sample_rate):

1. MaFaulDa -- download locally from Kaggle/research repo, then point
   load_mafaulda() or load_mafaulda_by_deployment() at the extracted folder.
   Fixed sample rate: 50,000 Hz.

2. Synthetic generator (`generate_synthetic_dataset`) -- realistic vibration
   signals modeling 1X rotational frequency, unbalance harmonics (2X),
   mounting gain variation, and sensor noise floor at the hardware-feasible
   sample rate (800 Hz). Used for ablation studies and offline testing when
   raw dataset files are not present.
"""
import glob
import os
import numpy as np

SAMPLE_RATE_SYNTHETIC = 800  # Hz -- matches hardware ADXL345 ODR and Arduino acquisition
MAFAULDA_SAMPLE_RATE = 50000  # Hz -- fixed by the MaFaulDa laboratory acquisition setup


# ---------------------------------------------------------------------
# 1. MaFaulDa loader (for local use after downloading the dataset)
# ---------------------------------------------------------------------
def load_mafaulda(root: str, bearing_axis: int = 2, window_s: float = 1.0):
    """Loads MaFaulDa 'normal' and 'imbalance' recordings, splits each
    long recording into fixed-length windows, and returns them as a
    binary normal(0) vs faulty(1) dataset.

    bearing_axis: which accelerometer column to use as the vibration
    channel. Default 2 = underhang bearing, radial direction.
    Column layout: 0=tachometer, 1=underhang axial, 2=underhang radial,
    3=underhang tangential, 4=overhang axial, 5=overhang radial,
    6=overhang tangential, 7=microphone.
    """
    sample_rate = MAFAULDA_SAMPLE_RATE
    window_len = int(window_s * sample_rate)

    def _windows_from_file(path):
        data = np.loadtxt(path, delimiter=",")
        signal = data[:, bearing_axis]
        n_windows = len(signal) // window_len
        return [signal[i * window_len:(i + 1) * window_len] for i in range(n_windows)]

    windows, labels = [], []
    for path in glob.glob(os.path.join(root, "normal", "*.csv")):
        for w in _windows_from_file(path):
            windows.append(w)
            labels.append(0)

    for path in glob.glob(os.path.join(root, "imbalance", "*", "*.csv")):
        for w in _windows_from_file(path):
            windows.append(w)
            labels.append(1)

    if not windows:
        raise FileNotFoundError(
            f"No MaFaulDa files found under {root}. Expected "
            f"{root}/normal/*.csv and {root}/imbalance/<grams>g/*.csv"
        )

    return np.array(windows), np.array(labels), sample_rate


def load_mafaulda_by_deployment(root: str, bearing_axis: int = 2, window_s: float = 1.0):
    """Loads MaFaulDa recordings grouped by rotation-frequency setpoint (deployment)."""
    sample_rate = MAFAULDA_SAMPLE_RATE
    window_len = int(window_s * sample_rate)

    def _deployment_id(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            freq = float(stem)
        except ValueError:
            return path
        return round(freq)

    def _windows_from_file(path):
        data = np.loadtxt(path, delimiter=",")
        signal = data[:, bearing_axis]
        n_windows = len(signal) // window_len
        return [signal[i * window_len:(i + 1) * window_len] for i in range(n_windows)]

    windows, labels, groups = [], [], []

    for path in glob.glob(os.path.join(root, "normal", "*.csv")):
        deployment_id = _deployment_id(path)
        for w in _windows_from_file(path):
            windows.append(w)
            labels.append(0)
            groups.append(deployment_id)

    for path in glob.glob(os.path.join(root, "imbalance", "*", "*.csv")):
        deployment_id = _deployment_id(path)
        for w in _windows_from_file(path):
            windows.append(w)
            labels.append(1)
            groups.append(deployment_id)

    if not windows:
        raise FileNotFoundError(
            f"No MaFaulDa files found under '{root}'. Expected "
            f"'{root}/normal/*.csv' and '{root}/imbalance/<grams>g/*.csv'."
        )

    return np.array(windows), np.array(labels), np.array(groups), sample_rate


# ---------------------------------------------------------------------
# 2. Synthetic vibration generator (physics-based, no external download needed)
# ---------------------------------------------------------------------
def _make_signal(duration_s, sample_rate, base_freq, gain, noise_std, rng, imbalance=False):
    """Synthesizes a realistic motor vibration signal:
    - Fundamental rotational frequency (1X = base_freq)
    - Rotor unbalance creates a pronounced 1X centrifugal radial acceleration component
      and a 2X harmonic.
    - Broadband Gaussian sensor/mechanical noise.
    """
    t = np.arange(0, duration_s, 1.0 / sample_rate)
    if not imbalance:
        # Normal condition: low-amplitude 1X residual vibration, small harmonics, low noise
        signal = 0.35 * gain * np.sin(2 * np.pi * base_freq * t)
        signal += 0.08 * gain * np.sin(2 * np.pi * 2 * base_freq * t)
        signal += rng.normal(0, noise_std, size=t.shape)
    else:
        # Imbalance condition: elevated 1X component (centrifugal force proportional to omega^2 * unbalance)
        # plus higher harmonic distortion and mechanical shudder
        signal = 1.25 * gain * np.sin(2 * np.pi * base_freq * t)
        signal += 0.40 * gain * np.sin(2 * np.pi * 2 * base_freq * t)
        signal += rng.normal(0, noise_std * 2.0, size=t.shape)
    return signal


def generate_synthetic_dataset(n_windows_per_class=40, window_s=0.5,
                                sample_rate=SAMPLE_RATE_SYNTHETIC,
                                deployment_shift=0.0, seed=0):
    """Generates synthetic vibration windows for one machine deployment."""
    deployment_rng = np.random.default_rng(seed * 97 + int(abs(deployment_shift) * 1000) + (1 if deployment_shift < 0 else 0))
    # Motor parameters for this deployment
    gain = deployment_rng.uniform(0.8, 2.2)
    base_freq = deployment_rng.uniform(25.0, 55.0)  # ~1500 to 3300 RPM
    noise_floor = deployment_rng.uniform(0.04, 0.12)

    rng = np.random.default_rng(seed)

    windows, labels = [], []
    for _ in range(n_windows_per_class):
        windows.append(_make_signal(window_s, sample_rate, base_freq,
                                     gain=gain, noise_std=noise_floor, rng=rng, imbalance=False))
        labels.append(0)
    for _ in range(n_windows_per_class):
        windows.append(_make_signal(window_s, sample_rate, base_freq,
                                     gain=gain, noise_std=noise_floor, rng=rng, imbalance=True))
        labels.append(1)

    order = rng.permutation(len(windows))
    windows = np.array(windows)[order]
    labels = np.array(labels)[order]
    return windows, labels, sample_rate


def generate_synthetic_deployments(deployment_shifts, n_windows_per_class=40,
                                   window_s=0.5, sample_rate=SAMPLE_RATE_SYNTHETIC, seed=42):
    """Generates synthetic vibration windows across multiple independent deployments."""
    all_windows, all_labels, all_groups = [], [], []
    for i, shift in enumerate(deployment_shifts):
        w, l, _ = generate_synthetic_dataset(
            n_windows_per_class=n_windows_per_class,
            window_s=window_s,
            sample_rate=sample_rate,
            deployment_shift=shift,
            seed=seed + i * 13,
        )
        all_windows.append(w)
        all_labels.append(l)
        all_groups.append(np.full(len(l), shift))

    return np.vstack(all_windows), np.concatenate(all_labels), np.concatenate(all_groups), sample_rate