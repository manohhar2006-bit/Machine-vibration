"""
Dataset loading.

Two sources, same output shape (windows, labels, sample_rate):

1. MaFaulDa -- download locally from Kaggle first, then point
   load_mafaulda() at the extracted folder.

   Expected layout after download:
     <root>/normal/*.csv
     <root>/imbalance/<grams>g/*.csv        e.g. 6g/, 10g/, 15g/, 20g/, 25g/, 30g/, 35g/
   Each CSV, 8 columns (per the official MAFAULDA column description):
     col 0            tachometer signal (derive rotation frequency from this)
     cols 1-3          underhang bearing accel: axial, radial, tangential
     cols 4-6          overhang bearing accel: axial, radial, tangential
     col 7             microphone
   Fixed sample rate: 50000 Hz, 5s per file (250,000 samples).

2. Synthetic placeholder (`generate_synthetic_dataset`) -- vibration-like
   signals with a controllable per-deployment gain, base frequency and
   noise floor, for testing the pipeline (features -> calibration ->
   classifier -> ablation) without a dataset or rig. Swap in
   `load_mafaulda()` for real data -- everything downstream (features.py,
   calibration.py, model.py) is agnostic to the source.
"""
import glob
import os

import numpy as np

SAMPLE_RATE_SYNTHETIC = 4000  # Hz -- matches an easily achievable Arduino sampling rate
MAFAULDA_SAMPLE_RATE = 50000  # Hz -- fixed by the MaFaulDa acquisition setup


# ---------------------------------------------------------------------
# 1. MaFaulDa loader (for local use after downloading the dataset)
# ---------------------------------------------------------------------
def load_mafaulda(root: str, bearing_axis: int = 2, window_s: float = 1.0):
    """Loads MaFaulDa 'normal' and 'imbalance' recordings, splits each
    long recording into fixed-length windows, and returns them as a
    binary normal(0) vs faulty(1) dataset.

    bearing_axis: which accelerometer column to use as the vibration
    channel. Default 2 = underhang bearing, radial direction, the
    closest analog to a single-axis accelerometer mounted radially on
    a motor housing. Column layout: 0=tachometer, 1=underhang axial,
    2=underhang radial, 3=underhang tangential, 4=overhang axial,
    5=overhang radial, 6=overhang tangential, 7=microphone.
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
    """Loads MaFaulDa recordings as load_mafaulda() does, but additionally
    returns a deployment identifier per window.

    MaFaulDa filenames encode the rotation-frequency setpoint in Hz (e.g.
    "12.288.csv"), and the same set of setpoints recurs across the
    normal recordings and the imbalance recordings at each load weight.
    Treating each setpoint as a distinct deployment allows a
    per-deployment calibration baseline to be computed from that
    setpoint's own normal recording, then applied to every window
    (normal and imbalance) recorded at that setpoint -- the same
    calibration scheme used at inference time (see calibration.py).

    If a filename cannot be parsed as a frequency, its full path is
    used as the deployment identifier instead, so the file still loads
    but is not paired with recordings from other folders.

    Returns (windows, labels, groups, sample_rate), where groups[i] is
    the deployment identifier for windows[i].
    """
    sample_rate = MAFAULDA_SAMPLE_RATE
    window_len = int(window_s * sample_rate)

    def _deployment_id(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            freq = float(stem)
        except ValueError:
            return path
        # MaFaulDa setpoints are spaced roughly 1 Hz apart, but the
        # exact measured frequency logged in each filename varies
        # slightly between recordings aimed at the same nominal
        # setpoint. Rounding to the nearest whole Hz groups recordings
        # by setpoint rather than by exact logged value.
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
            f"No MaFaulDa files found under {root}. Expected "
            f"{root}/normal/*.csv and {root}/imbalance/<grams>g/*.csv"
        )

    return np.array(windows), np.array(labels), np.array(groups), sample_rate


# ---------------------------------------------------------------------
# 2. Synthetic placeholder (no download required)
# ---------------------------------------------------------------------
def _make_signal(duration_s, sample_rate, base_freq, gain, noise_std, rng):
    t = np.arange(0, duration_s, 1.0 / sample_rate)
    signal = gain * np.sin(2 * np.pi * base_freq * t)
    signal += 0.3 * gain * np.sin(2 * np.pi * 2 * base_freq * t)
    signal += rng.normal(0, noise_std, size=t.shape)
    return signal


def generate_synthetic_dataset(n_windows_per_class=40, window_s=1.0,
                                sample_rate=SAMPLE_RATE_SYNTHETIC,
                                deployment_shift=0.0, seed=0):
    """One 'deployment' represents one synthetic machine, with its own
    gain, rotor frequency, and noise floor -- standing in for a
    different motor, or the same motor after reassembly.

    deployment_shift only seeds an independent random draw of that
    machine's characteristics (gain, frequency, noise floor); it is
    deliberately not a smooth linear parameter. A fleet of real motors
    does not vary continuously along one axis with a fittable global
    trend -- each unit's mounting stiffness, bearing wear, and sensor
    contact are approximately independent draws. A smooth linear shift
    would let a single global scaler extrapolate to unseen deployments,
    understating the generalization problem per-deployment calibration
    is intended to address.
    """
    deployment_rng = np.random.default_rng(seed * 97 + int(deployment_shift * 1000))
    gain = deployment_rng.uniform(0.5, 2.6)
    base_freq = deployment_rng.uniform(40.0, 60.0)
    noise_floor = deployment_rng.uniform(0.03, 0.18)

    rng = np.random.default_rng(seed)

    windows, labels = [], []
    for _ in range(n_windows_per_class):
        # normal: baseline amplitude for this deployment
        windows.append(_make_signal(window_s, sample_rate, base_freq,
                                     gain=0.4 * gain, noise_std=noise_floor, rng=rng))
        labels.append(0)
    for _ in range(n_windows_per_class):
        # faulty (imbalance): stronger harmonic + higher noise, same deployment
        windows.append(_make_signal(window_s, sample_rate, base_freq,
                                     gain=1.1 * gain, noise_std=noise_floor * 2.2, rng=rng))
        labels.append(1)

    order = rng.permutation(len(windows))
    windows = np.array(windows)[order]
    labels = np.array(labels)[order]
    return windows, labels, sample_rate