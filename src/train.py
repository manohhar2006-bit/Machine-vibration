"""
Offline training on the MaFaulDa dataset.

Each MaFaulDa recording is made at a fixed rotation-frequency setpoint,
and the same set of setpoints recurs across the normal recordings and
the imbalance recordings (see dataset.load_mafaulda_by_deployment).
Each setpoint is treated as one deployment: its own normal recording
supplies the calibration baseline, and every window recorded at that
setpoint, normal and imbalance alike, is normalized against that
baseline before being added to the pooled training set.

Deployments, not windows, are split into training and held-out sets, so
no calibration statistics leak between the two -- the held-out accuracy
reported here reflects generalization to a rotation-frequency setpoint
the classifier never saw calibration or training data from.
"""
import json
import os

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from calibration import BaselineCalibrator
from dataset import load_mafaulda_by_deployment
from features import window_to_vector
from model import build_model, save_model

DATA_ROOT = "../data/raw"
MODEL_PATH = "../serving/model_store/classifier.joblib"
RESULTS_DIR = "../results/training"
TEST_FRACTION = 0.25
RANDOM_STATE = 42
WINDOW_S = 1.0
CALIBRATION_WINDOWS = 3  # per deployment, out of 5 normal windows at WINDOW_S=1.0


def build_calibrated_set(windows, labels, groups, sample_rate, deployment_ids,
                          calibration_windows=CALIBRATION_WINDOWS):
    """Extracts features and applies per-deployment calibration for the
    given subset of deployment identifiers, then pools the result.

    A fixed number of each deployment's normal windows are reserved for
    computing the calibration baseline and excluded from the pooled
    set. Without this split, a deployment's normal windows would be
    evaluated against a baseline computed from themselves, which is not
    a genuine test of generalization to unseen normal readings.
    """
    X, y = [], []
    skipped = []
    for deployment_id in deployment_ids:
        mask = groups == deployment_id
        dep_windows = windows[mask]
        dep_labels = labels[mask]

        feats = np.array([window_to_vector(w, sample_rate) for w in dep_windows])
        normal_idx = np.where(dep_labels == 0)[0]
        if len(normal_idx) <= calibration_windows:
            skipped.append(deployment_id)
            continue

        calib_idx = normal_idx[:calibration_windows]
        eval_idx = np.setdiff1d(np.arange(len(dep_labels)), calib_idx)

        calibrator = BaselineCalibrator().fit(feats[calib_idx])
        X.append(calibrator.transform(feats[eval_idx]))
        y.append(dep_labels[eval_idx])

    if skipped:
        print(f"skipped {len(skipped)} deployment(s) with insufficient normal "
              f"windows for calibration: {skipped}")

    return np.vstack(X), np.concatenate(y)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    windows, labels, groups, sample_rate = load_mafaulda_by_deployment(DATA_ROOT, window_s=WINDOW_S)
    unique_deployments = np.unique(groups)
    print(f"loaded {len(windows)} windows across {len(unique_deployments)} deployments")

    train_deployments, test_deployments = train_test_split(
        unique_deployments, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )

    train_X, train_y = build_calibrated_set(windows, labels, groups, sample_rate, train_deployments)
    test_X, test_y = build_calibrated_set(windows, labels, groups, sample_rate, test_deployments)

    model = build_model()
    model.fit(train_X, train_y)

    train_acc = model.score(train_X, train_y)
    test_acc = model.score(test_X, test_y)
    print(f"train    -- deployments: {len(train_deployments):3d}  windows: {len(train_X):5d}  accuracy: {train_acc:.3f}")
    print(f"held-out -- deployments: {len(test_deployments):3d}  windows: {len(test_X):5d}  accuracy: {test_acc:.3f}")

    test_pred = model.predict(test_X)
    report = classification_report(test_y, test_pred, target_names=["normal", "imbalance"], output_dict=True)
    print(classification_report(test_y, test_pred, target_names=["normal", "imbalance"]))

    metrics = {
        "n_deployments_total": int(len(unique_deployments)),
        "n_deployments_train": int(len(train_deployments)),
        "n_deployments_test": int(len(test_deployments)),
        "n_windows_train": int(len(train_X)),
        "n_windows_test": int(len(test_X)),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_classification_report": report,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(test_y, test_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["normal", "imbalance"]).plot(ax=ax, colorbar=False)
    ax.set_title("Held-out deployments: confusion matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    save_model(model, MODEL_PATH)
    print(f"saved classifier -> {MODEL_PATH}")
    print(f"saved metrics and confusion matrix -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()