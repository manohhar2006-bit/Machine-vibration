"""
Offline training and evaluation pipeline for Predictive Machine Vibration Monitoring.

Supports:
1. Supervised training on the MaFaulDa benchmark dataset if raw CSV files are
   present in data/raw.
2. Synthetic multi-deployment training (--synthetic) if raw benchmark files are absent.
3. Rigorous held-out deployment evaluation computing:
   - Full confusion matrix
   - Normal-class precision, recall, F1-score
   - Fault-class precision, recall, F1-score
   - False Positive Rate (FPR) and False Negative Rate (FNR)
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from calibration import BaselineCalibrator
from dataset import (
    SAMPLE_RATE_SYNTHETIC,
    generate_synthetic_deployments,
    load_mafaulda_by_deployment,
)
from features import window_to_vector
from model import build_model, save_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "data", "raw")
MODEL_PATH = os.path.join(BASE_DIR, "..", "serving", "model_store", "classifier.joblib")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "training")
TEST_FRACTION = 0.25
RANDOM_STATE = 42
WINDOW_S = 1.0
CALIBRATION_WINDOWS = 3


def build_calibrated_set(windows, labels, groups, sample_rate, deployment_ids,
                          calibration_windows=CALIBRATION_WINDOWS):
    """Extracts features and applies per-deployment baseline calibration."""
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
        print(f"Skipped {len(skipped)} deployment(s) with insufficient normal windows for calibration: {skipped}")

    if not X:
        raise ValueError("No valid calibration windows could be extracted.")

    return np.vstack(X), np.concatenate(y)


def evaluate_model(model, X_test, y_test, results_dir=RESULTS_DIR):
    """Computes full evaluation metrics including FPR, FNR, and class-specific metrics."""
    os.makedirs(results_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    normal_precision = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
    normal_recall = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
    normal_f1 = float(f1_score(y_test, y_pred, pos_label=0, zero_division=0))

    fault_precision = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    fault_recall = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    fault_f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    accuracy = float((tp + tn) / (tp + tn + fp + fn))

    report = classification_report(y_test, y_pred, target_names=["normal", "imbalance"], output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "rates": {
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        },
        "normal_class": {
            "precision": normal_precision,
            "recall": normal_recall,
            "f1_score": normal_f1,
        },
        "fault_class": {
            "precision": fault_precision,
            "recall": fault_recall,
            "f1_score": fault_f1,
        },
        "classification_report": report,
    }

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Faulty (1)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Held-Out Deployment Confusion Matrix")
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "confusion_matrix.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n================== EVALUATION REPORT ==================")
    print(f"Overall Accuracy:       {accuracy * 100:.2f}%")
    print(f"Confusion Matrix:       TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"False Positive Rate:    {fpr * 100:.2f}% (healthy machines falsely alarmed)")
    print(f"False Negative Rate:    {fnr * 100:.2f}% (unbalance faults missed)")
    print(f"Normal Class  -- P: {normal_precision:.3f}, R: {normal_recall:.3f}, F1: {normal_f1:.3f}")
    print(f"Faulty Class  -- P: {fault_precision:.3f}, R: {fault_recall:.3f}, F1: {fault_f1:.3f}")
    print(f"Saved artifacts to {results_dir}")
    print("=======================================================\n")

    return metrics


def train_pipeline(use_synthetic=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not use_synthetic and os.path.exists(DATA_ROOT) and glob.glob(os.path.join(DATA_ROOT, "normal", "*.csv")):
        print(f"Loading raw MaFaulDa files from {DATA_ROOT}...")
        windows, labels, groups, sample_rate = load_mafaulda_by_deployment(DATA_ROOT, window_s=WINDOW_S)
    else:
        if not use_synthetic:
            print(f"Notice: Raw MaFaulDa data directory '{DATA_ROOT}' not found.")
            print("Switching to synthetic multi-deployment generation for pipeline verification.")
        shifts = [-15, -10, -5, 0, 5, 10, 15, 20]
        windows, labels, groups, sample_rate = generate_synthetic_deployments(
            shifts, n_windows_per_class=40, window_s=0.5, sample_rate=SAMPLE_RATE_SYNTHETIC, seed=RANDOM_STATE
        )

    unique_deployments = np.unique(groups)
    print(f"Loaded {len(windows)} windows across {len(unique_deployments)} deployments (sample_rate={sample_rate} Hz)")

    train_deps, test_deps = train_test_split(
        unique_deployments, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )

    train_X, train_y = build_calibrated_set(windows, labels, groups, sample_rate, train_deps)
    test_X, test_y = build_calibrated_set(windows, labels, groups, sample_rate, test_deps)

    print(f"Train set: {len(train_X)} windows from {len(train_deps)} deployments")
    print(f"Held-out test set: {len(test_X)} windows from {len(test_deps)} deployments")

    model = build_model(random_state=RANDOM_STATE)
    model.fit(train_X, train_y)

    evaluate_model(model, test_X, test_y, results_dir=RESULTS_DIR)

    save_model(model, MODEL_PATH)
    print(f"Successfully saved trained classifier to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vibration classifier.")
    parser.add_argument("--synthetic", action="store_true", help="Force use of synthetic vibration dataset.")
    args = parser.parse_args()
    train_pipeline(use_synthetic=args.synthetic)