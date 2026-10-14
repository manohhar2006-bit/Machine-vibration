"""
Ablation: does per-deployment baseline calibration improve generalization
to an unseen machine, compared to a single fixed scale learned once from
the training deployments?

To isolate the effect of calibration timing rather than normalization
itself, both classifiers below receive standardized features; the only
difference is when the scale is computed.

  - clf_fixed     features standardized with one StandardScaler fit once
                  on the pooled training deployments and then frozen.
  - clf_adaptive  features standardized per deployment via
                  BaselineCalibrator, recomputed on every machine.

Both classifiers are evaluated on deployments neither was trained on.
The fixed path reuses the frozen training-time scaler; the adaptive
path records a fresh calibration window on the held-out deployment.
"""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from calibration import BaselineCalibrator
from dataset import generate_synthetic_dataset
from features import window_to_vector
from model import build_model

SAMPLE_RATE = 4000
TRAIN_SHIFTS = [-8, -4, 0, 4, 8]       # deployments used for training
HELD_OUT_SHIFTS = [-12, 12, 20]        # never-seen deployments
N_PER_CLASS = 40
CALIBRATION_WINDOWS = 15


def build_deployment(shift, seed):
    windows, labels, sr = generate_synthetic_dataset(
        n_windows_per_class=N_PER_CLASS,
        sample_rate=SAMPLE_RATE,
        deployment_shift=shift,
        seed=seed,
    )
    feats = np.array([window_to_vector(w, sr) for w in windows])
    return feats, labels


def run_ablation():
    train_X, train_y = [], []
    for i, shift in enumerate(TRAIN_SHIFTS):
        feats, labels = build_deployment(shift, seed=100 + i)
        train_X.append(feats)
        train_y.append(labels)
    train_X = np.vstack(train_X)
    train_y = np.concatenate(train_y)

    # Fixed path: one scaler fit once on pooled training data, then frozen.
    fixed_scaler = StandardScaler().fit(train_X)
    clf_fixed = build_model().fit(fixed_scaler.transform(train_X), train_y)

    # Adaptive path: a calibrator computed independently within each
    # training deployment, then pooled for training.
    norm_train_X = []
    for i, shift in enumerate(TRAIN_SHIFTS):
        feats, labels = build_deployment(shift, seed=100 + i)
        normal_idx = np.where(labels == 0)[0][:CALIBRATION_WINDOWS]
        calib = BaselineCalibrator().fit(feats[normal_idx])
        norm_train_X.append(calib.transform(feats))
    norm_train_X = np.vstack(norm_train_X)
    clf_adaptive = build_model().fit(norm_train_X, train_y)

    print("held-out deployment generalization (classifier never trained on these)\n")
    header = f"{'shift':>7} | {'fixed acc':>9} {'fixed auc':>9} | {'adapt acc':>9} {'adapt auc':>9}"
    print(header)
    print("-" * len(header))

    fixed_accs, adapt_accs = [], []
    for i, shift in enumerate(HELD_OUT_SHIFTS):
        feats, labels = build_deployment(shift, seed=200 + i)

        fixed_feats = fixed_scaler.transform(feats)
        fixed_pred = clf_fixed.predict(fixed_feats)
        fixed_proba = clf_fixed.predict_proba(fixed_feats)[:, 1]

        normal_idx = np.where(labels == 0)[0][:CALIBRATION_WINDOWS]
        calib = BaselineCalibrator().fit(feats[normal_idx])
        adapt_feats = calib.transform(feats)
        adapt_pred = clf_adaptive.predict(adapt_feats)
        adapt_proba = clf_adaptive.predict_proba(adapt_feats)[:, 1]

        fixed_acc = accuracy_score(labels, fixed_pred)
        fixed_auc = roc_auc_score(labels, fixed_proba)
        adapt_acc = accuracy_score(labels, adapt_pred)
        adapt_auc = roc_auc_score(labels, adapt_proba)
        fixed_accs.append(fixed_acc)
        adapt_accs.append(adapt_acc)

        print(f"{shift:>+7} | {fixed_acc:>9.3f} {fixed_auc:>9.3f} | {adapt_acc:>9.3f} {adapt_auc:>9.3f}")

    print("-" * len(header))
    print(f"{'mean':>7} | {np.mean(fixed_accs):>9.3f} {'':>9} | {np.mean(adapt_accs):>9.3f}")
    print(f"\nshifts -8..8 were used for training; -12, 12, 20 are unseen deployments.")
    print(f"'fixed' = one StandardScaler fit once on training data (a fixed-baseline")
    print(f"approach). 'adapt' = BaselineCalibrator refit on each deployment's own")
    print(f"calibration window. Same classifier architecture and training deployments")
    print(f"in both cases; only the scale source differs.")


if __name__ == "__main__":
    run_ablation()