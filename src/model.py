"""
Classifier definition and persistence.

Uses scikit-learn's MLPClassifier: a few KB to serialize (joblib), no
native/binary dependency issues on Render's free tier, and trains in
milliseconds at this feature-vector scale (4 features in, tens of
examples per deployment). Swap for a tf.keras.Sequential here if
TensorFlow/Keras is preferred -- nothing else in the pipeline depends
on which library this uses.
"""
import joblib
from sklearn.neural_network import MLPClassifier


def build_model(hidden_layer_sizes=(16, 8), random_state=42):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        max_iter=2000,
        random_state=random_state,
    )


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)