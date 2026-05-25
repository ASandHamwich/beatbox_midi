"""
Train a classifier on beatbox audio features.

Usage:
    uv run python -m training.train # default: random_forest
    uv run python -m training.train --model svm
    uv run python -m training.train --model gradient_boosting --no-save
"""

import argparse
import logging
import time
from datetime import datetime
from utils.logger import setup_logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


from training.config import (
    CLASSES,
    CLASSIFIERS_CFG,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from training.data import load_beatbox
from training.features import build_feature_matrix

logger = logging.getLogger("training.train")


# ── Classifier registry ───────────────────────────────────────────────────────
# Maps the class-name strings from conf/classifiers.yaml to actual sklearn types.
# To add a new classifier: add an entry in conf/classifiers.yaml and import the
# class here.
_CLASS_MAP: dict[str, type] = {
    "RandomForestClassifier":        RandomForestClassifier,
    "SVC":                           SVC,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
}


def build_classifier(name: str = "random_forest") -> ClassifierMixin:
    """Return a fresh, unfitted classifier configured via ``conf/classifiers.yaml``.

    Args:
        name: Key from ``conf/classifiers.yaml`` (e.g. ``"random_forest"``).

    Raises:
        ValueError: If *name* is not present in the YAML registry.
    """
    if name not in CLASSIFIERS_CFG:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(CLASSIFIERS_CFG)}"
        )
    cfg    = CLASSIFIERS_CFG[name]
    cls    = _CLASS_MAP[cfg["class"]]
    params = dict(cfg.get("params", {}))
    params.setdefault("random_state", RANDOM_STATE)  # injected from model.yaml
    return cls(**params)


# Pipeline steps

def _load_data():
    """
    Step 1 — load the dataset and log a summary.
    """
    logger.info("── Step 1/4  Loading dataset")
    df = load_beatbox(split="train")
    logger.info("  %d samples  ·  %d classes", len(df), df["label"].nunique())
    return df


def _extract_features(df):
    """
    Step 2 — build the feature matrix and encode string labels to integers.
    """
    logger.info("── Step 2/4  Extracting features")
    X, y_raw = build_feature_matrix(df)

    le = LabelEncoder()
    le.classes_ = np.array(CLASSES)  # pin order so class indices are deterministic
    y = le.transform(y_raw)
    return X, y, le


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    clf: ClassifierMixin,
) -> tuple[ClassifierMixin, np.ndarray, np.ndarray, float]:
    """
    Step 3 — split data and fit the classifier.

    Returns the fitted classifier, the held-out test split, and fit duration.
    """
    logger.info("── Step 3/4  Training  [%s]", clf.__class__.__name__)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("  Train: %d  ·  Test: %d", len(X_train), len(X_test))
    logger.info("  Fitting …")
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("  Done in %.1fs", elapsed)
    return clf, X_test, y_test, elapsed


def _evaluate(
    clf: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> None:
    """
    Step 4 — log classification report and save the confusion matrix.
    """
    logger.info("── Step 4/4  Evaluation")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    logger.info("Classification report:\n%s", report)

    cm      = confusion_matrix(y_test, y_pred)
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    _save_confusion_matrix(cm, list(le.classes_), str(cm_path))
    logger.info("Confusion matrix saved → %s", cm_path)


def _persist(clf: ClassifierMixin, le: LabelEncoder, model: str, elapsed: float) -> None:
    """Save the fitted classifier and label encoder to ``models/``.

    Filenames include the model name and a timestamp so successive runs
    never overwrite each other:  ``{model}_{YYYYMMDD_HHMMSS}.joblib``
    """
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    clf_path = MODELS_DIR / f"{model}_{ts}.joblib"
    joblib.dump(clf, clf_path)
    joblib.dump(le,  MODELS_DIR / "label_encoder.joblib")
    logger.info("Model saved → %s  (trained in %.1fs)", clf_path.name, elapsed)


def _save_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Orchestrator
def train(model: str = "random_forest", save: bool = True) -> ClassifierMixin:
    """Run the full pipeline: load → extract → fit → evaluate → (optionally) save.

    Args:
        model: Classifier name. See ``build_classifier`` for available options.
        save:  If True, persist the model and label encoder to ``models/``.

    Returns:
        The fitted classifier.
    """
    df = _load_data()
    X, y, le = _extract_features(df)
    clf, Xt, yt, elapsed = _fit(X, y, build_classifier(model))
    _evaluate(clf, Xt, yt, le)

    if save:
        _persist(clf, le, model, elapsed)

    return clf


# CLI entry point

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Train a beatbox classifier.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=list(CLASSIFIERS_CFG),
        help="Classifier to use (default: random_forest)",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Skip saving the model to disk",
    )
    args = parser.parse_args()
    train(model=args.model, save=args.save)
