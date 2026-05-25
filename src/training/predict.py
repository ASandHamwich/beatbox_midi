"""Load a saved model and predict the class of new audio."""

import logging

import numpy as np
import joblib

from training.config import MODELS_DIR, SAMPLE_RATE
from training.features import extract_features

logger = logging.getLogger(__name__)


def load_model():
    """
    Return ``(classifier, label_encoder)`` loaded from ``models/``.
    """
    logger.debug("Loading model from %s", MODELS_DIR)

    clf = joblib.load(MODELS_DIR / "classifier.joblib")
    le  = joblib.load(MODELS_DIR / "label_encoder.joblib")

    logger.debug("Loaded %s with classes %s", clf.__class__.__name__, list(le.classes_))
    return clf, le


def predict(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """
    Predict the beatbox class of a raw audio waveform.

    Args:
        audio: Float32 mono waveform array.
        sr: Sample rate of the audio.

    Returns:
        Predicted class label (``"clap"``, ``"hihat"``, ``"kick"``, or ``"snare"``).
    """
    clf, le = load_model()
    features = extract_features(audio, sr=sr).reshape(1, -1)
    idx = clf.predict(features)[0]
    label = le.inverse_transform([idx])[0]
    
    logger.debug("predict() -> '%s'", label)
    return label


def predict_proba(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """
    Return class probabilities for a raw audio waveform.

    Returns:
        Dict mapping each class name to its predicted probability.
    """
    clf, le = load_model()
    features = extract_features(audio, sr=sr).reshape(1, -1)
    probs = clf.predict_proba(features)[0]
    result = dict(zip(le.classes_, probs))
    logger.debug("predict_proba() -> %s", {k: f"{v:.3f}" for k, v in result.items()})
    return result
