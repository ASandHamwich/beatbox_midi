"""
Audio feature extraction.

Each audio clip is represented by a 28-dimensional feature vector:
    - MFCC mean (13) — timbral shape
    - MFCC std (13) — timbral variability
    - Spectral centroid mean (1) — brightness
    - Zero-crossing rate mean (1) — noisiness / percussiveness
"""

import logging

import numpy as np
import librosa
import pandas as pd

from training.config import SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT

logger = logging.getLogger(__name__)

N_FEATURES = N_MFCC * 2 + 2  # 28


def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Return a 28-d feature vector for a single audio clip.

    Args:
        audio: Float32 mono waveform.
        sr: Sample rate.

    Returns:
        1-D numpy array of length ``N_FEATURES`` (28).
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mfcc_mean = mfcc.mean(axis=1) # [N_MFCC]
    mfcc_std  = mfcc.std(axis=1) # [N_MFCC]

    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    ).mean()

    zcr = librosa.feature.zero_crossing_rate(
        y=audio, hop_length=HOP_LENGTH
    ).mean()

    return np.concatenate([mfcc_mean, mfcc_std, [centroid], [zcr]])


def build_feature_matrix(
    df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features for every row in *df*.

    Args:
        df: DataFrame with columns ``[path, label]``.
        verbose: Log progress every 500 samples at DEBUG level.

    Returns:
        X: Feature matrix of shape ``[n_samples, N_FEATURES]``.
        y: String label array of shape ``[n_samples]``.
    """
    n = len(df)
    logger.info("Extracting features for %d samples …", n)

    X, y = [], []
    for i, row in enumerate(df.itertuples(index=False), 1):
        if verbose and i % 500 == 0:
            logger.debug("  %d / %d samples processed", i, n)
        audio, sr = librosa.load(row.path, sr=SAMPLE_RATE, mono=True)
        X.append(extract_features(audio, sr=sr))
        y.append(row.label)

    X_arr = np.array(X, dtype=np.float32)
    logger.info("Feature extraction complete — matrix shape: %s", X_arr.shape)
    return X_arr, np.array(y)
