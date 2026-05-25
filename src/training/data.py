"""Load the beatbox dataset from Hugging Face."""

import gzip
import io
import logging
import tarfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from training.config import CACHE_DIR, HF_DATASET

logger = logging.getLogger(__name__)


# Helpers

def _extract_audio(split: str) -> Path:
    """
    Download and extract the audio tarball for *split* into CACHE_DIR.

    Uses a sentinel file so extraction only runs once.
    Returns the directory containing the extracted WAV files.
    """
    audio_dir = CACHE_DIR / split
    sentinel  = audio_dir / "_done"

    if sentinel.exists():
        logger.debug("Audio cache hit — skipping download (%s)", audio_dir)
        return audio_dir

    logger.info("Downloading %s audio from Hugging Face (first run only) …", split)
    tar_path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=f"dataset/audio_{split}.tar.gz",
        repo_type="dataset",
    )
    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Extracting tarball → %s", audio_dir)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(audio_dir)

    sentinel.touch()
    logger.info("Extraction complete: %s", audio_dir)
    return audio_dir


def _load_metadata(split: str, audio_dir: Path) -> pd.DataFrame:
    """
    Download and parse the metadata CSV for *split*.

    Returns a DataFrame with columns ``[path, label]`` where ``path`` is
    an absolute path to the extracted WAV file.
    """
    logger.debug("Fetching metadata CSV for split '%s'", split)
    csv_path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=f"dataset/metadata_{split}.csv.gz",
        repo_type="dataset",
    )
    with open(csv_path, "rb") as f:
        raw = gzip.decompress(f.read())
    df = pd.read_csv(io.BytesIO(raw))
    df = df.rename(columns={"class": "label"})

    wav_map   = {p.name: p for p in audio_dir.rglob("*.wav")}
    df["path"] = df["path"].apply(lambda p: str(wav_map.get(Path(p).name, "")))
    df = df[df["path"] != ""].reset_index(drop=True)

    logger.debug("Metadata loaded: %d rows, %d matched WAV files", len(df), len(wav_map))
    return df[["path", "label"]]


# Loading from API

def load_beatbox(split: str = "train") -> pd.DataFrame:
    """
    Return a DataFrame with columns ``[path, label]`` for *split*.

    Audio is extracted to ``CACHE_DIR`` on the first call and reused
    on all subsequent calls.

    Args:
        split: Dataset split — currently only ``"train"`` has data.

    Returns:
        pandas DataFrame with one row per audio clip.
    """
    audio_dir = _extract_audio(split)
    return _load_metadata(split, audio_dir)


def describe(df: pd.DataFrame) -> None:
    """
    Log a quick summary of a loaded dataset DataFrame.
    """
    counts = df["label"].value_counts().sort_index()
    lines  = [
        f"Rows   : {len(df):,}",
        f"Labels : {sorted(df['label'].unique())}",
        "Class distribution:",
    ]
    for cls, n in counts.items():
        lines.append(f"  {cls:<8} {n:>5}  ({n / len(df):.1%})")
    logger.info("\n".join(lines))
