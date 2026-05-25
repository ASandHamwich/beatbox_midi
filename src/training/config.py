"""Load project configuration from conf/*.yaml.

All module-level names are identical to the old hard-coded config, so every existing import continues to work without modification.

Source of truth:
    conf/data.yaml — dataset name and class list
    conf/audio.yaml — sample rate and MFCC extraction parameters
    conf/model.yaml — random seed and train/test split ratio
"""

from pathlib import Path
import yaml

# Paths (derived at runtime) 
ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "conf"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / ".cache" / "beatbox"

for _d in (MODELS_DIR, REPORTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── YAML loader ───────────────────────────────────────────────────────────────
def _load(filename: str) -> dict:
    with open(CONF_DIR / filename) as f:
        return yaml.safe_load(f)


# ── Dataset ───────────────────────────────────────────────────────────────────
_data = _load("data.yaml")

HF_DATASET: str = _data["hf_dataset"]
CLASSES: list[str] = _data["classes"]

# ── Audio ─────────────────────────────────────────────────────────────────────
_audio = _load("audio.yaml")

SAMPLE_RATE: int = _audio["sample_rate"]
N_MFCC: int = _audio["n_mfcc"]
HOP_LENGTH: int = _audio["hop_length"]
N_FFT: int = _audio["n_fft"]

# ── Model ─────────────────────────────────────────────────────────────────────
_model = _load("model.yaml")

RANDOM_STATE: int = _model["random_state"]
TEST_SIZE: float = _model["test_size"]

# ── Classifiers ──────────────────────────────────────────────────────────────
CLASSIFIERS_CFG: dict = _load("classifiers.yaml")
