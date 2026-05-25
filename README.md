# beatbox-midi

Classify beatbox sounds and (eventually) generate MIDI from them.

Built with Python + uv, trained on the [`maxardito/beatbox`](https://huggingface.co/datasets/maxardito/beatbox) HuggingFace dataset.

## Project layout

```
beatbox_midi/
├── conf/                   # YAML configuration
│   ├── audio.yaml          # sample rate, MFCC settings
│   ├── classifiers.yaml    # classifier registry & hyperparameters
│   ├── data.yaml           # dataset name, class list
│   └── model.yaml          # random state, train/test split
├── notebooks/
│   └── eda.ipynb        # exploratory data analysis
├── scripts/
│   └── train_all.sh        # train all three models in sequence
├── src/
│   ├── training/
│   │   ├── config.py       # YAML loader; re-exports all settings
│   │   ├── data.py         # dataset download & loading
│   │   ├── features.py     # 28-d MFCC feature extraction
│   │   ├── train.py        # training pipeline
│   │   └── predict.py      # inference on new audio
│   └── utils/
│       └── logger.py       # shared logging setup
├── models/                 # saved classifiers (gitignored)
├── reports/                # confusion matrix plots (gitignored)
└── logs/                   # per-run log files (gitignored)
```

## Quickstart

```bash
# Install dependencies
uv sync

# Train a single model (default: random_forest)
uv run python -m training.train

# Train with a specific classifier
uv run python -m training.train --model svm
uv run python -m training.train --model gradient_boosting

# Train all three models
bash scripts/train_all.sh
```

## Features

Each audio sample is represented as a **28-dimensional feature vector**:

| Feature | Dimensions |
|---|---|
| MFCC mean | 13 |
| MFCC std | 13 |
| Spectral centroid mean | 1 |
| Zero-crossing rate mean | 1 |

Audio settings (22 050 Hz sample rate, 13 MFCCs, 512 hop length, 2048 FFT window) are defined in `conf/audio.yaml`.

## Results

Evaluated on a stratified 80/20 train/test split of 5 058 samples across 4 classes: **clap**, **hihat**, **kick**, **snare**.

### Random Forest

| Class | Precision | Recall | F1 |
|---|---|---|---|
| clap | 0.99 | 1.00 | 1.00 |
| hihat | 1.00 | 1.00 | 1.00 |
| kick | 1.00 | 1.00 | 1.00 |
| snare | 1.00 | 1.00 | 1.00 |
| **accuracy** | | | **1.00** |

Training time: **0.4 s**

### SVM (RBF kernel)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| clap | 0.49 | 0.98 | 0.66 |
| hihat | 0.98 | 0.87 | 0.92 |
| kick | 0.96 | 0.84 | 0.89 |
| snare | 0.77 | 0.64 | 0.70 |
| **accuracy** | | | **0.81** |

Training time: **0.9 s**

### Gradient Boosting (HistGradientBoosting)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| clap | 1.00 | 1.00 | 1.00 |
| hihat | 1.00 | 1.00 | 1.00 |
| kick | 1.00 | 1.00 | 1.00 |
| snare | 1.00 | 1.00 | 1.00 |
| **accuracy** | | | **1.00** |

Training time: **3.0 s**

### Summary

| Model | Accuracy | Train time |
|---|---|---|
| Random Forest | **1.00** | 0.4 s |
| SVM | 0.81 | 0.9 s |
| Gradient Boosting | **1.00** | 3.0 s |

Random Forest and Gradient Boosting both achieve perfect classification on the held-out test set. SVM struggles with **clap** (F1 0.66), likely due to its spectral overlap with snare; the default RBF kernel without feature scaling is a probable cause. Random Forest is the preferred default — same accuracy as Gradient Boosting at 7× less training time.

---

## Roadmap

### 1 · Expanding the dataset

The current dataset covers four core drum sounds. Real-world beatboxing is considerably richer. Recommended directions:

**More sound classes**
- Extended percussives: rim shot, tom, open hihat, cymbal crash, shaker
- Vocal-specific sounds: bass drum with mouth closed ("bm"), throat bass, inward sounds, lip rolls
- Hybrid/layered sounds — a common beatbox technique where two sounds are combined (e.g. kick + hihat simultaneously)

**More data per class**
- The current split is imbalanced (hihat/kick ~310 vs clap ~122). Collecting or augmenting to a balanced 300+ per class per new sound would keep training stable.
- Augmentation strategies: pitch shift (±2 semitones), time stretch (0.9–1.1×), additive white/pink noise, room impulse response convolution. All can be done with `librosa` and `soundfile` without needing new recordings.

**Additional datasets to consider**
- BaDumTss Dataset: A multi-task learning dataset that provides isolated beatbox samples and full polyphonic tracks with per-instrument labels like kick, snare, and hi-hats.
- Queen Mary University (QMUL) Beatbox Dataset: A seminal corpus in the field where experienced beatboxers recorded files that were manually annotated with time-aligned utterances and standard drum sounds.
- Amateur Vocal Percussion (AVP) Dataset: Contains over 9,000 utterances from participants with varying levels of experience, specifically annotated with human vocalizations mapped to kick drums, snare drums, and hi-hats.
- Self-recorded data via `sounddevice` or a simple recording script — keeps the class distribution under control and allows artist-specific fine-tuning

---

### 2 · Improving model performance

The perfect scores on RF and Gradient Boosting are encouraging but warrant scepticism — they likely reflect how acoustically clean and homogeneous the current dataset is. Robustness will drop when the model meets real-world input (background noise, microphone variation, room acoustics). Areas to address:

**Feature engineering**
- Add chroma features and tonnetz for tonal sounds
- Add `delta` and `delta-delta` MFCCs to capture temporal dynamics within a clip
- Experiment with mel-spectrogram patches as input (opens the door to CNNs)
- Normalise features with `StandardScaler` before SVM — this alone should bring SVM accuracy up significantly

**Model improvements**
- `StandardScaler` pipeline wrapper: `Pipeline([("scaler", StandardScaler()), ("clf", SVC(...))])` — required for SVM, harmless for tree models
- Cross-validation: replace single train/test split with `StratifiedKFold(n_splits=5)` to get more reliable accuracy estimates
- Hyperparameter search: `RandomizedSearchCV` or `Optuna` over the YAML-defined param spaces
- Consider a lightweight CNN on log-mel spectrograms (e.g. a 3-layer conv net via PyTorch) — likely the biggest accuracy leap once data volume increases

**Evaluation rigour**
- Per-class confusion analysis across multiple random seeds
- Test on held-out *recordings* (not just held-out samples from the same clips) to measure generalisation
- Latency profiling of the predict path — important for real-time use

---

### 3 · MIDI generation

The classifier is the first building block. The end goal is mapping beatbox audio to MIDI events in two modes:

#### Non-real-time (file to MIDI)

The offline pipeline: record or load a full beatbox performance, run the classifier frame-by-frame, and emit a `.mid` file.

```
audio file → onset detection → segment → classify → MIDI note events → .mid
```

- **Onset detection**: `librosa.onset.onset_detect` with `backtrack=True` to snap onsets to the nearest energy trough — gives cleaner segment boundaries than peak-picking alone
- **Segmentation**: slice audio at detected onsets with a fixed window (e.g. 100 ms) and run `extract_features` on each slice
- **MIDI mapping**: assign each class a General MIDI percussion note (kick → 36, snare → 38, hihat → 42, clap → 39) and write events with `midiutil` or `pretty_midi`
- **Quantisation**: snap event times to the nearest 16th note grid given a user-supplied BPM — keeps output musically usable


#### Real-time (microphone to live MIDI)

Continuous microphone input, classified on the fly, MIDI sent to a DAW or synthesiser.

```
mic → ring buffer → onset detection → classify → MIDI output → DAW
```

- **Audio I/O**: `sounddevice` with a callback-based stream (`sd.InputStream`) — low-latency, cross-platform
- **Ring buffer**: keep a rolling ~200 ms window; on detected onset, extract the preceding 100 ms and classify
- **MIDI output**: `python-rtmidi` to open a virtual MIDI port that DAWs can subscribe to
- **Latency budget**: feature extraction + RF inference runs in <5 ms on a modern laptop; the bottleneck is the audio buffer size (typically 10–20 ms at 512 samples / 22 050 Hz)

#### DAW integration

| Approach | How |
|---|---|
| **Virtual MIDI port** (macOS/Linux) | `python-rtmidi` opens an IAC Driver bus; Ableton/Logic subscribes to it as a MIDI input |
| **Virtual MIDI port** (Windows) | Requires loopMIDI (free third-party driver) |
| **VST/AU plugin** | Long-term: wrap the model in a C++ plugin shell via JUCE, embedding the Python model via `onnxruntime` (export with `sklearn-onnx`) |
| **Max/MSP or Pure Data patch** | Call the Python predict script via `shell` object or use OSC bridge (`python-osc`) |
| **Ableton Live + Max for Live** | M4L device running `subprocess` or a local socket server that calls `predict.py` |

The shortest path to a working DAW demo is the virtual MIDI port approach — no plugin required, works with any DAW that accepts external MIDI, and the entire pipeline fits in a single Python process.
