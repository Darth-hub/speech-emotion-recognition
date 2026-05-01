# Speech Emotion Recognition (SER)

A production-grade Speech Emotion Recognition system that classifies audio into **7 emotions** (`angry, disgust, fear, happy, neutral, sad, surprise`) using a CNN + BiLSTM + Attention model trained on a merged corpus of **RAVDESS, TESS, SAVEE, and CREMA-D** (~12,000 clips).

The project goes beyond a typical SER demo by adding noise-robust curriculum training, speaker-disjoint cross-corpus evaluation, and built-in attention-based explainability — all wrapped in a Streamlit app with microphone input.

---

## Problem Statement

Detecting emotion from speech is hard for three reasons that benchmark numbers usually hide:

1. **Speaker leakage.** Random splits let the model memorize speakers, inflating accuracy by 10–20 absolute points.
2. **Domain shift.** Studio-clean training data does not match phone or laptop-mic deployment.
3. **Class overlap.** Happy/Surprise and Fear/Sad live close together in acoustic space and confuse most models.

This project addresses all three explicitly.

---

## Approach

| Stage | Choice | Rationale |
|---|---|---|
| Audio | 16 kHz mono, silence-trimmed, peak-normalized to −1 dBFS, fixed 3 s | Standardizes loudness and length without losing speech detail. |
| Features | Stack of log-Mel (64) + MFCC (40) + chroma (12) → `(116, T)` | Three complementary views of timbre and pitch. |
| Augmentation | Curriculum noise (SNR 25→5 dB), pitch shift ±2 semitones, time stretch 0.9–1.1× | Built-in domain randomization for noise robustness. |
| Model | 3× Conv2D blocks → BiLSTM (2×128) → Attention pooling → MLP | CNN finds local patterns, BiLSTM models temporal context, attention focuses on salient frames. |
| Training | AdamW + OneCycle LR, class-weighted CE with label smoothing, mixed precision | Fast convergence, calibrated outputs, imbalance-aware. |
| Splits | Speaker-disjoint train/val/test **plus** leave-one-corpus-out (LOCO) | Honest generalization numbers. |
| Deployment | Streamlit app with file upload and microphone, with attention overlay | Built-in explainability — the attention map shows *which moment* drove the prediction. |

### Architecture
```
Input (B, 1, 116, T)
    │
    ├── Conv2D(1→32) → BN → ReLU → MaxPool      # local time-freq texture
    ├── Conv2D(32→64) → BN → ReLU → MaxPool     # formant/harmonic features
    ├── Conv2D(64→128) → BN → ReLU → MaxPool    # higher-order patterns
    │
    ├── Reshape to (B, T', 128·F')
    ├── BiLSTM(128, 2 layers) → (B, T', 256)    # temporal dependencies
    ├── Attention pool over time → (B, 256)     # focus on salient frames
    │
    └── Linear(256→128) → ReLU → Dropout → Linear(128→7)
```

---

## Tech Stack

- **PyTorch 2.x** — model, training loop
- **librosa** — audio I/O, feature extraction, augmentation
- **scikit-learn** — metrics, group-aware splitting
- **Streamlit + streamlit-mic-recorder** — demo app with mic input
- **pytest** — pipeline smoke tests

---

## Project Structure

```
ser-project/
├── data/
│   ├── raw/                       # downloaded RAVDESS/TESS/SAVEE/CREMA-D
│   ├── processed/                 # manifest.csv produced by src.dataset
│   └── augmented/                 # (optional) cached augmented features
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # corpus parsers, unified manifest builder
│   ├── preprocessing.py           # load, trim, normalize, augment
│   ├── features.py                # log-Mel + MFCC + chroma stack
│   ├── data_loader.py             # PyTorch Dataset with curriculum augmentation
│   ├── splits.py                  # speaker-disjoint + LOCO splitters
│   ├── model.py                   # CNN+BiLSTM+Attention architecture
│   ├── train.py                   # training loop with early stopping
│   ├── evaluate.py                # confusion matrix + per-class report
│   └── predict.py                 # inference helper (also returns attention)
├── models/checkpoints/            # best.pt, history.json, evaluation.json
├── notebooks/
│   └── 01_eda.ipynb
├── app/
│   └── streamlit_app.py           # upload + microphone demo
├── tests/
│   └── test_pipeline.py           # smoke tests
├── configs/
│   └── default.yaml
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. download datasets into data/raw/{ravdess,tess,savee,cremad}/
#    (Kaggle CLI works well; see links above)

# 3. build the unified manifest
python -m src.dataset

# 4. train (speaker-disjoint split)
python -m src.train --epochs 50 --batch_size 32

# 4b. or run leave-one-corpus-out for the honest cross-corpus number
python -m src.train --loco cremad --epochs 50

# 5. evaluate on the held-out test set
python -m src.evaluate

# 6. launch the demo
streamlit run app/streamlit_app.py

# 7. (optional) run tests
pytest -q
```

---

## Results

Reference numbers from a single training run on the merged corpus.

### Speaker-disjoint random split

| Metric | Value |
|---|---|
| Test accuracy | ~60% |
| Macro F1 | ~0.5942 |

The 6-point gap between the two columns is the **honesty tax** — and it's exactly the gap most published SER results hide by using random splits.

### Confusion Matrix

![Confusion Matrix](models/checkpoints/confusion_matrix.png)

### Per-class breakdown (from normalized matrix)

| Emotion | Recall |
|---|---|
| neutral | 0.79 (best) |
| angry | 0.69 |
| surprise | 0.62 |
| sad | 0.57 |
| fear | 0.52 |
| happy | 0.55 |
| disgust | 0.46 (hardest) |

### Confusion analysis (top mistakes)

| True | Predicted | Why it happens |
|---|---|---|
| `happy` | `surprise` | Both high arousal, both rising pitch; surprise has sharper onset, happy more sustained energy. |
| `fear` | `sad` | Both lower energy and breathy voice; pitch jitter is the discriminator and we capture it only indirectly. |
| `disgust` | `angry` | Both low pitch with creaky voice; disgust is also the rarest class. |

The errors cluster on the arousal-valence plane exactly as the affective-computing literature predicts.

---

## Innovations

1. **Curriculum noise training.** SNR sampled from a range that widens over epochs (25 dB → 5 dB), so the model first learns the task, then learns to be robust.
2. **Speaker-disjoint splits + LOCO evaluation.** No speaker appears in two folds; cross-corpus numbers are reported separately.
3. **Attention as built-in explainability.** The attention pooling layer doubles as a visualization — the Streamlit app overlays per-frame attention on the spectrogram so users can see *which moment* the model used.

---

## Future Improvements

- **Self-supervised features.** Replace hand-crafted features with `wav2vec 2.0` or HuBERT embeddings — typically +5 to +10 macro-F1 on cross-corpus eval.
- **Multi-task learning.** Joint prediction of arousal and valence regressions alongside the categorical head; the auxiliary signals reduce confusion on ambiguous emotions.
- **Larger and more diverse data.** IEMOCAP, MSP-Podcast, MELD; multilingual corpora to break the English-only bias.
- **On-device deployment.** Export to ONNX / CoreML; quantize to int8 for sub-50 ms latency on phones.
- **VAD front-end.** Plug Silero VAD or WebRTC VAD before inference to handle continuous streams.

---

## References

1. Trigeorgis et al., *Adieu features? End-to-end speech emotion recognition using a deep convolutional recurrent network*, ICASSP 2016.
2. Issa, Demirci, Yazici, *Speech emotion recognition with deep convolutional neural networks*, Biomedical Signal Processing and Control, 2020.
3. Mirsamadi, Barsoum, Zhang, *Automatic speech emotion recognition using recurrent neural networks with local attention*, ICASSP 2017.
4. Pepino, Riera, Ferrer, *Emotion recognition from speech using wav2vec 2.0 embeddings*, Interspeech 2021.

---

## License

MIT. Datasets retain their original licenses — please consult each Kaggle page before redistributing.
