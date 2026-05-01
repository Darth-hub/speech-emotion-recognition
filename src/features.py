"""
Feature extraction.

We extract three complementary representations and stack them on the feature
axis so a 2D CNN can consume the result as a single multi-channel "image":

    log-Mel  (64 bands)  -> spectral envelope, perceptual frequency scale
    MFCCs    (40 coeffs) -> decorrelated envelope, compact, robust
    Chroma   (12 bins)   -> pitch-class energy, captures intonation contour

All three are computed with the SAME hop length so their time axes align,
giving a final tensor of shape (116, T) per clip.

Why three together? Each captures something the others miss:
- log-Mel is rich but high-dimensional and correlated.
- MFCCs throw away pitch but keep timbre cleanly.
- Chroma keeps pitch and throws away timbre.
The model picks what it needs from each stream.
"""

from __future__ import annotations

import numpy as np
import librosa

from .preprocessing import SAMPLE_RATE


N_FFT = 1024
HOP_LENGTH = 256          # 16 ms hop at 16 kHz -> ~187 frames over 3 s
N_MELS = 64
N_MFCC = 40
N_CHROMA = 12
FEATURE_DIM = N_MELS + N_MFCC + N_CHROMA  # 116


def log_mel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Log-amplitude Mel spectrogram. Shape: (N_MELS, T)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max)


def mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """MFCCs computed from the same Mel basis. Shape: (N_MFCC, T)."""
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )


def chroma(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Chroma-STFT pitch-class profile. Shape: (12, T)."""
    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Stack all three feature sets along the frequency axis. Shape: (116, T)."""
    feats = np.vstack([log_mel(y, sr), mfcc(y, sr), chroma(y, sr)])
    # Per-clip standardization stabilizes training across recording conditions.
    mean = feats.mean(axis=1, keepdims=True)
    std = feats.std(axis=1, keepdims=True) + 1e-6
    return ((feats - mean) / std).astype(np.float32)
