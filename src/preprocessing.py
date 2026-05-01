"""
Audio preprocessing and augmentation.

Design choices (and why):
- 16 kHz mono: speech information is sub-8kHz; 16 kHz is the production standard.
- Peak normalization: remove recording-loudness as a confound.
- Silence trim: avoid wasting model capacity on non-discriminative regions.
- Fixed 3-second duration: CNN inputs must be fixed-shape; 3 s captures prosody
  while keeping memory reasonable.
- On-the-fly augmentation: each epoch sees a different perturbed version of the
  clip, which acts as a strong regularizer without inflating disk usage.
"""

from __future__ import annotations

import numpy as np
import librosa


SAMPLE_RATE = 16_000
DURATION = 3.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at the target sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def trim_silence(y: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading/trailing silence below `top_db` dB relative to peak."""
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    # Guard against effects.trim returning an empty array on near-silent clips.
    return yt if yt.size > 0 else y


def peak_normalize(y: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Scale so the loudest sample sits at `target_dbfs` (default -1 dBFS)."""
    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        return y
    target_amp = 10.0 ** (target_dbfs / 20.0)
    return (y / peak) * target_amp


def fix_length(y: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Right-pad with zeros if too short; center-crop if too long."""
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n]
    out = np.zeros(n, dtype=y.dtype)
    out[: len(y)] = y
    return out


def preprocess(path: str) -> np.ndarray:
    """Full clean-pipeline: load -> trim -> normalize -> fix length."""
    y = load_audio(path)
    y = trim_silence(y)
    y = peak_normalize(y)
    y = fix_length(y)
    return y


# ---------------------------------------------------------------------------
# Augmentations (called from the Dataset's __getitem__ during training only)
# ---------------------------------------------------------------------------

def add_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix Gaussian noise at the requested SNR. Lower SNR = noisier."""
    sig_power = np.mean(y ** 2) + 1e-12
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(y.dtype)
    return y + noise


def pitch_shift(y: np.ndarray, n_steps: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Shift pitch by `n_steps` semitones (positive = up, negative = down)."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(y.dtype)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Speed up (rate>1) or slow down (rate<1). Re-fixes length to N_SAMPLES."""
    stretched = librosa.effects.time_stretch(y, rate=rate).astype(y.dtype)
    return fix_length(stretched)


def random_augment(y: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply each augmentation independently with probability `p`.
    SNR sampled uniformly in [10, 30] dB so the model sees both light and heavy noise."""
    if np.random.rand() < p:
        y = add_noise(y, snr_db=float(np.random.uniform(10.0, 30.0)))
    if np.random.rand() < p:
        y = pitch_shift(y, n_steps=float(np.random.uniform(-2.0, 2.0)))
    if np.random.rand() < p:
        y = time_stretch(y, rate=float(np.random.uniform(0.9, 1.1)))
    return y
