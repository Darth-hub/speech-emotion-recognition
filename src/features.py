import numpy as np
import librosa
from .preprocessing import SAMPLE_RATE

N_FFT = 1024
HOP_LENGTH = 256
N_MFCC = 40
FEATURE_DIM = N_MFCC  # only 40 now, not 116

def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract only MFCC features. Shape: (40, T)"""
    feats = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mean = feats.mean(axis=1, keepdims=True)
    std = feats.std(axis=1, keepdims=True) + 1e-6
    return ((feats - mean) / std).astype(np.float32)
