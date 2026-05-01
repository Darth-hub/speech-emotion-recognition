"""Smoke tests. Run with `pytest -q`."""

import numpy as np
import torch

from src.features import extract_features, FEATURE_DIM
from src.model import build_model
from src.preprocessing import (
    N_SAMPLES, add_noise, fix_length, peak_normalize, pitch_shift, time_stretch,
)


def _fake_audio(n: int = N_SAMPLES) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, size=n).astype(np.float32)


def test_fix_length_pads_and_crops():
    short = np.zeros(1000, dtype=np.float32)
    assert fix_length(short).shape == (N_SAMPLES,)
    long = np.zeros(N_SAMPLES * 2, dtype=np.float32)
    assert fix_length(long).shape == (N_SAMPLES,)


def test_peak_normalize_to_target():
    y = _fake_audio() * 0.01
    out = peak_normalize(y, target_dbfs=-1.0)
    assert abs(np.max(np.abs(out)) - 10 ** (-1 / 20)) < 1e-3


def test_augmentations_preserve_length():
    y = _fake_audio()
    assert add_noise(y, snr_db=20).shape == y.shape
    assert pitch_shift(y, n_steps=1).shape == y.shape
    assert time_stretch(y, rate=1.1).shape == y.shape  # fix_length re-applied


def test_feature_shape():
    feats = extract_features(_fake_audio())
    assert feats.shape[0] == FEATURE_DIM
    assert feats.dtype == np.float32


def test_model_forward_shape():
    model = build_model(n_classes=7).eval()
    x = torch.randn(2, FEATURE_DIM, 187)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 7)
