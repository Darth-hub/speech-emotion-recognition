"""
PyTorch Dataset that reads the manifest CSV and returns (features, label).

Features are extracted on-the-fly from preprocessed audio. Augmentation is
applied only when `train=True`, with SNR sampled from a curriculum range that
widens as training progresses (see set_epoch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    add_noise, fix_length, peak_normalize, pitch_shift,
    preprocess as clean_preprocess, time_stretch,
)
from .features import extract_features


class SERDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        train: bool = False,
        aug_prob: float = 0.5,
        max_epochs: int = 50,
    ):
        self.df = manifest.reset_index(drop=True)
        self.train = train
        self.aug_prob = aug_prob
        self.max_epochs = max_epochs
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Called once per epoch by the trainer to advance the curriculum."""
        self._epoch = epoch

    def _curriculum_snr(self) -> float:
        """SNR window widens from [25, 30] dB at epoch 0 to [5, 30] dB at the end.
        Lower SNR = harder. This is easy-to-hard curriculum learning."""
        progress = min(1.0, self._epoch / max(1, self.max_epochs - 1))
        snr_low = 25.0 - 20.0 * progress    # 25 -> 5
        return float(np.random.uniform(snr_low, 30.0))

    def _augment(self, y: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.aug_prob:
            y = add_noise(y, snr_db=self._curriculum_snr())
        if np.random.rand() < self.aug_prob:
            y = pitch_shift(y, n_steps=float(np.random.uniform(-2.0, 2.0)))
        if np.random.rand() < self.aug_prob:
            y = time_stretch(y, rate=float(np.random.uniform(0.9, 1.1)))
        # Re-normalize after augmentation so noise/stretching don't change loudness.
        return fix_length(peak_normalize(y))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        y = clean_preprocess(row["filepath"])
        if self.train:
            y = self._augment(y)
        feats = extract_features(y)
        return torch.from_numpy(feats), torch.tensor(int(row["label_id"]), dtype=torch.long)
