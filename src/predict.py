"""
Single-clip inference helper used by both the Streamlit app and the CLI.

Returns:
    label                : top predicted emotion string
    probs (dict)         : full posterior over all 7 classes
    attention (np.ndarray): per-frame attention weights, shape (T',)
                            -- aligned with the time axis of the spectrogram
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import EMOTIONS
from .features import extract_features
from .model import build_model
from .preprocessing import preprocess


class Predictor:
    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(n_classes=len(EMOTIONS)).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.labels = state.get("labels", EMOTIONS)

    @torch.no_grad()
    def predict(self, audio_path: str) -> dict:
        y = preprocess(audio_path)
        feats = extract_features(y)                              # (116, T)
        x = torch.from_numpy(feats).unsqueeze(0).to(self.device) # (1, 116, T)

        # Run the network manually to capture attention weights for the UI.
        x_in = x.unsqueeze(1)                                    # (1, 1, 116, T)
        h = self.model.cnn(x_in)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)
        h, _ = self.model.lstm(h)
        attn = F.softmax(self.model.attn.score(h), dim=1).squeeze(-1).squeeze(0).cpu().numpy()
        pooled = (h * F.softmax(self.model.attn.score(h), dim=1)).sum(dim=1)
        logits = self.model.classifier(pooled)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return {
            "label": self.labels[int(probs.argmax())],
            "probs": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
            "attention": attn,
            "features": feats,
        }
