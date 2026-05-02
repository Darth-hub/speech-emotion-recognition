"""
CNN + BiLSTM + Attention model for Speech Emotion Recognition.

Pipeline:
    (B, 1, 116, T)  ---CNN--->  (B, C, F', T')  ---reshape--->  (B, T', C*F')
                    --BiLSTM->  (B, T', 2H)     --attention-->  (B, 2H)
                    --MLP----> logits (B, 7)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """3x3 conv -> BN -> ReLU -> 2x2 max-pool. Standard CNN building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class AttentionPool(nn.Module):
    """Additive attention pooling over the time axis.

    Given (B, T, D) it learns scalar weights alpha_t that sum to 1 over T,
    then returns the weighted mean of shape (B, D). This lets the model focus
    on emotionally salient frames (stressed syllables, sighs, bursts).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.score(x), dim=1)   # (B, T, 1)
        return (weights * x).sum(dim=1)             # (B, D)


class SERModel(nn.Module):
    """CNN front-end -> BiLSTM -> attention pool -> MLP classifier."""

    def __init__(
        self,
        n_classes: int = 7,
        n_freq: int = 40,        # log-Mel(64) + MFCC(40) + chroma(12)
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.Dropout2d(dropout),
        )

        # After three 2x2 pools the frequency axis shrinks by 8.
        cnn_freq = n_freq // 8
        self.lstm_input = 128 * cnn_freq

        self.lstm = nn.LSTM(
            input_size=self.lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attn = AttentionPool(2 * lstm_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 116, T) -- add the channel dim for Conv2d
        if x.dim() == 3:
            x = x.unsqueeze(1)               # (B, 1, 116, T)

        x = self.cnn(x)                      # (B, 128, F', T')
        b, c, f, t = x.shape

        # Move time to the sequence axis: (B, T', C*F')
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)

        x, _ = self.lstm(x)                  # (B, T', 2H)
        x = self.attn(x)                     # (B, 2H)
        return self.classifier(x)            # (B, n_classes)


def build_model(n_classes: int = 7) -> SERModel:
    return SERModel(n_classes=n_classes)


if __name__ == "__main__":
    # Sanity check on a synthetic batch.
    m = build_model()
    dummy = torch.randn(4, 116, 187)
    out = m(dummy)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"output shape: {tuple(out.shape)}  |  trainable params: {n_params:,}")
