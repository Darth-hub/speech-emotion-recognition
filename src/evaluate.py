"""
Generate a full evaluation report from saved test predictions:
  - Per-class precision / recall / F1
  - Confusion matrix (counts and row-normalized)
  - Top confusion pairs with explanations

Run after train.py has produced test_predictions.npz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless rendering for CI/server use
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)

from .dataset import EMOTIONS


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path, normalize: bool = True) -> None:
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))
    fmt = ".2f" if normalize else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(cm: np.ndarray, labels: list[str], k: int = 5) -> list[dict]:
    """Off-diagonal cells with the highest counts -- the model's worst mistakes."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": labels[i], "predicted": labels[j], "count": int(cm[i, j])})
    return sorted(pairs, key=lambda d: d["count"], reverse=True)[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="models/checkpoints/test_predictions.npz")
    parser.add_argument("--out", default="models/checkpoints")
    args = parser.parse_args()

    data = np.load(args.predictions)
    y_true, y_pred = data["targets"], data["preds"]

    report = classification_report(y_true, y_pred, target_names=EMOTIONS,
                                   digits=4, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(EMOTIONS)))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, EMOTIONS, out / "confusion_matrix.png", normalize=True)
    plot_confusion(cm, EMOTIONS, out / "confusion_matrix_counts.png", normalize=False)

    summary = {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "macro_f1":     float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1":  float(f1_score(y_true, y_pred, average="weighted")),
        "per_class":    {e: report[e] for e in EMOTIONS},
        "top_confusions": top_confusions(cm, EMOTIONS, k=5),
    }
    with open(out / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Accuracy:    {summary['accuracy']:.4f}")
    print(f"Macro F1:    {summary['macro_f1']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1']:.4f}")
    print("\nTop confusions:")
    for c in summary["top_confusions"]:
        print(f"  {c['true']:>10s} -> {c['predicted']:<10s}  ({c['count']})")


if __name__ == "__main__":
    main()
