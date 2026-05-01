"""
Training loop with:
- AdamW + OneCycle LR schedule (warmup + cosine decay)
- Class-weighted cross-entropy (imbalance) + label smoothing 0.05 (calibration)
- Mixed precision on CUDA
- Early stopping on validation macro-F1
- Best-model checkpointing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from .data_loader import SERDataset
from .dataset import EMOTIONS
from .model import build_model
from .splits import class_weights, leave_one_corpus_out, speaker_disjoint_split


def make_loaders(train_df, val_df, test_df, batch_size, num_workers, max_epochs):
    train_ds = SERDataset(train_df, train=True, max_epochs=max_epochs)
    val_ds   = SERDataset(val_df, train=False)
    test_ds  = SERDataset(test_df, train=False)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **common),
        DataLoader(val_ds,   shuffle=False, **common),
        DataLoader(test_ds,  shuffle=False, **common),
        train_ds,
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    return {
        "loss": float(np.mean(losses)),
        "acc":  float(accuracy_score(targets, preds)),
        "f1":   float(f1_score(targets, preds, average="macro")),
        "preds": preds, "targets": targets,
    }


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = pd.read_csv(args.manifest)
    if args.loco:
        train_df, val_df, test_df = leave_one_corpus_out(df, held_out=args.loco)
        print(f"[LOCO] held out: {args.loco}  test clips: {len(test_df)}")
    else:
        train_df, val_df, test_df = speaker_disjoint_split(df)
    print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    train_loader, val_loader, test_loader, train_ds = make_loaders(
        train_df, val_df, test_df, args.batch_size, args.num_workers, args.epochs
    )

    model = build_model(n_classes=len(EMOTIONS)).to(device)
    weights = torch.from_numpy(class_weights(train_df)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, patience, history = 0.0, 0, []
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out) / "best.pt"

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)             # advance curriculum SNR
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        val = evaluate(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1"]})
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  val_acc={val['acc']:.4f}  val_f1={val['f1']:.4f}")

        if val["f1"] > best_f1:
            best_f1, patience = val["f1"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_f1": val["f1"], "labels": EMOTIONS}, ckpt_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stopping (no val_f1 improvement for {args.patience} epochs)")
                break

    # Reload best weights and evaluate on the held-out test set.
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device, criterion)
    print(f"\nTEST  acc={test_metrics['acc']:.4f}  f1={test_metrics['f1']:.4f}")

    # Persist everything needed for downstream evaluation/analysis.
    np.savez(Path(args.out) / "test_predictions.npz",
             preds=test_metrics["preds"], targets=test_metrics["targets"])
    with open(Path(args.out) / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/manifest.csv")
    p.add_argument("--out", default="models/checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--loco", default=None,
                   help="Hold out one corpus for LOCO eval: ravdess|tess|savee|cremad")
    train(p.parse_args())


if __name__ == "__main__":
    main()
