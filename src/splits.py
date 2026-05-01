"""
Speaker-disjoint train/val/test splitting and leave-one-corpus-out helper.

Why speaker-disjoint?
    If two clips from the same speaker appear in both train and test, the model
    learns to recognize the *speaker* and inflate accuracy by 10-20 absolute
    points. A speaker-disjoint split is the honest benchmark.

Why leave-one-corpus-out (LOCO)?
    Even speaker-disjoint splits within a corpus share recording conditions
    (mic, room, post-processing). LOCO trains on 3 corpora and tests on the
    4th, which is the closest proxy we have for true real-world generalization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def speaker_disjoint_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split so no speaker_id appears in more than one fold."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["speaker_id"]))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # Re-scale val_size relative to the remaining train_val pool.
    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val, groups=train_val["speaker_id"]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    # Sanity: speaker disjointness across all three folds.
    assert not (set(train["speaker_id"]) & set(val["speaker_id"]))
    assert not (set(train["speaker_id"]) & set(test["speaker_id"]))
    assert not (set(val["speaker_id"]) & set(test["speaker_id"]))
    return train, val, test


def leave_one_corpus_out(
    df: pd.DataFrame, held_out: str, val_frac: float = 0.1, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on every corpus except `held_out`; test on `held_out`. Validation
    is taken from the training corpora with a speaker-disjoint sub-split."""
    test = df[df["dataset"] == held_out].reset_index(drop=True)
    rest = df[df["dataset"] != held_out].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(gss.split(rest, groups=rest["speaker_id"]))
    return (rest.iloc[train_idx].reset_index(drop=True),
            rest.iloc[val_idx].reset_index(drop=True),
            test)


def class_weights(df: pd.DataFrame, n_classes: int = 7) -> np.ndarray:
    """Inverse-frequency class weights (normalized) to combat imbalance."""
    counts = df["label_id"].value_counts().reindex(range(n_classes), fill_value=0).values
    weights = 1.0 / np.maximum(counts, 1)
    return (weights / weights.sum() * n_classes).astype(np.float32)
