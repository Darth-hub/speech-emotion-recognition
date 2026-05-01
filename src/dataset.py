"""
Dataset loader for the merged SER corpus (RAVDESS + TESS + SAVEE + CREMA-D).

Produces a single `manifest.csv` with columns:
    filepath, dataset, speaker_id, emotion, gender

Each corpus encodes labels differently in its filenames; we normalize them
to a shared 7-class label space:
    {angry, disgust, fear, happy, neutral, sad, surprise}

`calm` (RAVDESS only) is merged into `neutral` because it represents low-arousal
non-emotional speech and would otherwise be a single-corpus class.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

# Unified label vocabulary used across the project.
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}

# RAVDESS encodes emotion as the 3rd dash-separated token in the filename.
# Format: 03-01-EE-IN-ST-RE-AC.wav  (channel-vocal-emotion-intensity-stmt-rep-actor)
RAVDESS_EMO = {
    "01": "neutral", "02": "neutral",  # 02 = "calm" -> neutral
    "03": "happy",   "04": "sad",
    "05": "angry",   "06": "fear",
    "07": "disgust", "08": "surprise",
}

# CREMA-D format: 1001_DFA_ANG_XX.wav  (actor_sentence_emotion_intensity)
CREMAD_EMO = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

# SAVEE format: <speaker>_<emotion-code><digits>.wav  e.g. DC_a01.wav, JE_su03.wav
SAVEE_EMO = {
    "a": "angry", "d": "disgust", "f": "fear",
    "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise",
}

# TESS format: OAF_word_emotion.wav or YAF_word_emotion.wav
TESS_EMO = {
    "angry": "angry", "disgust": "disgust", "fear": "fear",
    "happy": "happy", "neutral": "neutral", "sad": "sad",
    "ps": "surprise", "pleasant_surprise": "surprise",
}


@dataclass
class Clip:
    filepath: str
    dataset: str
    speaker_id: str
    emotion: str
    gender: str  # 'M', 'F', or 'U' for unknown


def _walk_wavs(root: Path) -> Iterator[Path]:
    for p in root.rglob("*.wav"):
        yield p


def parse_ravdess(root: Path) -> Iterator[Clip]:
    for wav in _walk_wavs(root):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        emo = RAVDESS_EMO.get(parts[2])
        if emo is None:
            continue
        actor_id = int(parts[6])
        gender = "F" if actor_id % 2 == 0 else "M"  # RAVDESS convention
        yield Clip(str(wav), "ravdess", f"ravdess_{actor_id:02d}", emo, gender)


def parse_cremad(root: Path) -> Iterator[Clip]:
    for wav in _walk_wavs(root):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        emo = CREMAD_EMO.get(parts[2])
        if emo is None:
            continue
        speaker = parts[0]
        # CREMA-D ships a SpeakerDemographics.csv; gender resolution is best-effort
        yield Clip(str(wav), "cremad", f"cremad_{speaker}", emo, "U")


def parse_savee(root: Path) -> Iterator[Clip]:
    pattern = re.compile(r"^([A-Z]{2})_([a-z]+)\d+$")
    for wav in _walk_wavs(root):
        m = pattern.match(wav.stem)
        if not m:
            continue
        speaker, code = m.group(1), m.group(2)
        # 'su' must be checked before 'sa' because both start with 's'
        if code.startswith("su"):
            emo = "surprise"
        elif code.startswith("sa"):
            emo = "sad"
        else:
            emo = SAVEE_EMO.get(code[0])
        if emo is None:
            continue
        yield Clip(str(wav), "savee", f"savee_{speaker}", emo, "M")  # SAVEE = all male


def parse_tess(root: Path) -> Iterator[Clip]:
    for wav in _walk_wavs(root):
        stem = wav.stem.lower()
        emo = None
        for key, label in TESS_EMO.items():
            if stem.endswith(f"_{key}"):
                emo = label
                break
        if emo is None:
            continue
        speaker = "tess_OAF" if stem.startswith("oaf") else "tess_YAF"
        yield Clip(str(wav), "tess", speaker, emo, "F")  # TESS = all female


def build_manifest(raw_dir: str | Path, out_csv: str | Path) -> pd.DataFrame:
    """Walk all four corpora and emit a unified manifest CSV."""
    raw = Path(raw_dir)
    parsers = {
        "ravdess": parse_ravdess,
        "cremad":  parse_cremad,
        "savee":   parse_savee,
        "tess":    parse_tess,
    }
    rows: list[Clip] = []
    for name, fn in parsers.items():
        sub = raw / name
        if not sub.exists():
            print(f"[warn] {sub} not found, skipping {name}")
            continue
        before = len(rows)
        rows.extend(fn(sub))
        print(f"[ok]   {name}: +{len(rows) - before} clips")

    df = pd.DataFrame([c.__dict__ for c in rows])
    df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)
    df["label_id"] = df["emotion"].map(LABEL2ID)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nTotal: {len(df)} clips across {df['speaker_id'].nunique()} speakers")
    print(df["emotion"].value_counts().to_string())
    return df


if __name__ == "__main__":
    build_manifest("data/raw", "data/processed/manifest.csv")
