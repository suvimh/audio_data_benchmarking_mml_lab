from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.config.dataset_config import DatasetConfig


@dataclass
class AudioMetadata:
    filepath: str
    label: str
    singer_id: str
    gender: str
    technique: str
    exercise: Optional[str] = None


def scan_audio_files(
    audio_dir: str | Path,
    extension: str = ".wav",
    train_singers: Optional[List[str]] = None,
    val_singers: Optional[List[str]] = None,
    gender_filter: Optional[str] = None,
    include_labels: Optional[List[str]] = None,
    exclude_labels: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    audio_dir = Path(audio_dir)
    all_files = list(audio_dir.rglob(f"*{extension}"))

    records = []
    for fp in all_files:
        rel = fp.relative_to(audio_dir)
        parts = list(rel.parts)
        singer_id = parts[0]
        technique = parts[1] if len(parts) > 1 else "unknown"
        gender = "female" if singer_id.startswith("f") else "male" if singer_id.startswith("m") else "unknown"

        if gender_filter and gender_filter != "mixed" and gender != gender_filter:
            continue
        if include_labels and technique not in include_labels:
            continue
        if exclude_labels and technique in exclude_labels:
            continue

        records.append({
            "filepath": str(fp),
            "singer_id": singer_id,
            "gender": gender,
            "technique": technique,
            "label": technique,
        })

    df = pd.DataFrame(records)

    result = {}

    if train_singers is not None and val_singers is not None:
        train_df = df[df["singer_id"].isin(train_singers)]
        val_df = df[df["singer_id"].isin(val_singers)]
        result["train"] = train_df
        result["val"] = val_df
    else:
        from sklearn.model_selection import train_test_split
        singers = df["singer_id"].unique()
        train_ids, val_ids = train_test_split(
            singers, test_size=0.25, random_state=42
        )
        train_df = df[df["singer_id"].isin(train_ids)]
        val_df = df[df["singer_id"].isin(val_ids)]
        result["train"] = train_df
        result["val"] = val_df

    return result


def load_audio_file(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio


def frame_audio(
    audio: np.ndarray,
    sample_rate: int,
    frame_duration: float,
    overlap: float,
) -> List[np.ndarray]:
    frame_len = int(sample_rate * frame_duration)
    hop_len = int(frame_len * (1 - overlap))
    if len(audio) < frame_len:
        padding = frame_len - len(audio)
        audio = np.pad(audio, (0, padding))
    frames = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frames.append(audio[start: start + frame_len])
    return frames
