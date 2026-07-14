from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.config.dataset_config import DatasetConfig


@dataclass
class FeaturesData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    class_names: List[str]
    input_dim: int
    n_classes: int
    n_timesteps: Optional[int] = None


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _flatten_pad_sequences(features_list: List[np.ndarray]) -> Tuple[np.ndarray, Optional[int]]:
    lengths = [f.shape[0] for f in features_list]
    max_len = max(lengths)
    padded = np.zeros((len(features_list), max_len, features_list[0].shape[1]))
    for i, f in enumerate(features_list):
        padded[i, : f.shape[0]] = f
    return padded, max_len


def _flatten_concat(features_list: List[np.ndarray]) -> np.ndarray:
    return np.array([f.ravel() for f in features_list])


def _apply_label_filter(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    label_col = config.label_column
    if config.include_labels:
        df = df[df[label_col].isin(config.include_labels)]
    if config.exclude_labels:
        df = df[~df[label_col].isin(config.exclude_labels)]
    return df


def _build_class_mapping(df: pd.DataFrame, config: DatasetConfig) -> Tuple[List[str], Dict[str, int]]:
    label_col = config.label_column
    if config.label_map:
        classes = sorted(config.label_map.keys(), key=lambda c: config.label_map[c])
        return classes, config.label_map
    classes = sorted(df[label_col].unique())
    class_map = {c: i for i, c in enumerate(classes)}
    return classes, class_map


def _encode_labels(df: pd.DataFrame, class_map: dict, config: DatasetConfig) -> np.ndarray:
    label_col = config.label_column
    return np.array([class_map[c] for c in df[label_col]])


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def _is_parquet(path: str | Path) -> bool:
    return str(path).endswith((".parquet", ".pq"))


def _parse_parquet_embeddings(df: pd.DataFrame) -> List[np.ndarray]:
    raw = df["embedding"].values
    first = raw[0]
    if isinstance(first, bytes):
        return [np.frombuffer(b, dtype=np.float32) for b in raw]
    if isinstance(first, np.ndarray):
        return list(raw)
    raise TypeError(f"Unexpected embedding type: {type(first)}")


def _load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def _auto_detect_label_columns(df: pd.DataFrame, config: DatasetConfig) -> List[str]:
    if config.label_column not in df.columns and config.label_column not in ("Features", "Class"):
        skip = {"embedding", "embedding_dtype", "embedding_shape",
                "filename", "filepath", "frame_index", "extractor"}
        candidates = [c for c in df.columns if c not in skip]
        if len(candidates) == 1:
            return candidates
        if config.label_level and config.label_level in df.columns:
            return [config.label_level]
        return candidates
    return [config.label_column]


def _add_singer_column(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    if config.singer_column and config.singer_column in df.columns:
        return df
    if "singer" in df.columns:
        config.singer_column = "singer"
    elif "filepath" in df.columns:
        config.singer_column = "filepath"
    return df


def _split_by_singer(df: pd.DataFrame, config: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    singer_col = config.singer_column
    if singer_col and config.train_singer_ids and config.val_singer_ids:
        train_df = df[df[singer_col].isin(config.train_singer_ids)]
        val_df = df[df[singer_col].isin(config.val_singer_ids)]
        return train_df, val_df

    if config.train_data_path and config.val_data_path:
        return df, pd.DataFrame()

    return df, pd.DataFrame()


# ---------------------------------------------------------------------------
# Pkl helpers
# ---------------------------------------------------------------------------

def _is_pkl(path: str | Path) -> bool:
    return str(path).endswith(".pkl")


def _load_pkl(path: str | Path) -> pd.DataFrame:
    return pd.read_pickle(path)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def _resolve_data_paths(config: DatasetConfig) -> Tuple[Optional[str], Optional[str]]:
    if config.train_data_path and config.val_data_path:
        return config.train_data_path, config.val_data_path

    if config.train_data_paths and config.val_data_paths:
        gender = config.gender_split or "mixed"
        if gender in config.train_data_paths:
            return config.train_data_paths[gender], config.val_data_paths[gender]
        if gender == "mixed":
            return None, None

    if config.data_dir and config.parquet_file:
        parquet_path = str(Path(config.data_dir) / config.parquet_file)
        return parquet_path, None

    if config.data_dir:
        parquet_files = sorted(Path(config.data_dir).glob("*.parquet"))
        if not parquet_files:
            pkl_files = sorted(Path(config.data_dir).glob("*.pkl"))
            if pkl_files:
                return str(pkl_files[0]), None
            return None, None
        return str(parquet_files[0]), None

    return None, None


def load_features_data(config: DatasetConfig) -> FeaturesData:
    train_path, val_path = _resolve_data_paths(config)

    if not train_path:
        raise ValueError(
            "No data source found. Set train_data_path + val_data_path, "
            "train_data_paths + val_data_paths, or data_dir + parquet_file."
        )

    is_parquet_file = _is_parquet(train_path)

    if is_parquet_file:
        if val_path and _is_parquet(val_path):
            train_df = _load_parquet(train_path)
            val_df = _load_parquet(val_path)
        else:
            df = _load_parquet(train_path)
            df = _add_singer_column(df, config)
            train_df, val_df = _split_by_singer(df, config)

        label_cols = _auto_detect_label_columns(pd.concat([train_df, val_df] if not val_df.empty else [train_df], ignore_index=True), config)
        if config.label_column not in df.columns and config.label_column in ("Features", "Class"):
            config.label_column = label_cols[0]

        embeddings_train = _parse_parquet_embeddings(train_df)
        embeddings_val = _parse_parquet_embeddings(val_df) if not val_df.empty else []
    else:
        if config.train_data_paths and config.val_data_paths and not (config.train_data_path and config.val_data_path):
            gender = config.gender_split or "mixed"
            if gender == "mixed":
                train_dfs, val_dfs = [], []
                for g in ["female", "male"]:
                    tp = config.train_data_paths.get(g)
                    vp = config.val_data_paths.get(g)
                    if tp and vp:
                        train_dfs.append(pd.read_pickle(tp))
                        val_dfs.append(pd.read_pickle(vp))
                train_df = pd.concat(train_dfs, ignore_index=True)
                val_df = pd.concat(val_dfs, ignore_index=True)
            else:
                train_df = pd.read_pickle(config.train_data_paths[gender])
                val_df = pd.read_pickle(config.val_data_paths[gender])
        else:
            train_df = pd.read_pickle(train_path)
            val_df = pd.read_pickle(val_path) if val_path else pd.DataFrame()

        embeddings_train = train_df[config.feature_column].tolist()
        embeddings_val = val_df[config.feature_column].tolist() if not val_df.empty else []

    train_df = _apply_label_filter(train_df, config)
    if not val_df.empty:
        val_df = _apply_label_filter(val_df, config)

    classes, class_map = _build_class_mapping(train_df, config)

    y_train = _encode_labels(train_df, class_map, config)
    y_val = _encode_labels(val_df, class_map, config) if not val_df.empty else np.array([])

    if not embeddings_train:
        raise ValueError("No training data after filtering")

    input_dim = embeddings_train[0].shape[-1] if embeddings_train[0].ndim > 0 else 1

    if config.pad_sequences:
        X_train, n_timesteps = _flatten_pad_sequences(embeddings_train)
        X_val, _ = _flatten_pad_sequences(embeddings_val) if embeddings_val else (np.array([]), None)
        if config.add_channel_dim:
            X_train = X_train[..., np.newaxis]
            if X_val.size:
                X_val = X_val[..., np.newaxis]
    else:
        X_train = _flatten_concat(embeddings_train)
        X_val = _flatten_concat(embeddings_val) if embeddings_val else np.array([])
        n_timesteps = None

    return FeaturesData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_names=classes,
        input_dim=input_dim,
        n_classes=len(classes),
        n_timesteps=n_timesteps,
    )
