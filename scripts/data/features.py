from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.feature_selection.base import resolve_feature_selection

_METADATA_COLUMNS = frozenset({
    "embedding",
    "embedding_dtype",
    "embedding_shape",
    "filename",
    "filepath",
    "frame_index",
    "extractor",
})

_FINNISH_EMOTION_MAP = {
    "ilo": "joy",
    "surullisuus": "sadness",
    "suru": "sadness",
    "viha": "anger",
    "lempeys": "gentleness",
    "neutraali": "neutral",
    "neutral": "neutral",
}


@dataclass
class FeaturesData:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    class_names: List[str]
    n_classes: int
    input_dim: int
    preprocessed: bool = False
    feature_selection_info: Optional[Dict[str, Any]] = None


def load_features_data(
    dataset_config: DatasetConfig,
    benchmark_config: Optional[BenchmarkConfig] = None,
) -> FeaturesData:
    if dataset_config.parquet_file and dataset_config.data_dir:
        return _load_from_parquet(dataset_config, benchmark_config)
    if dataset_config.train_data_path or dataset_config.train_data_paths:
        return _load_from_pkl(dataset_config, benchmark_config)
    raise ValueError(
        "No data source found. Set data_dir + parquet_file or train/val data paths."
    )


def _load_from_parquet(
    dataset_config: DatasetConfig,
    benchmark_config: Optional[BenchmarkConfig],
) -> FeaturesData:
    path = Path(dataset_config.data_dir) / dataset_config.parquet_file
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    df = _deserialize_embeddings(df, dataset_config.feature_column)
    df = _fill_missing_emotion_labels(df, dataset_config)
    df = _apply_label_filters(df, dataset_config)
    df = _drop_empty_labels(df, dataset_config.label_column)

    train_df, val_df = _split_by_singer(df, dataset_config)

    class_names = _resolve_class_names(train_df, val_df, dataset_config)
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    X_train = _stack_features(train_df, dataset_config)
    X_val = _stack_features(val_df, dataset_config)
    y_train = _encode_labels(train_df, dataset_config.label_column, label_to_idx)
    y_val = _encode_labels(val_df, dataset_config.label_column, label_to_idx)

    input_dim = X_train.shape[1] if X_train.ndim == 2 else X_train.shape[-1]

    X_train, X_val, fs_info, preprocessed, input_dim = _maybe_apply_feature_selection(
        X_train,
        X_val,
        y_train,
        dataset_config,
        benchmark_config,
        input_dim,
    )

    if dataset_config.pad_sequences:
        X_train = _reshape_for_cnn(X_train, dataset_config.add_channel_dim)
        X_val = _reshape_for_cnn(X_val, dataset_config.add_channel_dim)
        input_dim = int(X_train.shape[1])

    return FeaturesData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        class_names=class_names,
        n_classes=len(class_names),
        input_dim=input_dim,
        preprocessed=preprocessed,
        feature_selection_info=fs_info,
    )


def _load_from_pkl(
    dataset_config: DatasetConfig,
    benchmark_config: Optional[BenchmarkConfig],
) -> FeaturesData:
    from scripts.utils import load_pickle

    def _load_split(paths: Optional[Dict[str, str]], single: Optional[str]) -> pd.DataFrame:
        if paths:
            frames = [load_pickle(p) for p in paths.values()]
            if isinstance(frames[0], pd.DataFrame):
                return pd.concat(frames, ignore_index=True)
            return pd.concat([pd.DataFrame(f) for f in frames], ignore_index=True)
        if single:
            data = load_pickle(single)
            return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        raise ValueError("No pkl path configured.")

    train_df = _load_split(dataset_config.train_data_paths, dataset_config.train_data_path)
    val_df = _load_split(dataset_config.val_data_paths, dataset_config.val_data_path)

    feat_col = dataset_config.feature_column
    if feat_col in train_df.columns and isinstance(train_df[feat_col].iloc[0], (list, np.ndarray)):
        pass
    else:
        raise ValueError(f"Expected feature column '{feat_col}' in pkl data.")

    train_df = _apply_label_filters(train_df, dataset_config)
    val_df = _apply_label_filters(val_df, dataset_config)

    class_names = _resolve_class_names(train_df, val_df, dataset_config)
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    X_train = np.stack(train_df[feat_col].values)
    X_val = np.stack(val_df[feat_col].values)
    y_train = _encode_labels(train_df, dataset_config.label_column, label_to_idx)
    y_val = _encode_labels(val_df, dataset_config.label_column, label_to_idx)

    input_dim = X_train.shape[1]
    X_train, X_val, fs_info, preprocessed, input_dim = _maybe_apply_feature_selection(
        X_train,
        X_val,
        y_train,
        dataset_config,
        benchmark_config,
        input_dim,
    )

    if dataset_config.pad_sequences:
        X_train = _reshape_for_cnn(X_train, dataset_config.add_channel_dim)
        X_val = _reshape_for_cnn(X_val, dataset_config.add_channel_dim)
        input_dim = int(X_train.shape[1])

    return FeaturesData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        class_names=class_names,
        n_classes=len(class_names),
        input_dim=input_dim,
        preprocessed=preprocessed,
        feature_selection_info=fs_info,
    )


def _maybe_apply_feature_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    dataset_config: DatasetConfig,
    benchmark_config: Optional[BenchmarkConfig],
    input_dim: int,
) -> tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]], bool, int]:
    fs_config = resolve_feature_selection(
        dataset_config.feature_type,
        dataset_config.name,
        benchmark_config.feature_selection if benchmark_config else None,
    )
    if fs_config is None:
        return X_train, X_val, None, False, input_dim

    from scripts.feature_selection.pipeline import apply_feature_selection

    method, params = fs_config
    X_train, X_val, fs_info = apply_feature_selection(
        X_train, X_val, y_train, method, params
    )
    return X_train, X_val, fs_info, True, X_train.shape[1]


def _deserialize_embeddings(df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
    df = df.copy()

    def _to_array(val: Any) -> np.ndarray:
        if isinstance(val, (bytes, bytearray)):
            dtype = np.float32
            if "embedding_dtype" in df.columns:
                try:
                    dtype = np.dtype(df["embedding_dtype"].iloc[0])
                except (TypeError, ValueError):
                    dtype = np.float32
            return np.frombuffer(val, dtype=dtype)
        if isinstance(val, np.ndarray):
            return val.astype(np.float32, copy=False)
        return np.asarray(val, dtype=np.float32)

    df[feature_column] = df[feature_column].apply(_to_array)
    return df


def _fill_missing_emotion_labels(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    label_col = config.label_column
    if label_col not in df.columns or "filename" not in df.columns:
        return df

    df = df.copy()
    empty_mask = df[label_col].astype(str).str.len() == 0
    if not empty_mask.any():
        return df

    def _infer_from_filename(filename: str) -> str:
        stem = str(filename).lower().replace(".wav", "")
        for token, emotion in _FINNISH_EMOTION_MAP.items():
            if token in stem:
                return emotion
        return ""

    df.loc[empty_mask, label_col] = df.loc[empty_mask, "filename"].apply(_infer_from_filename)
    return df


def _apply_label_filters(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    label_col = config.label_column
    if config.include_labels:
        df = df[df[label_col].isin(config.include_labels)]
    if config.exclude_labels:
        df = df[~df[label_col].isin(config.exclude_labels)]
    return df.reset_index(drop=True)


def _drop_empty_labels(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    return df[df[label_column].astype(str).str.len() > 0].reset_index(drop=True)


def _split_by_singer(
    df: pd.DataFrame,
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    singer_col = config.singer_column or "singer"
    if config.train_singer_ids is None or config.val_singer_ids is None:
        raise ValueError(
            f"Singer split requires train_singer_ids and val_singer_ids "
            f"(column: {singer_col})."
        )
    train_df = df[df[singer_col].isin(config.train_singer_ids)].reset_index(drop=True)
    val_df = df[df[singer_col].isin(config.val_singer_ids)].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Empty split: train={len(train_df)}, val={len(val_df)}. "
            f"Check singer IDs against column '{singer_col}'."
        )
    return train_df, val_df


def _resolve_class_names(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: DatasetConfig,
) -> List[str]:
    if config.include_labels:
        return list(config.include_labels)
    labels = pd.concat([
        train_df[config.label_column],
        val_df[config.label_column],
    ]).unique()
    return sorted(str(l) for l in labels if str(l))


def _stack_features(df: pd.DataFrame, config: DatasetConfig) -> np.ndarray:
    features = np.stack(df[config.feature_column].values)
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    return features.astype(np.float32)


def _encode_labels(
    df: pd.DataFrame,
    label_column: str,
    label_to_idx: Dict[str, int],
) -> np.ndarray:
    return np.array([label_to_idx[str(label)] for label in df[label_column]], dtype=np.int32)


def _reshape_for_cnn(X: np.ndarray, add_channel_dim: bool, max_length: int = 128) -> np.ndarray:
    if X.ndim == 4:
        return X
    if X.ndim == 3:
        if add_channel_dim and X.shape[-1] != 1:
            X = X[..., np.newaxis]
        return X
    if X.ndim == 2:
        if X.shape[1] > max_length:
            step = int(np.ceil(X.shape[1] / max_length))
            X = X[:, ::step][:, :max_length]
        if add_channel_dim:
            return X[:, :, np.newaxis]
        return X
    raise ValueError(f"Expected 2D-4D feature array for CNN, got shape {X.shape}")
