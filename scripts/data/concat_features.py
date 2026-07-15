from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.data.features import (
    FeaturesData,
    _apply_label_filters,
    _deserialize_embeddings,
    _drop_empty_labels,
    _encode_labels,
    _fill_missing_emotion_labels,
    _maybe_apply_feature_selection,
    _resolve_class_names,
    _split_by_singer,
    _stack_features,
)


def load_concat_features_data(
    dataset_configs: List[DatasetConfig],
    benchmark_config: Optional[BenchmarkConfig] = None,
    join_keys: Optional[List[str]] = None,
) -> FeaturesData:
    """Load and horizontally concatenate embeddings from multiple parquet sources."""
    if len(dataset_configs) < 2:
        raise ValueError("load_concat_features_data requires at least two dataset configs")

    join_keys = join_keys or ["filename", "frame_index", "singer"]
    base_config = dataset_configs[0]
    merged_df = _load_merged_parquet(dataset_configs[0], join_keys)

    for config in dataset_configs[1:]:
        other_df = _load_merged_parquet(config, join_keys)
        embed_col = f"__embedding__{config.name}"
        other_df = other_df.rename(columns={config.feature_column: embed_col})
        keep_cols = join_keys + [embed_col]
        merged_df = merged_df.merge(
            other_df[keep_cols],
            on=join_keys,
            how="inner",
            validate="one_to_one",
        )

    if merged_df.empty:
        raise ValueError("Concatenated dataset is empty after inner join on join keys")

    merged_df = _apply_label_filters(merged_df, base_config)
    merged_df = _drop_empty_labels(merged_df, base_config.label_column)
    train_df, val_df = _split_by_singer(merged_df, base_config)
    class_names = _resolve_class_names(train_df, val_df, base_config)
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    embed_cols = [c for c in merged_df.columns if c.startswith("__embedding__")]
    X_train = _stack_concat_features(train_df, embed_cols)
    X_val = _stack_concat_features(val_df, embed_cols)
    y_train = _encode_labels(train_df, base_config.label_column, label_to_idx)
    y_val = _encode_labels(val_df, base_config.label_column, label_to_idx)

    input_dim = X_train.shape[1]
    concat_config = DatasetConfig(
        name="concat",
        data_type="features",
        feature_type="concat",
        include_labels=base_config.include_labels,
        exclude_labels=base_config.exclude_labels,
        label_column=base_config.label_column,
        train_singer_ids=base_config.train_singer_ids,
        val_singer_ids=base_config.val_singer_ids,
        singer_column=base_config.singer_column,
    )
    X_train, X_val, fs_info, preprocessed, input_dim = _maybe_apply_feature_selection(
        X_train,
        X_val,
        y_train,
        concat_config,
        benchmark_config,
        input_dim,
    )

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


def _load_merged_parquet(config: DatasetConfig, join_keys: List[str]) -> pd.DataFrame:
    path = Path(config.data_dir) / config.parquet_file
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    df = _deserialize_embeddings(df, config.feature_column)
    df = _fill_missing_emotion_labels(df, config)
    missing_keys = [key for key in join_keys if key not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing join keys {missing_keys} in {path}")
    return df


def _stack_concat_features(df: pd.DataFrame, embed_cols: List[str]) -> np.ndarray:
    parts = []
    for col in embed_cols:
        arr = np.stack(df[col].values)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        parts.append(arr.astype(np.float32))
    return np.hstack(parts)
