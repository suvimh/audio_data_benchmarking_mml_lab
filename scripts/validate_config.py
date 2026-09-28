from __future__ import annotations

from typing import List, Tuple

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.registry import list_models, supports_data_type


def validate(
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> List[Tuple[str, str]]:
    warnings = []

    if not benchmark_config.models:
        warnings.append(("error", "No models specified in benchmark config"))

    available = {m["name"] for m in list_models()}
    for model_name in benchmark_config.models:
        if model_name not in available:
            warnings.append(("error", f"Model '{model_name}' is not registered. Available: {sorted(available)}"))
            continue
        if not supports_data_type(model_name, dataset_config.data_type):
            warnings.append((
                "error",
                f"Model '{model_name}' does not support data type '{dataset_config.data_type}'. "
                f"Supported types: {_get_supported(model_name)}",
            ))

    if dataset_config.data_type == "features":
        has_pkl = dataset_config.train_data_path or dataset_config.train_data_paths
        has_parquet = dataset_config.data_dir and dataset_config.parquet_file
        if not (has_pkl or has_parquet or dataset_config.data_dir):
            warnings.append((
                "error",
                "Features dataset: no data source found. Set train_data_path + val_data_path, "
                "train_data_paths + val_data_paths, or data_dir + parquet_file.",
            ))
        if dataset_config.feature_type is None:
            warnings.append(("warning", "Features dataset: no feature_type specified"))
        if has_parquet and dataset_config.singer_column is None:
            if dataset_config.train_singer_ids or dataset_config.val_singer_ids:
                warnings.append((
                    "warning",
                    "Parquet: singer_column not set but train/val_singer_ids provided. "
                    "Set singer_column so the train/val split can be applied.",
                ))

    if dataset_config.data_type == "raw_audio":
        if not (dataset_config.audio_dir or dataset_config.data_dir):
            warnings.append(("error", "Raw audio dataset: no audio_dir or data_dir specified"))

    if dataset_config.data_type == "images":
        if not dataset_config.image_dir:
            warnings.append(("error", "Image dataset: no image_dir specified"))

    return warnings


def _get_supported(model_name: str) -> List[str]:
    for m in list_models():
        if m["name"] == model_name:
            return m["supported_data_types"]
    return []
