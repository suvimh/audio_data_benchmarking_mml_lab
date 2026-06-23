from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)
from scripts.config.partition_config import PartitionConfig
from scripts.registry import get_model, list_models
from scripts.utils import ensure_dir, set_seed, save_json
from scripts.validate_config import validate


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_config(path: str | Path, cls: type) -> Any:
    path = Path(path)
    if path.suffix == ".py":
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, cls):
                return attr
        raise ValueError(f"No {cls.__name__} instance found in {path}")
    elif path.suffix in (".yml", ".yaml"):
        return cls.load_yaml(path)
    elif path.suffix == ".json":
        return cls.load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return load_config(path, DatasetConfig)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return load_config(path, BenchmarkConfig)


def load_partition_config(path: str | Path) -> PartitionConfig:
    return load_config(path, PartitionConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return load_config(path, ExperimentConfig)


# ---------------------------------------------------------------------------
# Resolution: combine dataset + partition + filters into a full DatasetConfig
# ---------------------------------------------------------------------------

def resolve_dataset(
    dataset: DatasetConfig,
    partition: PartitionConfig,
    filters: ExperimentFilters,
    frame_duration: Optional[float] = None,
    overlap: Optional[float] = None,
) -> DatasetConfig:
    merged = {
        "name": dataset.name,
        "data_type": dataset.data_type,
        "data_dir": dataset.data_dir,
        "feature_type": dataset.feature_type,
        "train_data_path": dataset.train_data_path,
        "val_data_path": dataset.val_data_path,
        "train_data_paths": dataset.train_data_paths,
        "val_data_paths": dataset.val_data_paths,
        "feature_column": dataset.feature_column,
        "label_column": dataset.label_column,
        "parquet_file": dataset.parquet_file,
        "audio_dir": dataset.audio_dir,
        "sample_rate": dataset.sample_rate,
        "frame_duration": dataset.frame_duration,
        "overlap": dataset.overlap,
        "image_dir": dataset.image_dir,
        "image_size": dataset.image_size,
        "level_names": dataset.level_names,
        "label_level": dataset.label_level,
        "pad_sequences": dataset.pad_sequences,
        "add_channel_dim": dataset.add_channel_dim,
    }
    _override_if_set(merged, "train_singer_ids", partition.train_singer_ids)
    _override_if_set(merged, "val_singer_ids", partition.test_singer_ids)
    if partition.singer_column is not None:
        merged["singer_column"] = partition.singer_column
    _override_if_set(merged, "gender_split", filters.gender)
    _override_if_set(merged, "include_labels", filters.include_labels)
    _override_if_set(merged, "exclude_labels", filters.exclude_labels)
    _override_if_set(merged, "label_map", filters.label_map)
    _override_if_set(merged, "pad_sequences", filters.pad_sequences)
    _override_if_set(merged, "add_channel_dim", filters.add_channel_dim)
    _override_if_set(merged, "frame_duration", frame_duration)
    _override_if_set(merged, "overlap", overlap)
    return DatasetConfig(**merged)


def _override_if_set(target: dict, key: str, value, skip_value=None):
    if value is not None and value != skip_value:
        target[key] = value


def _model_already_completed(model_out: Path) -> bool:
    metrics_path = model_out / "metrics.csv"
    if not metrics_path.exists():
        return False
    import pandas as pd

    df = pd.read_csv(metrics_path)
    if "split" not in df.columns:
        return not df.empty
    return (df["split"] == "validation").any()


def _load_completed_result(model_out: Path, model_name: str) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(model_out / "metrics.csv")
    if "split" in df.columns:
        val_row = df[df["split"] == "validation"].iloc[-1]
        train_row = df[df["split"] == "train"].iloc[-1] if (df["split"] == "train").any() else None
    else:
        val_row = df.iloc[-1]
        train_row = None

    def _row_to_metrics(row) -> Dict[str, Any]:
        return {
            "accuracy": float(row["accuracy"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "n_samples": int(row["n_samples"]) if pd.notna(row.get("n_samples")) else None,
        }

    result: Dict[str, Any] = {
        "model_name": model_name,
        "val_metrics": _row_to_metrics(val_row),
    }
    if train_row is not None:
        result["train_metrics"] = _row_to_metrics(train_row)
    return result

# ---------------------------------------------------------------------------
# Old-style run: single dataset + single benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_config: DatasetConfig | str | Path,
    benchmark_config: BenchmarkConfig | str | Path,
) -> Dict[str, Any]:
    if not isinstance(dataset_config, DatasetConfig):
        dataset_config = load_dataset_config(dataset_config)
    if not isinstance(benchmark_config, BenchmarkConfig):
        benchmark_config = load_benchmark_config(benchmark_config)

    print(f"Dataset:  {dataset_config.name} (type: {dataset_config.data_type})")
    print(f"Benchmark: {benchmark_config.name}")
    print(f"Models:   {', '.join(benchmark_config.models)}")
    print()

    set_seed(benchmark_config.seed)

    warnings = validate(dataset_config, benchmark_config)
    has_errors = False
    for level, msg in warnings:
        print(f"[{level.upper()}] {msg}")
        if level == "error":
            has_errors = True
    if has_errors:
        print("Configuration errors found. Aborting.")
        sys.exit(1)

    output_dir = ensure_dir(benchmark_config.output_dir)

    results = {}
    for model_name in benchmark_config.models:
        print(f"\n{'=' * 60}")
        print(f"Running model: {model_name}")
        print(f"{'=' * 60}")

        model_output_dir = ensure_dir(
            output_dir / f"{benchmark_config.name}_{model_name}"
        )

        if _model_already_completed(model_output_dir):
            print(f"Skipping {model_name} (metrics already exist)")
            results[model_name] = _load_completed_result(model_output_dir, model_name)
            continue

        model_info = get_model(model_name)
        result = model_info["run"](dataset_config, benchmark_config, str(model_output_dir))
        results[model_name] = result

        print(f"Completed: {model_name}")

    summary_path = output_dir / f"{benchmark_config.name}_summary.json"
    summary = {
        "dataset": dataset_config.name,
        "benchmark": benchmark_config.name,
        "models": list(results.keys()),
        "results": {
            name: {
                "val_balanced_accuracy": res.get("val_metrics", {}).get("balanced_accuracy"),
                "val_accuracy": res.get("val_metrics", {}).get("accuracy"),
            }
            for name, res in results.items()
        },
    }
    save_json(summary, summary_path)
    print(f"\nSummary saved to {summary_path}")

    return results


# ---------------------------------------------------------------------------
# New-style run: experiment config with blocks
# ---------------------------------------------------------------------------

def run_experiment(experiment_config_path: str | Path) -> Dict[str, Any]:
    experiment = load_experiment_config(experiment_config_path)

    print(f"\n{'=' * 60}")
    print(f"Experiment: {experiment.name}")
    print(f"Blocks:     {len(experiment.blocks)}")
    print(f"{'=' * 60}\n")

    all_results = {}

    for block_idx, block in enumerate(experiment.blocks):
        print(f"\n{'─' * 60}")
        print(f"Block {block_idx + 1}: {block.model_type}")
        print(f"  Benchmark: {block.benchmark_config.name}")
        print(f"  Models:    {', '.join(block.benchmark_config.models)}")
        print(f"  Partition: {block.partition.name}")
        print(f"  Datasets:  {', '.join(d.name for d in block.datasets)}")
        if block.filters.gender:
            print(f"  Gender:    {block.filters.gender}")
        if block.filters.include_labels:
            print(f"  Labels:    {len(block.filters.include_labels)} classes")
        print(f"{'─' * 60}\n")

        set_seed(block.benchmark_config.seed)

        for dataset in block.datasets:
            resolved = resolve_dataset(
                dataset, block.partition, block.filters,
                frame_duration=block.frame_duration,
                overlap=block.overlap,
            )

            print(f"\n  Dataset: {dataset.name}")

            warnings = validate(resolved, block.benchmark_config)
            has_errors = False
            for level, msg in warnings:
                print(f"  [{level}] {msg}")
                if level == "error":
                    has_errors = True
            if has_errors:
                print("  Skipping due to errors.")
                continue

            block_out = ensure_dir(
                Path(experiment.output_dir)
                / experiment.name
                / block.benchmark_config.name
                / dataset.name
            )

            for model_name in block.benchmark_config.models:
                print(f"\n  {'=' * 50}")
                print(f"  Model: {model_name}")
                print(f"  {'=' * 50}")

                model_out = ensure_dir(block_out / model_name)

                if _model_already_completed(model_out):
                    print(f"  Skipping {model_name} (metrics already exist)")
                    key = f"{block.benchmark_config.name}/{dataset.name}/{model_name}"
                    all_results[key] = _load_completed_result(model_out, model_name)
                    print(f"  Completed: {key}")
                    continue

                model_info = get_model(model_name)
                result = model_info["run"](
                    resolved, block.benchmark_config, str(model_out)
                )

                key = f"{block.benchmark_config.name}/{dataset.name}/{model_name}"
                all_results[key] = result

                print(f"  Completed: {key}")

    summary_path = Path(experiment.output_dir) / f"{experiment.name}_summary.json"
    summary = {
        "experiment": experiment.name,
        "results": {
            key: {
                "val_balanced_accuracy": res.get("val_metrics", {}).get("balanced_accuracy"),
                "val_accuracy": res.get("val_metrics", {}).get("accuracy"),
            }
            for key, res in all_results.items()
        },
    }
    save_json(summary, summary_path)
    print(f"\nSummary saved to {summary_path}")

    return all_results
