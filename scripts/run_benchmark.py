from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.registry import get_model, list_models
from scripts.utils import ensure_dir, set_seed, save_json
from scripts.validate_config import validate


def load_dataset_config(path: str | Path) -> DatasetConfig:
    path = Path(path)
    if path.suffix == ".py":
        import importlib.util
        spec = importlib.util.spec_from_file_location("dataset_config", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, DatasetConfig):
                return attr
        raise ValueError(f"No DatasetConfig instance found in {path}")
    elif path.suffix in (".yml", ".yaml"):
        return DatasetConfig.load_yaml(path)
    elif path.suffix == ".json":
        return DatasetConfig.load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    if path.suffix == ".py":
        import importlib.util
        spec = importlib.util.spec_from_file_location("benchmark_config", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, BenchmarkConfig):
                return attr
        raise ValueError(f"No BenchmarkConfig instance found in {path}")
    elif path.suffix in (".yml", ".yaml"):
        return BenchmarkConfig.load_yaml(path)
    elif path.suffix == ".json":
        return BenchmarkConfig.load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def run_benchmark(
    dataset_config_path: str | Path,
    benchmark_config_path: str | Path,
) -> Dict[str, Any]:
    dataset_config = load_dataset_config(dataset_config_path)
    bench_config = load_benchmark_config(benchmark_config_path)

    print(f"Dataset:  {dataset_config.name} (type: {dataset_config.data_type})")
    print(f"Benchmark: {bench_config.name}")
    print(f"Models:   {', '.join(bench_config.models)}")
    print()

    set_seed(bench_config.seed)

    warnings = validate(dataset_config, bench_config)
    has_errors = False
    for level, msg in warnings:
        print(f"[{level.upper()}] {msg}")
        if level == "error":
            has_errors = True
    if has_errors:
        print("Configuration errors found. Aborting.")
        sys.exit(1)

    output_dir = ensure_dir(bench_config.output_dir)

    results = {}
    for model_name in bench_config.models:
        print(f"\n{'=' * 60}")
        print(f"Running model: {model_name}")
        print(f"{'=' * 60}")

        model_output_dir = ensure_dir(
            output_dir / f"{bench_config.name}_{model_name}"
        )

        model_info = get_model(model_name)
        result = model_info["run"](dataset_config, bench_config, str(model_output_dir))
        results[model_name] = result

        print(f"Completed: {model_name}")

    summary_path = output_dir / f"{bench_config.name}_summary.json"
    summary = {
        "dataset": dataset_config.name,
        "benchmark": bench_config.name,
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
