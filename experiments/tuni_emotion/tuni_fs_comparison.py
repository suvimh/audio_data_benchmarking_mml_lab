"""
Compare OpenSMILE full-embedding baseline vs filter feature selection.

Usage:
    python -m experiments.tuni_emotion.tuni_fs_comparison
"""
from __future__ import annotations

from pathlib import Path

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.experiment_config import ExperimentFilters
from scripts.config.partition_config import PartitionConfig
from scripts.feature_selection.base import FeatureSelectionConfig
from scripts.registry import get_model
from scripts.run_benchmark import resolve_dataset
from scripts.utils import ensure_dir, save_json

from experiments.tuni_emotion.tuni_benchmark_configs import (
    trad_ml_benchmark,
    trad_ml_benchmark_filter,
)
from experiments.tuni_emotion.tuni_data_featureset_embedding_configs import (
    make_opensmile_embedding_config,
)
from experiments.tuni_emotion.tuni_experiments_constants import (
    TUNI_EMOTIONS,
    TRAIN_SINGERS,
    VAL_SINGERS,
)

partition = PartitionConfig(
    name="tuni_singer_independent",
    train_singer_ids=TRAIN_SINGERS,
    test_singer_ids=VAL_SINGERS,
    singer_column="singer",
)

_filters = ExperimentFilters(
    gender="mixed",
    include_labels=TUNI_EMOTIONS,
)

opensmile = make_opensmile_embedding_config(frame_duration=3.0)


def _macro_f1(metrics: dict) -> float:
    report = metrics.get("classification_report", {})
    return float(report.get("macro avg", {}).get("f1-score", 0.0))


def _run_benchmark(
    benchmark: BenchmarkConfig,
    label: str,
    out_root: Path,
) -> dict:
    resolved = resolve_dataset(
        opensmile, partition, _filters,
        frame_duration=3.0,
        overlap=0.25,
    )
    results = {}
    combo_out = ensure_dir(out_root / label)

    for model_name in benchmark.models:
        model_out = ensure_dir(combo_out / model_name)
        model_info = get_model(model_name)
        result = model_info["run"](resolved, benchmark, str(model_out))
        results[model_name] = {
            "val_macro_f1": _macro_f1(result.get("val_metrics", {})),
            "val_accuracy": result.get("val_metrics", {}).get("accuracy"),
        }

    results["_mean_val_macro_f1"] = sum(
        r["val_macro_f1"] for name, r in results.items() if not name.startswith("_")
    ) / len(benchmark.models)
    return results


def run_opensmile_comparison(output_dir: str = "./results") -> dict:
    out_root = ensure_dir(Path(output_dir) / "tuni_emotion_fs_comparison")

    print("Running OpenSMILE full-embedding baseline...")
    baseline = _run_benchmark(trad_ml_benchmark, "baseline", out_root)

    print("Running OpenSMILE + filter feature selection...")
    with_fs = _run_benchmark(trad_ml_benchmark_filter, "filter", out_root)

    summary = {
        "extractor": "opensmile",
        "baseline": baseline,
        "filter": with_fs,
        "improvement_mean_macro_f1": (
            with_fs["_mean_val_macro_f1"] - baseline["_mean_val_macro_f1"]
        ),
    }

    summary_path = out_root / "tuni_emotion_opensmile_fs_comparison.json"
    save_json(summary, summary_path)
    print(f"\nComparison saved to {summary_path}")
    print(f"Baseline mean macro-F1: {baseline['_mean_val_macro_f1']:.4f}")
    print(f"Filter mean macro-F1:  {with_fs['_mean_val_macro_f1']:.4f}")
    return summary


if __name__ == "__main__":
    run_opensmile_comparison()
