"""
Compare OpenSMILE baseline vs filter vs generalized Fisher feature selection.

Usage:
    python -m experiments.tuni_emotion.tuni_fs_comparison
"""
from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.experiment_config import ExperimentFilters
from scripts.config.partition_config import PartitionConfig
from scripts.registry import get_model
from scripts.run_benchmark import resolve_dataset
from scripts.utils import ensure_dir, save_json, set_seed

from experiments.tuni_emotion.tuni_benchmark_configs import (
    trad_ml_benchmark,
    trad_ml_benchmark_filter,
    trad_ml_benchmark_generalized_fisher,
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

METHOD_BENCHMARKS = {
    "baseline": trad_ml_benchmark,
    "filter": trad_ml_benchmark_filter,
    "generalized_fisher": trad_ml_benchmark_generalized_fisher,
}
DEFAULT_SEEDS = [42, 52, 62]


def _macro_f1(metrics: dict) -> float:
    report = metrics.get("classification_report", {})
    return float(report.get("macro avg", {}).get("f1-score", 0.0))


def _run_benchmark_once(
    benchmark: BenchmarkConfig,
    method_label: str,
    out_root: Path,
    seed: int,
) -> dict:
    benchmark_run = deepcopy(benchmark)
    benchmark_run.seed = seed

    resolved = resolve_dataset(
        opensmile,
        partition,
        _filters,
        frame_duration=3.0,
        overlap=0.25,
    )
    results: dict = {"models": {}}
    combo_out = ensure_dir(out_root / method_label / f"seed_{seed}")

    set_seed(seed)
    for model_name in benchmark_run.models:
        model_out = ensure_dir(combo_out / model_name)
        model_info = get_model(model_name)
        start = time.perf_counter()
        result = model_info["run"](resolved, benchmark_run, str(model_out))
        elapsed = time.perf_counter() - start

        val_metrics = result.get("val_metrics", {})
        fs_info = result.get("feature_selection_info") or {}
        results["models"][model_name] = {
            "val_macro_f1": _macro_f1(val_metrics),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_balanced_accuracy": float(val_metrics.get("balanced_accuracy", 0.0)),
            "runtime_seconds": round(elapsed, 4),
            "selected_feature_count": fs_info.get("n_features_after"),
            "feature_selection_info": fs_info,
        }

    macro_f1_values = [m["val_macro_f1"] for m in results["models"].values()]
    results["mean_val_macro_f1"] = float(mean(macro_f1_values)) if macro_f1_values else 0.0
    return results


def _aggregate_runs(method_runs: list[dict]) -> dict:
    if not method_runs:
        return {"n_runs": 0, "models": {}, "mean_val_macro_f1": {"mean": 0.0, "std": 0.0}}

    model_names = method_runs[0]["models"].keys()
    aggregated_models = {}
    for model_name in model_names:
        rows = [run["models"][model_name] for run in method_runs]
        aggregated_models[model_name] = {
            "val_macro_f1": _mean_std([r["val_macro_f1"] for r in rows]),
            "val_accuracy": _mean_std([r["val_accuracy"] for r in rows]),
            "val_balanced_accuracy": _mean_std([r["val_balanced_accuracy"] for r in rows]),
            "runtime_seconds": _mean_std([r["runtime_seconds"] for r in rows]),
            "selected_feature_count": _mean_std_int_optional(
                [r["selected_feature_count"] for r in rows]
            ),
        }

    method_means = [run["mean_val_macro_f1"] for run in method_runs]
    return {
        "n_runs": len(method_runs),
        "models": aggregated_models,
        "mean_val_macro_f1": _mean_std(method_means),
    }


def _mean_std(values: Iterable[float]) -> dict:
    values = list(values)
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def _mean_std_int_optional(values: Iterable[int | None]) -> dict | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    stats = _mean_std([float(v) for v in valid])
    stats["mean"] = round(stats["mean"])
    return stats


def _build_deltas(aggregated: dict) -> dict:
    baseline = aggregated["baseline"]["mean_val_macro_f1"]["mean"]
    deltas = {}
    for method_name, agg in aggregated.items():
        deltas[method_name] = agg["mean_val_macro_f1"]["mean"] - baseline
    return deltas


def run_opensmile_comparison(
    output_dir: str = "./results",
    seeds: list[int] | None = None,
) -> dict:
    seeds = seeds or list(DEFAULT_SEEDS)
    out_root = ensure_dir(Path(output_dir) / "tuni_emotion_fs_comparison")

    all_runs: Dict[str, list[dict]] = {method: [] for method in METHOD_BENCHMARKS}
    for seed in seeds:
        print(f"\nRunning seed {seed}...")
        for method_name, benchmark in METHOD_BENCHMARKS.items():
            print(f"  Method: {method_name}")
            run_result = _run_benchmark_once(benchmark, method_name, out_root, seed)
            all_runs[method_name].append(run_result)

    aggregated = {
        method_name: _aggregate_runs(method_runs)
        for method_name, method_runs in all_runs.items()
    }
    deltas = _build_deltas(aggregated)

    summary = {
        "extractor": "opensmile",
        "seeds": seeds,
        "methods": all_runs,
        "aggregates": aggregated,
        "delta_mean_macro_f1_vs_baseline": deltas,
    }

    summary_path = out_root / "tuni_emotion_opensmile_fs_comparison.json"
    save_json(summary, summary_path)

    print(f"\nComparison saved to {summary_path}")
    for method_name in METHOD_BENCHMARKS:
        method_avg = aggregated[method_name]["mean_val_macro_f1"]["mean"]
        method_std = aggregated[method_name]["mean_val_macro_f1"]["std"]
        delta = deltas[method_name]
        print(
            f"{method_name:>18} macro-F1: {method_avg:.4f} ± {method_std:.4f} "
            f"(delta vs baseline {delta:+.4f})"
        )
    return summary


if __name__ == "__main__":
    run_opensmile_comparison()
