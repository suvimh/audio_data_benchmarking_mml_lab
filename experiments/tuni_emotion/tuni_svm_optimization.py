"""
SVM-focused grid search for TUNI emotion benchmarks.

Usage:
    python -m experiments.tuni_emotion.tuni_svm_optimization
    python -m experiments.tuni_emotion.tuni_svm_optimization --phase 1 --quick
    python -m experiments.tuni_emotion.tuni_svm_optimization --phase 2 --workers 4
    python -m experiments.tuni_emotion.tuni_svm_optimization --phase 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.config.experiment_config import ExperimentFilters
from scripts.config.partition_config import PartitionConfig
from scripts.feature_selection.base import FeatureSelectionConfig
from scripts.evaluation.metrics import compute_metrics
from scripts.models import svm as svm_model
from scripts.run_benchmark import resolve_dataset
from scripts.utils import ensure_dir, save_json, set_seed

from experiments.tuni_emotion.tuni_data_featureset_embedding_configs import (
    OPENSMILE_VARIANTS,
    make_clap_embedding_config,
    make_opensmile_cleaned_embedding_config,
    make_opensmile_embedding_config,
    make_opensmile_variant_config,
    make_whisper_embedding_config,
)
from experiments.tuni_emotion.tuni_experiments_constants import (
    TUNI_EMOTIONS,
    TRAIN_SINGERS,
    VAL_SINGERS,
)

DEFAULT_SEEDS = [42, 52, 62]
K_VALUES_FULL = [25, 50, 100, 200, 500, 1000, 2000, 6373]
K_VALUES_QUICK = [50, 100, 200, 500, 6373]
FS_METHODS = ["filter", "generalized_fisher", "wrapper", "embedded"]
GFS_REDUNDANCY_WEIGHTS = [0.0, 0.05, 0.1, 0.2]

PARTITION = PartitionConfig(
    name="tuni_singer_independent",
    train_singer_ids=TRAIN_SINGERS,
    test_singer_ids=VAL_SINGERS,
    singer_column="singer",
)
FILTERS = ExperimentFilters(gender="mixed", include_labels=TUNI_EMOTIONS)


@dataclass
class ComboSpec:
    combo_id: str
    phase: int
    extractor: str
    frame_duration: float = 3.0
    dataset_name: str = "opensmile"
    parquet_file: Optional[str] = None
    data_dir: Optional[str] = None
    feature_type: Optional[str] = None
    fs_method: str = "filter"
    fs_k: int = 100
    fs_extra: Dict[str, Any] = field(default_factory=dict)
    tune_svm: bool = False
    imbalance_strategy: Optional[str] = None
    late_fusion_members: List[str] = field(default_factory=list)
    concat_members: List[str] = field(default_factory=list)
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def _macro_f1(metrics: dict) -> float:
    report = metrics.get("classification_report", {})
    return float(report.get("macro avg", {}).get("f1-score", 0.0))


def _primary_metric(metrics: dict) -> float:
    return max(
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("balanced_accuracy", 0.0)),
        _macro_f1(metrics),
    )


def _build_fs_config(spec: ComboSpec) -> FeatureSelectionConfig:
    if spec.fs_method == "none":
        return FeatureSelectionConfig(enabled=False)

    params: Dict[str, Any] = {"random_state": spec.seed}
    if spec.fs_k < 6373:
        if spec.fs_method == "filter":
            params.update({"variance_threshold": 0.0, "k": spec.fs_k})
        elif spec.fs_method == "generalized_fisher":
            params.update(
                {
                    "k": spec.fs_k,
                    "gamma": 1e-6,
                    "redundancy_weight": spec.fs_extra.get("redundancy_weight", 0.1),
                    "max_iter": 100,
                    "tol": 1e-6,
                }
            )
        elif spec.fs_method == "wrapper":
            params.update(
                {
                    "n_features_to_select": spec.fs_k,
                    "estimator": spec.fs_extra.get("estimator", "svm"),
                    "step": spec.fs_extra.get("step", 25 if spec.fs_k <= 200 else 50),
                }
            )
        elif spec.fs_method == "embedded":
            params.update({"max_features": spec.fs_k, "C": spec.fs_extra.get("C", 0.1)})
        else:
            raise ValueError(f"Unknown fs_method: {spec.fs_method}")

    params.update({k: v for k, v in spec.fs_extra.items() if k not in params})
    return FeatureSelectionConfig(enabled=True, method=spec.fs_method, params=params)


def _build_benchmark(spec: ComboSpec) -> BenchmarkConfig:
    svm_grid = spec.fs_extra.get("svm_param_grid")
    if svm_grid is None and spec.tune_svm:
        from scripts.tuning.svm_grid import FAST_SVM_PARAM_GRID

        svm_grid = FAST_SVM_PARAM_GRID
    return BenchmarkConfig(
        name="svm_opt",
        models=["svm"],
        model_params={"svm": {}},
        feature_selection=_build_fs_config(spec),
        tune_svm=spec.tune_svm,
        svm_param_grid=svm_grid,
        imbalance_strategy=spec.imbalance_strategy,
        seed=spec.seed,
        output_dir="./metrics/svm_opt",
    )


def _resolve_dataset_for_spec(spec: ComboSpec) -> DatasetConfig:
    if spec.extractor == "opensmile":
        dataset = make_opensmile_embedding_config(frame_duration=spec.frame_duration)
    elif spec.extractor == "opensmile_cleaned":
        dataset = make_opensmile_cleaned_embedding_config(frame_duration=spec.frame_duration)
    elif spec.extractor == "opensmile_cleaned_aggressive":
        dataset = make_opensmile_cleaned_embedding_config(
            frame_duration=spec.frame_duration, aggressive=True
        )
    elif spec.extractor == "whisper":
        dataset = make_whisper_embedding_config(frame_duration=spec.frame_duration)
    elif spec.extractor == "clap":
        dataset = make_clap_embedding_config(frame_duration=spec.frame_duration)
    elif spec.extractor.startswith("opensmile_"):
        variant = spec.extractor.replace("opensmile_", "").replace("_", "-")
        dataset = make_opensmile_variant_config(
            variant=variant,
            frame_duration=spec.frame_duration,
            data_dir=spec.data_dir,
        )
    else:
        raise ValueError(f"Unknown extractor: {spec.extractor}")

    if spec.parquet_file:
        dataset.parquet_file = spec.parquet_file
    if spec.data_dir:
        dataset.data_dir = spec.data_dir
    if spec.feature_type:
        dataset.feature_type = spec.feature_type
    if spec.dataset_name:
        dataset.name = spec.dataset_name

    return resolve_dataset(
        dataset,
        PARTITION,
        FILTERS,
        frame_duration=spec.frame_duration,
        overlap=0.25,
    )


def _member_dataset(member: str, frame_duration: float) -> DatasetConfig:
    mapping = {
        "opensmile": lambda: make_opensmile_embedding_config(frame_duration=frame_duration),
        "whisper": lambda: make_whisper_embedding_config(frame_duration=frame_duration),
        "clap": lambda: make_clap_embedding_config(frame_duration=frame_duration),
    }
    if member not in mapping:
        raise ValueError(f"Unknown fusion member: {member}")
    return resolve_dataset(
        mapping[member](),
        PARTITION,
        FILTERS,
        frame_duration=frame_duration,
        overlap=0.25,
    )


def _run_late_fusion(spec: ComboSpec, benchmark: BenchmarkConfig) -> dict:
    from scripts.data.features import load_features_data

    models = []
    member_metrics = {}
    member_probs = []
    y_val = None
    class_names = None

    for member in spec.late_fusion_members:
        member_spec = deepcopy(spec)
        member_spec.extractor = member
        dataset = _resolve_dataset_for_spec(member_spec)
        member_benchmark = deepcopy(benchmark)
        if member != "opensmile":
            member_benchmark.feature_selection = FeatureSelectionConfig(enabled=False)
        data = load_features_data(dataset, member_benchmark)
        result = svm_model.run_on_features_data(data, member_benchmark)
        models.append(result["model"])
        member_probs.append(result["model"].predict_proba(data.X_val))
        member_metrics[member] = {
            "val_macro_f1": _macro_f1(result["val_metrics"]),
            "val_accuracy": float(result["val_metrics"].get("accuracy", 0.0)),
            "val_balanced_accuracy": float(result["val_metrics"].get("balanced_accuracy", 0.0)),
        }
        if y_val is None:
            y_val = data.y_val
            class_names = data.class_names

    avg_prob = np.mean(member_probs, axis=0)
    y_pred = np.argmax(avg_prob, axis=1)
    val_metrics = compute_metrics(y_val, y_pred, avg_prob, class_names)
    return {
        "val_metrics": val_metrics,
        "member_metrics": member_metrics,
        "feature_selection_info": None,
    }


def _run_concat(spec: ComboSpec, benchmark: BenchmarkConfig) -> dict:
    from scripts.data.concat_features import load_concat_features_data

    configs = [_member_dataset(member, spec.frame_duration) for member in spec.concat_members]
    data = load_concat_features_data(configs, benchmark)
    result = svm_model.run_on_features_data(data, benchmark)
    return {
        "val_metrics": result["val_metrics"],
        "feature_selection_info": result.get("feature_selection_info"),
        "svm_tuning_info": result.get("svm_tuning_info"),
    }


def _parquet_exists(spec: ComboSpec) -> bool:
    try:
        dataset = _resolve_dataset_for_spec(spec)
        path = Path(dataset.data_dir) / dataset.parquet_file
        return path.exists()
    except Exception:
        return False


def run_combo(spec: ComboSpec, output_dir: Path) -> dict:
    started = time.perf_counter()
    combo_out = ensure_dir(output_dir / spec.combo_id / f"seed_{spec.seed}")
    result_path = combo_out / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            cached = json.load(f)
        cached["cached"] = True
        return cached

    set_seed(spec.seed)
    benchmark = _build_benchmark(spec)

    try:
        if spec.late_fusion_members:
            run_result = _run_late_fusion(spec, benchmark)
        elif spec.concat_members:
            run_result = _run_concat(spec, benchmark)
        else:
            dataset = _resolve_dataset_for_spec(spec)
            run_result = svm_model.run(dataset, benchmark, combo_out)

        val_metrics = run_result["val_metrics"]
        payload = {
            "combo_id": spec.combo_id,
            "spec": spec.to_dict(),
            "val_macro_f1": _macro_f1(val_metrics),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_balanced_accuracy": float(val_metrics.get("balanced_accuracy", 0.0)),
            "primary_metric": _primary_metric(val_metrics),
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "feature_selection_info": run_result.get("feature_selection_info"),
            "svm_tuning_info": run_result.get("svm_tuning_info"),
            "member_metrics": run_result.get("member_metrics"),
            "cached": False,
        }
        save_json(payload, result_path)
        return payload
    except Exception as exc:
        payload = {
            "combo_id": spec.combo_id,
            "spec": spec.to_dict(),
            "error": str(exc),
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "primary_metric": 0.0,
            "cached": False,
        }
        save_json(payload, result_path)
        return payload


def _combo_id(*parts: Any) -> str:
    raw = "__".join(str(p) for p in parts)
    digest = hashlib.md5(raw.encode()).hexdigest()[:10]
    safe = raw.replace("/", "-").replace(" ", "_")[:80]
    return f"{safe}__{digest}"


def build_phase1_grid(quick: bool = False) -> List[ComboSpec]:
    combos: List[ComboSpec] = []
    k_values = K_VALUES_QUICK if quick else K_VALUES_FULL

    for fs_method in FS_METHODS:
        for k in k_values:
            fs_k = k
            fs_method_use = fs_method
            if k == 6373:
                fs_method_use = "none"
                fs_k = 6373
            if fs_method in {"wrapper", "embedded"} and k >= 1000:
                continue
            if quick and fs_method in {"wrapper", "embedded"} and k > 200:
                continue

            extra_variants = [{}]
            if fs_method == "generalized_fisher":
                extra_variants = [{"redundancy_weight": w} for w in GFS_REDUNDANCY_WEIGHTS]

            for fs_extra in extra_variants:
                combo = ComboSpec(
                    combo_id=_combo_id("p1", "opensmile", fs_method_use, fs_k, fs_extra),
                    phase=1,
                    extractor="opensmile",
                    fs_method=fs_method_use,
                    fs_k=fs_k,
                    fs_extra=fs_extra,
                )
                combos.append(combo)

    for extractor in ["whisper", "clap"]:
        combos.append(
            ComboSpec(
                combo_id=_combo_id("p1", extractor, "none"),
                phase=1,
                extractor=extractor,
                fs_method="none",
                fs_k=0,
            )
        )

    for extractor in ["opensmile", "opensmile_cleaned", "opensmile_cleaned_aggressive"]:
        combos.append(
            ComboSpec(
                combo_id=_combo_id("p1", extractor, "gfs", 200),
                phase=1,
                extractor=extractor,
                fs_method="generalized_fisher",
                fs_k=200,
            )
        )

    if not quick:
        combos.append(
            ComboSpec(
                combo_id=_combo_id("p1", "opensmile", "0.5s", "gfs", 200),
                phase=1,
                extractor="opensmile",
                frame_duration=0.5,
                fs_method="generalized_fisher",
                fs_k=200,
            )
        )
        combos.append(
            ComboSpec(
                combo_id=_combo_id("p1", "opensmile", "0.5s", "filter", 200),
                phase=1,
                extractor="opensmile",
                frame_duration=0.5,
                fs_method="filter",
                fs_k=200,
            )
        )
    return combos


def build_phase2_grid(top_specs: List[ComboSpec], quick: bool = False) -> List[ComboSpec]:
    combos: List[ComboSpec] = []
    fs_choices = top_specs[:3] if top_specs else [
        ComboSpec(combo_id="default", phase=1, extractor="opensmile", fs_method="generalized_fisher", fs_k=200)
    ]

    for base in fs_choices:
        tuned = deepcopy(base)
        tuned.phase = 2
        tuned.tune_svm = True
        tuned.combo_id = _combo_id("p2", "tune", base.extractor, base.fs_method, base.fs_k)
        combos.append(tuned)

    for base in fs_choices:
        combos.append(
            ComboSpec(
                combo_id=_combo_id("p2", "late_fusion", base.fs_method, base.fs_k),
                phase=2,
                extractor="late_fusion",
                fs_method=base.fs_method,
                fs_k=base.fs_k,
                fs_extra=base.fs_extra,
                tune_svm=True,
                late_fusion_members=["opensmile", "whisper", "clap"],
            )
        )
        combos.append(
            ComboSpec(
                combo_id=_combo_id("p2", "concat", base.fs_method, base.fs_k),
                phase=2,
                extractor="concat",
                fs_method=base.fs_method,
                fs_k=max(base.fs_k, 150),
                fs_extra=base.fs_extra,
                tune_svm=True,
                concat_members=["opensmile", "whisper", "clap"],
            )
        )

    for base in fs_choices[:2]:
        combos.append(
            ComboSpec(
                combo_id=_combo_id("p2", "smote", base.fs_method, base.fs_k),
                phase=2,
                extractor=base.extractor,
                fs_method=base.fs_method,
                fs_k=base.fs_k,
                fs_extra=base.fs_extra,
                tune_svm=True,
                imbalance_strategy="smote",
            )
        )
    return combos


def build_phase3_grid(quick: bool = False) -> List[ComboSpec]:
    combos: List[ComboSpec] = []
    k_values = [100, 200] if quick else [100, 200, 500]
    for variant in ["emobase", "is09", "egemaps"]:
        for fs_method in ["filter", "generalized_fisher"]:
            for k in k_values:
                spec = ComboSpec(
                    combo_id=_combo_id("p3", variant, fs_method, k),
                    phase=3,
                    extractor=f"opensmile_{variant}",
                    dataset_name=f"opensmile_{variant}",
                    fs_method=fs_method,
                    fs_k=k,
                )
                if _parquet_exists(spec):
                    combos.append(spec)
    return combos


def _filter_runnable(combos: Iterable[ComboSpec]) -> List[ComboSpec]:
    runnable = []
    for spec in combos:
        if spec.late_fusion_members or spec.concat_members:
            runnable.append(spec)
            continue
        if _parquet_exists(spec):
            runnable.append(spec)
    return runnable


def _aggregate_seed_results(results: List[dict]) -> dict:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"n_runs": 0, "primary_metric": {"mean": 0.0, "std": 0.0}}

    def stats(key: str) -> dict:
        values = [float(r[key]) for r in valid]
        if len(values) == 1:
            return {"mean": values[0], "std": 0.0}
        return {"mean": float(mean(values)), "std": float(pstdev(values))}

    return {
        "n_runs": len(valid),
        "primary_metric": stats("primary_metric"),
        "val_macro_f1": stats("val_macro_f1"),
        "val_accuracy": stats("val_accuracy"),
        "val_balanced_accuracy": stats("val_balanced_accuracy"),
        "spec": valid[0]["spec"],
    }


def _update_leaderboard(out_root: Path, new_results: List[dict]) -> dict:
    leaderboard_path = out_root / "leaderboard.json"
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {"combos": {}}

    grouped: Dict[str, List[dict]] = {}
    for result in new_results:
        combo_id = result["combo_id"]
        grouped.setdefault(combo_id, []).append(result)

    for combo_id, rows in grouped.items():
        leaderboard["combos"][combo_id] = _aggregate_seed_results(rows)

    ranked = sorted(
        leaderboard["combos"].items(),
        key=lambda item: item[1]["primary_metric"]["mean"],
        reverse=True,
    )
    leaderboard["ranked"] = [
        {"combo_id": combo_id, **stats} for combo_id, stats in ranked
    ]
    leaderboard["best"] = leaderboard["ranked"][0] if leaderboard["ranked"] else None
    save_json(leaderboard, leaderboard_path)

    best_path = out_root / "best_config.json"
    if leaderboard.get("best"):
        save_json(leaderboard["best"], best_path)
    return leaderboard


def _top_specs_from_leaderboard(out_root: Path, n: int = 3) -> List[ComboSpec]:
    leaderboard_path = out_root / "leaderboard.json"
    if not leaderboard_path.exists():
        return []
    with open(leaderboard_path) as f:
        leaderboard = json.load(f)
    specs = []
    for entry in leaderboard.get("ranked", [])[:n]:
        spec_dict = entry.get("spec") or {}
        if spec_dict:
            specs.append(ComboSpec(**spec_dict))
    return specs


def _run_combo_worker(args: tuple[ComboSpec, str]) -> dict:
    spec, output_dir = args
    return run_combo(spec, Path(output_dir))


def rebuild_leaderboard_from_disk(output_dir: str = "./results/tuni_svm_opt") -> dict:
    out_root = Path(output_dir)
    results: List[dict] = []
    for result_path in out_root.glob("**/result.json"):
        with open(result_path) as f:
            results.append(json.load(f))
    return _update_leaderboard(out_root, results)


def run_phase(
    phase: int,
    *,
    output_dir: str = "./results/tuni_svm_opt",
    seeds: Optional[List[int]] = None,
    workers: int = 1,
    quick: bool = False,
) -> dict:
    out_root = ensure_dir(Path(output_dir))
    seeds = seeds or [DEFAULT_SEEDS[0]]

    if phase == 1:
        combos = build_phase1_grid(quick=quick)
    elif phase == 2:
        top_specs = _top_specs_from_leaderboard(out_root)
        combos = build_phase2_grid(top_specs, quick=quick)
    elif phase == 3:
        combos = build_phase3_grid(quick=quick)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    combos = _filter_runnable(combos)
    jobs: List[ComboSpec] = []
    for spec in combos:
        for seed in seeds:
            job = deepcopy(spec)
            job.seed = seed
            job.combo_id = _combo_id(spec.combo_id, f"seed{seed}")
            jobs.append(job)

    print(f"Phase {phase}: running {len(jobs)} jobs with {workers} worker(s)")
    results: List[dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_combo_worker, (job, str(out_root))): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                results.append(result)
                metric = result.get("primary_metric", 0.0)
                status = "ERROR" if "error" in result else "OK"
                print(f"  [{status}] {job.combo_id}: primary={metric:.4f}")
    else:
        for job in jobs:
            result = run_combo(job, out_root)
            results.append(result)
            metric = result.get("primary_metric", 0.0)
            status = "ERROR" if "error" in result else "OK"
            print(f"  [{status}] {job.combo_id}: primary={metric:.4f}")

    leaderboard = _update_leaderboard(out_root, results)
    if leaderboard.get("best"):
        best = leaderboard["best"]
        print(
            f"\nBest so far: {best['combo_id']} "
            f"primary={best['primary_metric']['mean']:.4f}"
        )
    return leaderboard


def run_multiseed_confirmation(
    output_dir: str = "./results/tuni_svm_opt",
    seeds: Optional[List[int]] = None,
    workers: int = 1,
) -> dict:
    out_root = Path(output_dir)
    top_specs = _top_specs_from_leaderboard(out_root, n=5)
    if not top_specs:
        raise RuntimeError("No top specs found. Run phase 1 first.")

    seeds = seeds or DEFAULT_SEEDS
    jobs: List[ComboSpec] = []
    for spec in top_specs:
        for seed in seeds:
            job = deepcopy(spec)
            job.seed = seed
            job.tune_svm = True
            job.combo_id = _combo_id("confirm", spec.combo_id, seed)
            jobs.append(job)

    results = []
    for job in jobs:
        results.append(run_combo(job, out_root))
    return _update_leaderboard(out_root, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="TUNI SVM optimization grid")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--output-dir", default="./results/tuni_svm_opt")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="Re-run top configs across default seeds")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--rebuild-leaderboard", action="store_true")
    args = parser.parse_args()

    if args.rebuild_leaderboard:
        rebuild_leaderboard_from_disk(args.output_dir)
        return

    if args.confirm:
        run_multiseed_confirmation(
            output_dir=args.output_dir,
            seeds=args.seeds,
            workers=args.workers,
        )
        return

    run_phase(
        args.phase,
        output_dir=args.output_dir,
        seeds=args.seeds,
        workers=args.workers,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
