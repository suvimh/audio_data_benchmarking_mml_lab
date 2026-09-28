'''
    Orchestrator for the feature-selection variant of the traditional ML
    benchmark.  Runs a sweep over:

        embedding × model × feature-selection-method × n_features

    For every combination it:
      1. loads the full embedding data (per model-embedding pair),
      2. fits the selector (RFE or SelectKBest) on the TRAINING split only,
      3. transforms both splits with the fitted selector,
      4. trains the classifier and computes the full metric set,
      5. appends one row to a cumulative CSV (safe to interrupt and resume).

    Reuses the exact same model implementations / hyperparams as the
    original benchmark so the feature counts are directly comparable.
'''
from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from scripts.config.feature_selection_config import FeatureSelectionExperimentConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.data.features import FeaturesData, load_features_data
from scripts.evaluation.metrics import (
    compute_metrics,
    log_environment_snapshot,
    save_feature_selection_metrics_csv,
)
from scripts.run_benchmark import load_config, resolve_dataset
from scripts.utils import ensure_dir, save_json, save_pickle, set_seed
from scripts.registry import get_model


# ---------------------------------------------------------------------------
# Feature selection methods
# ---------------------------------------------------------------------------

_METHOD_ALIASES = {
    "rfe": "rfe",
    "RFE": "rfe",
    "recursive_feature_elimination": "rfe",
    "select_k_best": "select_k_best",
    "selectkbest": "select_k_best",
    "select-k-best": "select_k_best",
    "k_best": "select_k_best",
}

_METHOD_LABELS = {
    "rfe": "RFE",
    "select_k_best": "SelectKBest",
}


def _normalise_methods(methods: List[str]) -> List[str]:
    out = []
    for m in methods:
        key = _METHOD_ALIASES.get(str(m).strip())
        if key is None:
            raise ValueError(
                f"Unsupported feature selection method '{m}'. "
                f"Supported: {sorted(set(_METHOD_ALIASES.values()))}"
            )
        if key not in out:
            out.append(key)
    return out


def _make_rfe_estimator(config: FeatureSelectionExperimentConfig):
    from sklearn.ensemble import RandomForestClassifier

    kind = (config.rfe_estimator or "rf").strip().lower()
    params: Dict[str, Any] = dict(config.rfe_estimator_params or {})
    params.setdefault("random_state", config.seed)

    if kind in ("rf", "random_forest", "randomforest"):
        params.setdefault("n_estimators", 100)
        return RandomForestClassifier(**params)
    if kind in ("svm", "linear_svc", "linearsvc"):
        from sklearn.svm import LinearSVC
        return LinearSVC(**params)
    raise ValueError(
        f"Unsupported RFE estimator '{config.rfe_estimator}'. "
        "Use 'rf' or 'svm'."
    )


def _make_feature_selector(
    method: str, n_features: int, config: FeatureSelectionExperimentConfig
):
    if method == "select_k_best":
        from sklearn.feature_selection import SelectKBest, f_classif
        return SelectKBest(score_func=f_classif, k=n_features)
    if method == "rfe":
        from sklearn.feature_selection import RFE
        return RFE(
            estimator=_make_rfe_estimator(config),
            n_features_to_select=n_features,
            step=config.rfe_step,
        )
    raise ValueError(f"Unsupported method: {method}")


# ---------------------------------------------------------------------------
# Model helpers (identical hyper-parameter merge to the original benchmark)
# ---------------------------------------------------------------------------

def _load_model_module(model_name: str):
    try:
        return importlib.import_module(f"scripts.models.{model_name}")
    except ImportError:
        raise ImportError(
            f"Could not import scripts.models.{model_name}. "
            "Only 'features'-type models are supported here."
        )


def _merged_model_params(config: FeatureSelectionExperimentConfig, model_name: str) -> Dict:
    module = _load_model_module(model_name)
    metadata = getattr(module, "MODEL_METADATA", {}) or {}
    defaults = dict(metadata.get("default_params", {}) or {})
    defaults.update(config.get_model_params(model_name))
    return defaults


def _model_key_parameters(model_name: str, params: Dict) -> str:
    if model_name == "knn":
        return f"k={params.get('n_neighbors', 5)}, Euclidean distance"
    if model_name == "rf":
        return f"{params.get('n_estimators', 100)} trees"
    if model_name == "svm":
        return (
            f"kernel={params.get('kernel', 'rbf')}, "
            f"C={params.get('C', 1.0)}, "
            f"class_weight={params.get('class_weight', 'None')}"
        )
    if model_name == "mlp":
        hidden = params.get("hidden_layer_sizes", (12,))
        units = hidden[0] if isinstance(hidden, (tuple, list)) else hidden
        return (
            f"1 hidden layer ({units} units), "
            f"lr={params.get('learning_rate_init')}, "
            f"momentum={params.get('momentum')}, "
            f"max {params.get('max_iter')} iters"
        )
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# Single combination runner
# ---------------------------------------------------------------------------

def _run_combination(
    config: FeatureSelectionExperimentConfig,
    dataset: DatasetConfig,
    model_name: str,
    full: FeaturesData,
    method: str,
    n_features: int,
    run_out: Path,
) -> Dict[str, Any]:
    set_seed(config.seed)

    selector = _make_feature_selector(method, n_features, config)
    selector.fit(full.X_train, full.y_train)
    selected_indices = np.asarray(selector.get_support(indices=True)).astype(int)

    reduced = FeaturesData(
        X_train=selector.transform(full.X_train),
        y_train=full.y_train,
        X_val=selector.transform(full.X_val),
        y_val=full.y_val,
        class_names=full.class_names,
        input_dim=n_features,
        n_classes=full.n_classes,
        n_timesteps=None,
    )

    module = _load_model_module(model_name)
    params = _merged_model_params(config, model_name)

    from scripts.evaluation.metrics_utils import ResourceTracker
    tracker = ResourceTracker()
    tracker.start()
    start_time = time.perf_counter()

    model = module.train(reduced, dict(params))

    y_train_pred = model.predict(reduced.X_train)
    y_val_pred = model.predict(reduced.X_val)
    try:
        y_train_prob = model.predict_proba(reduced.X_train)
        y_val_prob = model.predict_proba(reduced.X_val)
    except Exception:
        y_train_prob = None
        y_val_prob = None

    elapsed = time.perf_counter() - start_time
    resource_stats = tracker.stop()

    train_metrics = compute_metrics(
        reduced.y_train, y_train_pred, y_train_prob, reduced.class_names
    )
    val_metrics = compute_metrics(
        reduced.y_val, y_val_pred, y_val_prob, reduced.class_names
    )

    total_features = full.X_train.shape[1]
    method_label = _METHOD_LABELS[method]

    metrics_path = Path(run_out) / "metrics.csv"
    save_feature_selection_metrics_csv(
        metrics_path,
        model_name,
        dataset.name,
        method_label,
        n_features,
        selected_indices,
        train_metrics,
        val_metrics,
        total_features=total_features,
        extra={
            "Epochs": "-",
            "Batch Size": "-",
            "Train+Eval Time (s)": round(elapsed, 4),
            "Key Parameters": _model_key_parameters(model_name, params),
            "Input Features": n_features,
            "Input Feature Type": dataset.feature_type or dataset.name,
            "frame_duration": config.frame_duration,
            "overlap": config.overlap,
            **resource_stats,
        },
    )

    save_pickle(model, run_out / "model.pkl")
    save_pickle(selector, run_out / "feature_selector.pkl")

    return {
        "model_name": model_name,
        "dataset": dataset.name,
        "method": method_label,
        "n_features": n_features,
        "selected_features": selected_indices.tolist(),
        "total_features": total_features,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "resource_stats": resource_stats,
        "train_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_feature_selection_experiment(
    config: FeatureSelectionExperimentConfig | str | Path,
) -> Dict[str, Any]:
    if not isinstance(config, FeatureSelectionExperimentConfig):
        config = load_config(config, FeatureSelectionExperimentConfig)

    experiment_out = ensure_dir(Path(config.output_dir) / config.name)
    log_environment_snapshot(experiment_out / "environment.txt")

    metrics_path = experiment_out / f"{config.name}_feature_selection_metrics.csv"

    methods = _normalise_methods(config.feature_selection_methods)
    n_features_list = [int(n) for n in config.n_features_list]

    print(f"\n{'=' * 60}")
    print(f"Feature Selection Experiment: {config.name}")
    print(f"  Methods:      {', '.join(_METHOD_LABELS[m] for m in methods)}")
    print(f"  N features:   {n_features_list}")
    print(f"  Pairs:        {len(config.pairs)}")
    for pair in config.pairs:
        print(f"    - {pair.dataset.name} × {', '.join(pair.models)}")
    print(f"{'=' * 60}\n")

    all_results: Dict[str, Any] = {}

    for pair_idx, pair in enumerate(config.pairs, start=1):
        dataset = pair.dataset
        resolved = (
            resolve_dataset(
                dataset,
                config.partition,
                config.filters,
                frame_duration=config.frame_duration,
                overlap=config.overlap,
            )
            if config.partition is not None
            else dataset
        )

        print(f"\n{'─' * 60}")
        print(f"Pair {pair_idx}/{len(config.pairs)}: {dataset.name}")
        print(f"{'─' * 60}")

        full = load_features_data(resolved)
        total_features = full.X_train.shape[1]

        effective_n = [
            n for n in n_features_list if 0 < n < total_features
        ]
        if not effective_n:
            print(
                f"  Skipping {dataset.name}: no requested feature count "
                f"({n_features_list}) is below its {total_features} features."
            )
            continue
        skipped = [n for n in n_features_list if n not in effective_n]
        if skipped:
            print(
                f"  Note: skipping n_features {skipped} (>= {total_features} "
                f"total features for {dataset.name}). Using {effective_n}."
            )

        for model_name in pair.models:
            model_info = get_model(model_name)
            if "features" not in model_info["supported_data_types"]:
                print(
                    f"  Skipping model '{model_name}' on {dataset.name}: "
                    "it does not support 'features' data."
                )
                continue
            _load_model_module(model_name)

            for method in methods:
                method_label = _METHOD_LABELS[method]
                for n_features in effective_n:
                    print(
                        f"  [{dataset.name}] {model_name} / {method_label} "
                        f"/ {n_features} features ..."
                    )

                    run_out = ensure_dir(
                        experiment_out
                        / dataset.name
                        / model_name
                        / f"{method_label.lower()}_{n_features}features"
                    )

                    result = _run_combination(
                        config, dataset, model_name, full, method,
                        n_features, run_out,
                    )

                    key = f"{dataset.name}/{model_name}/{method_label}/{n_features}"
                    all_results[key] = {
                        "model": model_name,
                        "dataset": dataset.name,
                        "method": method_label,
                        "n_features": n_features,
                        "val_balanced_accuracy": result["val_metrics"].get(
                            "balanced_accuracy"
                        ),
                        "val_accuracy": result["val_metrics"].get("accuracy"),
                    }

                    # write a cumulative row (one per combination) after every run
                    save_feature_selection_metrics_csv(
                        metrics_path,
                        model_name,
                        dataset.name,
                        method_label,
                        n_features,
                        result["selected_features"],
                        result["train_metrics"],
                        result["val_metrics"],
                        total_features=result["total_features"],
                        extra={
                            "Epochs": "-",
                            "Batch Size": "-",
                            "Train+Eval Time (s)": round(result["train_time_s"], 4),
                            "Key Parameters": _model_key_parameters(
                                model_name, _merged_model_params(config, model_name)
                            ),
                            "Input Features": n_features,
                            "Input Feature Type": dataset.feature_type or dataset.name,
                            "frame_duration": config.frame_duration,
                            "overlap": config.overlap,
                            **result["resource_stats"],
                        },
                    )
                    print(
                        f"    val bacc = "
                        f"{result['val_metrics'].get('balanced_accuracy', float('nan')):.4f}"
                    )

    summary_path = experiment_out / f"{config.name}_summary.json"
    save_json(
        {
            "experiment": config.name,
            "feature_selection_methods": [_METHOD_LABELS[m] for m in methods],
            "n_features_list": n_features_list,
            "results": all_results,
        },
        summary_path,
    )
    print(f"\nSummary saved to {summary_path}")
    print(f"Metrics CSV saved to {metrics_path}")

    # Graphs of validation balanced accuracy vs feature count
    try:
        from scripts.evaluation.visualization import plot_feature_selection_curves
        plot_feature_selection_curves(metrics_path, experiment_out)
    except Exception as exc:  # plotting must not take down finished runs
        print(f"Warning: could not generate feature-selection plots: {exc}")

    return all_results