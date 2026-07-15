from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.data.features import FeaturesData
from scripts.evaluation.metrics import compute_metrics
from scripts.tuning.svm_grid import search_svm_hyperparams
from scripts.utils import ensure_dir, save_pickle

MODEL_METADATA = {
    "name": "svm",
    "supported_data_types": ["features"],
    "description": "Support Vector Machine classifier (RBF kernel)",
    "default_params": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
    },
}


def _maybe_apply_smote(
    data: FeaturesData,
    benchmark_config: BenchmarkConfig,
) -> FeaturesData:
    if benchmark_config.imbalance_strategy != "smote":
        return data

    from scripts.data.imbalance import apply_smote

    X_train, y_train = apply_smote(
        data.X_train,
        data.y_train,
        random_state=benchmark_config.seed,
    )
    return FeaturesData(
        X_train=X_train,
        X_val=data.X_val,
        y_train=y_train,
        y_val=data.y_val,
        class_names=data.class_names,
        n_classes=data.n_classes,
        input_dim=data.input_dim,
        preprocessed=data.preprocessed,
        feature_selection_info=data.feature_selection_info,
    )


def train(data: FeaturesData, params: Dict[str, Any]) -> Pipeline:
    svm_params = {k: v for k, v in params.items()}
    steps = []
    if not data.preprocessed:
        steps.append(("scaler", StandardScaler()))
    steps.append(("svm", SVC(**svm_params)))
    pipeline = Pipeline(steps)
    pipeline.fit(data.X_train, data.y_train)
    return pipeline


def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    from scripts.data.features import load_features_data

    output_dir = ensure_dir(output_dir)
    data = load_features_data(dataset_config, bench_config)
    data = _maybe_apply_smote(data, bench_config)

    params = bench_config.get_model_params("svm")
    default_params = MODEL_METADATA["default_params"].copy()
    default_params.update(params)

    tuning_info: Optional[Dict[str, Any]] = None
    if bench_config.tune_svm:
        best_params, best_score = search_svm_hyperparams(
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            param_grid=bench_config.svm_param_grid,
            preprocessed=data.preprocessed,
            seed=bench_config.seed,
        )
        default_params.update(best_params)
        tuning_info = {
            "best_params": best_params,
            "val_macro_f1": best_score,
        }

    model = train(data, default_params)

    y_train_pred = model.predict(data.X_train)
    y_val_pred = model.predict(data.X_val)

    try:
        y_train_prob = model.predict_proba(data.X_train)
        y_val_prob = model.predict_proba(data.X_val)
    except Exception:
        y_train_prob = None
        y_val_prob = None

    train_metrics = compute_metrics(data.y_train, y_train_pred, y_train_prob, data.class_names)
    val_metrics = compute_metrics(data.y_val, y_val_pred, y_val_prob, data.class_names)

    from scripts.evaluation.metrics import print_metrics_summary, save_metrics_csv

    print_metrics_summary(train_metrics, "SVM (train)")
    print_metrics_summary(val_metrics, "SVM (validation)")

    metrics_path = output_dir / "metrics.csv"
    save_metrics_csv(val_metrics, metrics_path, "svm", {"split": "validation"})
    save_metrics_csv(train_metrics, metrics_path, "svm", {"split": "train"})

    model_path = output_dir / "model.pkl"
    save_pickle(model, model_path)

    result = {
        "model_name": "svm",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_path": str(model_path),
        "feature_selection_info": data.feature_selection_info,
    }
    if tuning_info is not None:
        result["svm_tuning_info"] = tuning_info
    return result


def run_on_features_data(
    data: FeaturesData,
    bench_config: BenchmarkConfig,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Train/evaluate SVM directly on pre-loaded feature matrices."""
    data = _maybe_apply_smote(data, bench_config)

    params = bench_config.get_model_params("svm")
    default_params = MODEL_METADATA["default_params"].copy()
    default_params.update(params)

    tuning_info: Optional[Dict[str, Any]] = None
    if bench_config.tune_svm:
        best_params, best_score = search_svm_hyperparams(
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            param_grid=bench_config.svm_param_grid,
            preprocessed=data.preprocessed,
            seed=bench_config.seed,
        )
        default_params.update(best_params)
        tuning_info = {
            "best_params": best_params,
            "val_macro_f1": best_score,
        }

    model = train(data, default_params)
    y_val_pred = model.predict(data.X_val)
    try:
        y_val_prob = model.predict_proba(data.X_val)
    except Exception:
        y_val_prob = None

    val_metrics = compute_metrics(data.y_val, y_val_pred, y_val_prob, data.class_names)

    result: Dict[str, Any] = {
        "model_name": "svm",
        "val_metrics": val_metrics,
        "feature_selection_info": data.feature_selection_info,
        "model": model,
    }
    if tuning_info is not None:
        result["svm_tuning_info"] = tuning_info
    if output_dir is not None:
        output_dir = ensure_dir(output_dir)
        save_pickle(model, output_dir / "model.pkl")
        from scripts.evaluation.metrics import save_metrics_csv

        save_metrics_csv(val_metrics, output_dir / "metrics.csv", "svm", {"split": "validation"})
        result["model_path"] = str(output_dir / "model.pkl")
    return result
