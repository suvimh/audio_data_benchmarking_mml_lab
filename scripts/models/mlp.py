from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.data.features import FeaturesData
from scripts.evaluation.metrics import (
    compute_metrics,
    print_metrics_summary,
    save_metrics_csv,
)
from scripts.utils import ensure_dir, save_pickle

MODEL_METADATA = {
    "name": "mlp",
    "supported_data_types": ["features"],
    "description": "Multi-Layer Perceptron classifier (sklearn)",
    "default_params": {
        "hidden_layer_sizes": (12,),
        "max_iter": 500,
        "random_state": 42,
    },
}


def train(data: FeaturesData, params: Dict[str, Any]) -> Pipeline:
    mlp_params = params.copy()
    steps = []
    if not data.preprocessed:
        steps.append(("scaler", StandardScaler()))
    steps.append(("mlp", MLPClassifier(**mlp_params)))
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

    params = bench_config.get_model_params("mlp")
    default_params = MODEL_METADATA["default_params"].copy()
    default_params.update(params)

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

    print_metrics_summary(train_metrics, "MLP (train)")
    print_metrics_summary(val_metrics, "MLP (validation)")

    metrics_path = output_dir / "metrics.csv"
    save_metrics_csv(val_metrics, metrics_path, "mlp", {"split": "validation"})
    save_metrics_csv(train_metrics, metrics_path, "mlp", {"split": "train"})

    model_path = output_dir / "model.pkl"
    save_pickle(model, model_path)

    return {
        "model_name": "mlp",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_path": str(model_path),
        "feature_selection_info": data.feature_selection_info,
    }
