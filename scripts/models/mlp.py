from __future__ import annotations

import time
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
    save_metrics_csv,
)
from scripts.utils import ensure_dir, save_pickle

MODEL_METADATA = {
    "name": "mlp",
    "supported_data_types": ["features"],
    "description": "Multi-Layer Perceptron classifier (sklearn)",
    "default_params": {
        "hidden_layer_sizes": (12,),
        "max_iter": 2000,
        "random_state": 42,
        "momentum": 0.5,
        "learning_rate_init": 0.01,
    },
}


def train(data: FeaturesData, params: Dict[str, Any]) -> Pipeline:
    mlp_params = params.copy()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(**mlp_params)),
    ])
    pipeline.fit(data.X_train, data.y_train)
    return pipeline


def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    from scripts.data.features import load_features_data

    output_dir = ensure_dir(output_dir)
    start_time = time.perf_counter()
    data = load_features_data(dataset_config)

    params = bench_config.get_model_params("mlp")
    default_params = MODEL_METADATA["default_params"].copy()
    default_params.update(params)

    from scripts.evaluation.metrics_utils import ResourceTracker
    resource_tracker = ResourceTracker()
    resource_tracker.start()

    model = train(data, default_params)

    y_train_pred = model.predict(data.X_train)
    y_val_pred = model.predict(data.X_val)

    try:
        y_train_prob = model.predict_proba(data.X_train)
        y_val_prob = model.predict_proba(data.X_val)
    except Exception:
        y_train_prob = None
        y_val_prob = None

    resource_stats = resource_tracker.stop()

    train_metrics = compute_metrics(data.y_train, y_train_pred, y_train_prob, data.class_names)
    val_metrics = compute_metrics(data.y_val, y_val_pred, y_val_prob, data.class_names)


    metrics_path = Path(dataset_config.metrics_path or output_dir / "metrics.csv")
    elapsed = time.perf_counter() - start_time
    n_features = data.X_train.shape[1]
    hidden = default_params["hidden_layer_sizes"]
    save_metrics_csv(
        metrics_path,
        "mlp",
        dataset_config.name,
        train_metrics,
        val_metrics,
        extra={
            "Epochs": "-",
            "Batch Size": "-",
            "Train+Eval Time (s)": round(elapsed, 4),
            "Key Parameters": f"1 hidden layer ({hidden[0]} units), lr={default_params['learning_rate_init']}, momentum={default_params['momentum']}, max {default_params['max_iter']} iters",
            "Input Features": n_features,
            "Input Feature Type": dataset_config.feature_type or dataset_config.name,
            **resource_stats,
        },
    )

    model_path = output_dir / "model.pkl"
    save_pickle(model, model_path)

    return {
        "model_name": "mlp",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_path": str(model_path),
    }
