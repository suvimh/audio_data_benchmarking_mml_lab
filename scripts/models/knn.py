from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from sklearn.neighbors import KNeighborsClassifier

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.data.features import FeaturesData
from scripts.evaluation.metrics import (
    compute_metrics,
    save_metrics_csv,
)
from scripts.utils import ensure_dir, save_pickle

MODEL_METADATA = {
    "name": "knn",
    "supported_data_types": ["features"],
    "description": "K-Nearest Neighbors classifier",
    "default_params": {"n_neighbors": 5},
}


def train(data: FeaturesData, params: Dict[str, Any]) -> KNeighborsClassifier:
    model = KNeighborsClassifier(**params)
    model.fit(data.X_train, data.y_train)
    return model


def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    from scripts.data.features import load_features_data

    output_dir = ensure_dir(output_dir)
    start_time = time.perf_counter()
    data = load_features_data(dataset_config)

    params = bench_config.get_model_params("knn")
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
    save_metrics_csv(
        metrics_path,
        "knn",
        dataset_config.name,
        train_metrics,
        val_metrics,
        extra={
            "Epochs": "-",
            "Batch Size": "-",
            "Train+Eval Time (s)": round(elapsed, 4),
            "Key Parameters": f"k={default_params['n_neighbors']}, Euclidean distance",
            "Input Features": n_features,
            "Input Feature Type": dataset_config.feature_type or dataset_config.name,
            **resource_stats,
        },
    )

    model_path = output_dir / "model.pkl"
    save_pickle(model, model_path)

    return {
        "model_name": "knn",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_path": str(model_path),
    }
