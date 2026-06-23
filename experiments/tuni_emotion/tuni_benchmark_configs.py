from scripts.config.benchmark_config import BenchmarkConfig
from scripts.feature_selection.base import FeatureSelectionConfig

# CNN is intentionally excluded for now; we only run the traditional ML models.
# Its params are kept here so it can be re-enabled by adding "cnn" back to a
# benchmark's `models` list.
_MODEL_PARAMS = {
    "knn": {"n_neighbors": 5},
    "rf": {"n_estimators": 100, "random_state": 42},
    "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500, "random_state": 42},
    "svm": {"kernel": "rbf", "C": 1.0, "class_weight": "balanced"},
    "cnn": {"filters": [32, 64, 128], "epochs": 25, "patience": 5, "batch_size": 64},
}

trad_ml_benchmark = BenchmarkConfig(
    name="trad_ml",
    models=["knn", "rf", "mlp", "svm"],
    model_params=_MODEL_PARAMS,
    output_dir="./metrics/trad_ml",
    seed=42,
)

trad_ml_benchmark_filter = BenchmarkConfig(
    name="trad_ml_fs_filter",
    models=["knn", "rf", "mlp", "svm"],
    model_params=_MODEL_PARAMS,
    feature_selection=FeatureSelectionConfig(
        enabled=True,
        method="filter",
        params={"variance_threshold": 0.0, "k": 500},
    ),
    output_dir="./metrics/trad_ml_fs_filter",
    seed=42,
)
