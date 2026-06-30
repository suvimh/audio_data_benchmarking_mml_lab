from scripts.config.benchmark_config import BenchmarkConfig

trad_ml_benchmark = BenchmarkConfig(
    name="trad_ml",
    models=["knn", "rf", "mlp", "svm", "cnn"],
    model_params={
        "knn": {"n_neighbors": 5},
        "rf": {"n_estimators": 100, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500, "random_state": 42},
        "svm": {"kernel": "rbf", "C": 1.0, "class_weight": "balanced"},
        "cnn": {"filters": [32, 64, 128], "epochs": 100, "patience": 10},
    },
    output_dir="./metrics/trad_ml",
    seed=42,
)
