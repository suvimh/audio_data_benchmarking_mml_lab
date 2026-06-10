from scripts.config.benchmark_config import BenchmarkConfig

benchmark = BenchmarkConfig(
    name="trad_ml",
    models=["knn", "svm", "rf", "mlp", "cnn"],
    model_params={
        "knn": {"n_neighbors": 5},
        "svm": {"kernel": "rbf", "C": 1.0, "class_weight": "balanced"},
        "rf": {"n_estimators": 100, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500, "random_state": 42},
        "cnn": {"filters": [32, 64, 128], "epochs": 100, "patience": 10},
    },
    output_dir="./results/trad_ml",
    seed=42,
    batch_size=32,
    epochs=100,
    early_stopping_patience=10,
)
