from scripts.config.benchmark_config import BenchmarkConfig

raw_audio_input_benchmark = BenchmarkConfig(
    name="raw_audio_finetuning",
    models=["whisper", "clap"],
    model_params={
        "whisper": {
            "model_name": "openai/whisper-base",
            "learning_rate": 5e-5,
            "batch_size": 16,
            "epochs": 50,
            "patience": 10,
        },
        "clap": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,
        },
    },
    output_dir="./metrics/raw_audio",
    seed=42,
)

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
