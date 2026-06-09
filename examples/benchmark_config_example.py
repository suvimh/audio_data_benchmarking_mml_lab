from scripts.config.benchmark_config import BenchmarkConfig

features_benchmark = BenchmarkConfig(
    name="vtc_features",
    models=["knn", "rf", "mlp", "cnn_vggish"],
    model_params={
        "knn": {"n_neighbors": 5},
        "rf": {"n_estimators": 100, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500},
        "cnn_vggish": {"epochs": 100, "patience": 10},
    },
    output_dir="./results",
    seed=42,
)

audio_benchmark = BenchmarkConfig(
    name="vtc_audio",
    models=["whisper", "clap"],
    model_params={
        "whisper": {"model_name": "openai/whisper-base", "epochs": 50},
        "clap": {"epochs": 100, "hidden_dim": 512},
    },
    output_dir="./results",
    seed=42,
)

image_benchmark = BenchmarkConfig(
    name="vtc_images",
    models=["imagenet"],
    model_params={
        "imagenet": {"backbone": "efficientnet", "epochs": 100},
    },
    output_dir="./results",
    seed=42,
)

full_benchmark = BenchmarkConfig(
    name="vtc_full",
    models=["knn", "rf", "mlp", "cnn_vggish", "imagenet", "whisper", "clap"],
    model_params={
        "knn": {"n_neighbors": 5},
        "rf": {"n_estimators": 100},
        "mlp": {"hidden_layer_sizes": (12,)},
        "cnn_vggish": {"epochs": 100, "patience": 10},
        "imagenet": {"backbone": "efficientnet"},
        "whisper": {"model_name": "openai/whisper-base"},
        "clap": {"epochs": 100},
    },
    output_dir="./results",
    seed=42,
)
