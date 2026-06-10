from scripts.config.benchmark_config import BenchmarkConfig

benchmark = BenchmarkConfig(
    name="imagenet",
    models=["imagenet"],
    model_params={
        "imagenet": {
            "backbone": "efficientnet",
            "learning_rate": 0.001,
            "dense_units": 128,
            "dropout_rate": 0.5,
            "epochs": 100,
            "patience": 15,
        },
    },
    output_dir="./results/imagenet",
    seed=42,
    batch_size=32,
    epochs=100,
    early_stopping_patience=15,
)
