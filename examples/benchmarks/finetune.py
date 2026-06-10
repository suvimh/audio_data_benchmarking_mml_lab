from scripts.config.benchmark_config import BenchmarkConfig

benchmark = BenchmarkConfig(
    name="finetune",
    models=["whisper", "clap"],
    model_params={
        "whisper": {
            "model_name": "openai/whisper-base",
            "freeze_encoder": True,
            "learning_rate": 5e-5,
            "encoder_learning_rate": 1e-6,
            "batch_size": 16,
            "epochs": 50,
            "patience": 10,
        },
        "clap": {
            "finetune_mode": "head_only",
            "learning_rate": 0.001,
            "encoder_learning_rate": 1e-6,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,
            "sample_rate": 44100,
        },
    },
    output_dir="./results/finetune",
    seed=42,
    batch_size=16,
    epochs=50,
    early_stopping_patience=10,
)
