from scripts.config.benchmark_config import BenchmarkConfig

whisper_finetuning_benchmark = BenchmarkConfig(
    name="finetuning_frozen_whisper",
    models=["whisper"],
    model_params={
        "whisper": {
            "model_name": "openai/whisper-base",
            "learning_rate": 1e-4,
            "batch_size": 16,
            "epochs": 200,
            "patience": 10,
        },
    },
    output_dir="./metrics/finetuning",
    seed=42,
)

clap_finetuning_benchmark = BenchmarkConfig(
    name="finetuning_frozen_clap",
    models=["clap"],
    model_params={
        "clap": {
            "learning_rate": 0.1e-4,
            "batch_size": 16,
            "epochs": 200,
            "patience": 10,
        },
    },
    output_dir="./metrics/finetuning",
    seed=42,
)

# Full fine-tuning variant: both encoder and head trainable
full_whisper_finetuning_benchmark = BenchmarkConfig(
    name="full_finetuning_whisper",
    models=["whisper"],
    model_params={
        "whisper": {
            "model_name": "openai/whisper-base",
            "freeze_encoder": False,  # Enable full fine-tuning
            "learning_rate": 5e-5,
            "encoder_learning_rate": 1e-4,
            "batch_size": 16,
            "epochs": 200,
            "patience": 10,
        },
    },
    output_dir="./metrics/finetuning",
    seed=42,
)

full_clap_finetuning_benchmark = BenchmarkConfig(
    name="full_finetuning_clap",
    models=["clap"],
    model_params={
        "clap": {
            "finetune_mode": "full",  # Enable full fine-tuning
            "learning_rate": 0.001,
            "encoder_learning_rate": 1e-4,
            "batch_size": 16,
            "epochs": 200,
            "patience": 10,
        },
    },
    output_dir="./metrics/finetuning",
    seed=42,
)

trad_ml_benchmark = BenchmarkConfig(
    name="trad_ml",
    models=["knn", "rf", "mlp", "svm"],
    model_params={
        "knn": {"n_neighbors": 5},
        "rf": {"n_estimators": 100, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500, "random_state": 42},
        "svm": {"kernel": "rbf", "C": 1.0},
    },
    output_dir="./metrics/trad_ml",
    seed=42,
)

# cnn_benchmark = BenchmarkConfig(
#     name = "cnn",
#     models = ["cnn"],
#     model_params={
#         "cnn": {"filters": [32, 64, 128], "epochs": 100, "patience": 10},
#     },
#     output_dir="./metrics/trad_ml",
#     seed=42,
# )

# imagenet_benchmark = BenchmarkConfig(
#     name="imagenet",
#     models=["resnet", "mobilenet", "efficientnet"],
#     model_params={
#         "resnet": {"epochs": 100, "patience": 10},
#         "mobilenet": {"epochs": 100, "patience": 10},
#         "efficientnet": {"epochs": 100, "patience": 10},
#     },
#     output_dir="./metrics/imagenet",
#     seed=42,
# )
