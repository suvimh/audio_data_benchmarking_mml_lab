from scripts.config.dataset_config import DatasetConfig
from scripts.config.benchmark_config import BenchmarkConfig

dataset = DatasetConfig(
    name="vocalset_melspectrograms",
    data_type="images",
    image_dir="path/to/melspectrogram/images",
    image_size=(224, 224),
)

benchmark = BenchmarkConfig(
    name="vocalset_images_full",
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
    output_dir="./results/vocalset_images",
    seed=42,
)
