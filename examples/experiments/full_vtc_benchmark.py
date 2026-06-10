from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)

VOCALSET_10 = ["vibrato", "straight", "belt", "breathy", "lip_trill",
               "inhaled", "spoken", "trill", "trillo", "vocal_fry"]

# --- Datasets: pure data definitions (no partitioning, no filtering) ---

opensmile = DatasetConfig(
    name="opensmile",
    data_type="features",
    data_dir="output/opensmile",
    parquet_file="opensmile_egemapsv02_3.0s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
)

vggish = DatasetConfig(
    name="vggish",
    data_type="features",
    data_dir="output/vggish",
    parquet_file="vggish_window_3.0s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
)

mfcc = DatasetConfig(
    name="mfcc",
    data_type="features",
    data_dir="output/mfcc",
    parquet_file="mfcc_window_3.0s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
)

whisper_audio = DatasetConfig(
    name="whisper_audio",
    data_type="raw_audio",
    audio_dir="data/VocalSet/audio",
    sample_rate=16000,
    frame_duration=3.0,
    overlap=0.25,
    level_names=["singer_id", "technique", "exercise"],
    label_level="technique",
)

clap_audio = DatasetConfig(
    name="clap_audio",
    data_type="raw_audio",
    audio_dir="data/VocalSet/audio",
    sample_rate=44100,
    frame_duration=3.0,
    overlap=0.25,
    level_names=["singer_id", "technique", "exercise"],
    label_level="technique",
)

melspectrograms = DatasetConfig(
    name="melspectrograms",
    data_type="images",
    image_dir="data/melspectrograms",
    image_size=(224, 224),
)

# --- Partition: singer-independent split shared across all experiments ---

partition = PartitionConfig(
    name="singer_independent",
    train_singer_ids=[f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 7)],
    test_singer_ids=["f10", "f11", "m7", "m8", "m9"],
    singer_column="singer",
)

# --- Benchmark configs ---

trad_ml = BenchmarkConfig(
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
)

imagenet = BenchmarkConfig(
    name="imagenet",
    models=["imagenet"],
    model_params={
        "imagenet": {"backbone": "efficientnet", "epochs": 100, "patience": 15},
    },
    output_dir="./results/imagenet",
    seed=42,
)

finetune = BenchmarkConfig(
    name="finetune",
    models=["whisper", "clap"],
    model_params={
        "whisper": {"model_name": "openai/whisper-base", "freeze_encoder": True,
                     "epochs": 50, "patience": 10},
        "clap": {"finetune_mode": "head_only", "epochs": 100, "patience": 10},
    },
    output_dir="./results/finetune",
    seed=42,
)

# --- Experiment: all blocks together ---

experiment = ExperimentConfig(
    name="vtc_full_benchmark",
    output_dir="./results",
    blocks=[
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml,
            datasets=[opensmile, vggish, mfcc],
            partition=partition,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        ),
        ExperimentBlock(
            model_type="images",
            benchmark_config=imagenet,
            datasets=[melspectrograms],
            partition=partition,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        ),
        ExperimentBlock(
            model_type="raw_audio",
            benchmark_config=finetune,
            datasets=[whisper_audio, clap_audio],
            partition=partition,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        ),
    ],
)
