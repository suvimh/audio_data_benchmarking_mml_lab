from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)
from experiments.tuni_emotion.tuni_experiments_constants import (
    EMOTIONS,
    TRAIN_SINGERS,
    VAL_SINGERS,
)
from experiments.tuni_emotion.tuni_data_featureset_embedding_configs import (
    make_opensmile_embedding_config,
    make_whisper_embedding_config,
    make_clap_embedding_config,
)
from experiments.tuni_emotion.tuni_raw_audio_data_config import (
    make_whisper_audio_config,
    make_clap_audio_config,
)
from experiments.tuni_emotion.tuni_benchmark_configs import (
    trad_ml_benchmark,
    whisper_finetuning_benchmark,
    clap_finetuning_benchmark,
    full_whisper_finetuning_benchmark,
    full_clap_finetuning_benchmark,
)

# --- Singer-independent partition (shared across all blocks) ---

partition = PartitionConfig(
    name="tuni_emotion_singer_independent",
    train_singer_ids=TRAIN_SINGERS,
    test_singer_ids=VAL_SINGERS,
    singer_column="singer",
)

# --- Datasets ---

opensmile = make_opensmile_embedding_config()
whisper_emb = make_whisper_embedding_config()
clap_emb = make_clap_embedding_config()

feature_datasets = [opensmile, whisper_emb, clap_emb]

whisper_audio = make_whisper_audio_config()
clap_audio = make_clap_audio_config()

FINETUNING_METRICS_PATH = "./metrics/tuni_emotion/fine_tuning/metrics.csv"

# --- Experiment ---
tuni_emotion_experiment = ExperimentConfig(
    name="tuni_emotion",
    output_dir="./metrics",
    blocks=[
        # 1. Trad ML
        ExperimentBlock(
            model_type="trad_ml",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=0.5,
            overlap=0.25,
            filters=ExperimentFilters(
                include_labels=EMOTIONS,
            ),
        ),
        ExperimentBlock(
            model_type="clap_frozen_encoder",
            benchmark_config=clap_finetuning_benchmark,
            datasets=[clap_audio],
            partition=partition,
            frame_duration=0.5,
            overlap=0.25,
            metrics_path=FINETUNING_METRICS_PATH,
            filters=ExperimentFilters(
                include_labels=EMOTIONS,
            ),
        ),
        ExperimentBlock(
            model_type="whisper_frozen_encoder",
            benchmark_config=whisper_finetuning_benchmark,
            datasets=[whisper_audio],
            partition=partition,
            frame_duration=0.5,
            overlap=0.25,
            metrics_path=FINETUNING_METRICS_PATH,
            filters=ExperimentFilters(
                include_labels=EMOTIONS,
            ),
        ),
        # Raw audio full fine-tuning
        ExperimentBlock(
            model_type="clap_full_fine_tuning",
            benchmark_config=full_clap_finetuning_benchmark,
            datasets=[clap_audio],
            partition=partition,
            frame_duration=0.5,
            overlap=0.25,
            metrics_path=FINETUNING_METRICS_PATH,
            n_repeats=10,
            filters=ExperimentFilters(
                include_labels=EMOTIONS,
            ),
        ),
        ExperimentBlock(
            model_type="whisper_full_fine_tuning",
            benchmark_config=full_whisper_finetuning_benchmark,
            datasets=[whisper_audio],
            partition=partition,
            frame_duration=0.5,
            overlap=0.25,
            metrics_path=FINETUNING_METRICS_PATH,
            n_repeats=10,
            filters=ExperimentFilters(
                include_labels=EMOTIONS,
            ),
        ),
    ],
)
