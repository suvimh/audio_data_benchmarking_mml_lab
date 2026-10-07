from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)
from experiments.tuni_emotion.tuni_experiments_constants import (
    EMOTIONS,
    PERCEPTION_TEST_TEST_FILES
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
    name="tuni_emotion_preception_test_compare",
    test_data_ids=PERCEPTION_TEST_TEST_FILES,
    data_column_label="filename",
)

# --- Datasets ---

# opensmile_3sec = make_opensmile_embedding_config(
#     parquet_file="tuni_emotion_dataset_opensmile-compare-2016_3.0s.parquet"
# )
# whisper_emb_3sec = make_whisper_embedding_config(
#     parquet_file="tuni_emotion_dataset_whisper_whisper-base_3.0s.parquet"
# )
# clap_emb_3sec = make_clap_embedding_config(
#     parquet_file="tuni_emotion_dataset_clap-2023_3.0s.parquet"
# )


opensmile_500ms= make_opensmile_embedding_config(
    parquet_file="tuni_emotion_dataset_opensmile-compare-2016_0.5s.parquet"
)
whisper_emb_500ms = make_whisper_embedding_config(
    parquet_file="tuni_emotion_dataset_whisper_whisper-base_0.5s.parquet"
)
clap_emb_500ms = make_clap_embedding_config(
    parquet_file="tuni_emotion_dataset_clap-2023_0.5s.parquet"
)

# feature_datasets_3sec = [opensmile_3sec, whisper_emb_3sec, clap_emb_3sec]
feature_datasets_500ms = [opensmile_500ms, whisper_emb_500ms, clap_emb_500ms]

whisper_audio = make_whisper_audio_config()
clap_audio = make_clap_audio_config()

FINETUNING_METRICS_PATH = "./metrics/perception_test_comparison_500ms/fine_tuning/metrics.csv"

# --- Experiment ---
tuni_perception_test_comparison_experiment = ExperimentConfig(
    name="perception_test_comparison_500ms",
    output_dir="./metrics",
    blocks=[
        # 1. Trad ML
        ExperimentBlock(
            model_type="trad_ml",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets_500ms,
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
