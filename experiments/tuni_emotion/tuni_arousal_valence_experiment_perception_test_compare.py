from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)
from experiments.tuni_emotion.tuni_experiments_constants import (
    AROUSAL,
    VALENCE,
    PERCEPTION_TEST_TEST_FILES,
    VALENCE_AROUSAL_EMBEDDINGS_DATA_DIR,
)
from experiments.tuni_emotion.tuni_data_featureset_embedding_configs import (
    make_opensmile_embedding_config,
    make_whisper_embedding_config,
    make_clap_embedding_config,
)

from experiments.tuni_emotion.tuni_benchmark_configs import (
    trad_ml_benchmark,
)

# --- Singer-independent partition (shared across all blocks) ---

partition = PartitionConfig(
    name="tuni_emotion_preception_test_compare",
    test_data_ids=PERCEPTION_TEST_TEST_FILES,
    data_column_label="filename",
)

# --- Datasets ---
# Files carry both `emotion` and `arousal`/`valence` columns in the a_v map dir.

opensmile_arousal = make_opensmile_embedding_config(
    "tuni_emotion_dataset_opensmile-compare-2016_3.0s_valence_arousal.parquet",
    data_dir=VALENCE_AROUSAL_EMBEDDINGS_DATA_DIR,
    label_column="arousal",
)
whisper_emb_arousal = make_whisper_embedding_config(
    "tuni_emotion_dataset_whisper_whisper-base_3.0s_valence_arousal.parquet",
    data_dir=VALENCE_AROUSAL_EMBEDDINGS_DATA_DIR,
    label_column="arousal",
)
clap_emb_arousal = make_clap_embedding_config(
    "tuni_emotion_dataset_clap-2023_3.0s_valence_arousal.parquet",
    data_dir=VALENCE_AROUSAL_EMBEDDINGS_DATA_DIR,
    label_column="arousal",
)

feature_datasets_arousal = [opensmile_arousal, whisper_emb_arousal, clap_emb_arousal]

opensmile_valence = make_opensmile_embedding_config(
    "tuni_emotion_dataset_opensmile-compare-2016_3.0s_valence_arousal.parquet",
    data_dir=VALENCE_AROUSAL_EMBEDDINGS_DATA_DIR,
    label_column="valence",
)
whisper_emb_valence = make_whisper_embedding_config(
    "tuni_emotion_dataset_whisper_whisper-base_3.0s_valence_arousal.parquet",
    data_dir=VALENCE_AROUSAL_EMBEDDINGS_DATA_DIR,
    label_column="valence",
)
clap_emb_valence = make_clap_embedding_config(
    "tuni_emotion_dataset_clap-2023_3.0s_valence_arousal.parquet",
    data_dir=VALENCE_AROUSAL_EMBEDDINGS_DATA_DIR,
    label_column="valence",
)

feature_datasets_valence = [opensmile_valence, whisper_emb_valence, clap_emb_valence]

# --- Experiment ---
arousal_valence_experiment = ExperimentConfig(
    name="tuni_emotion_perception_test_comparison_arousal_valence",
    output_dir="./metrics/perception_test_comparison/arousal_valence",
    blocks=[
        # 1. Trad ML Arousal
        ExperimentBlock(
            model_type="trad_ml_arousal",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets_arousal,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                include_labels=AROUSAL,
            ),
        ),
        # 1. Trad ML Valence
        ExperimentBlock(
            model_type="trad_ml_valence",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets_valence,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                include_labels=VALENCE,
            ),
        ),
    ],
)
