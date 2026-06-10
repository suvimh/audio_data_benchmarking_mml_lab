from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)
from experiments.vtc_vocalset.vocalset_experiments_constants import (
    VOCALSET_10,
    PMT,
    TMT,
    TRAIN_SINGERS,
    VAL_SINGERS,
)
from experiments.vtc_vocalset.vocalset_data_featureset_embedding_configs import (
    make_opensmile_embedding_config,
    make_vggish_embedding_config,
    make_whisper_embedding_config,
    make_clap_embedding_config,
)
from experiments.vtc_vocalset.vocalset_raw_audio_data_config import (
    make_whisper_audio_config,
    make_clap_audio_config,
)
from experiments.vtc_vocalset.vocalset_benchmark_configs import (
    trad_ml_benchmark,
    raw_audio_input_benchmark,
)

# --- Singer-independent partition (shared across all blocks) ---

partition = PartitionConfig(
    name="vocalset_singer_independent",
    train_singer_ids=TRAIN_SINGERS,
    test_singer_ids=VAL_SINGERS,
    singer_column="singer",
)

# --- Datasets ---

opensmile = make_opensmile_embedding_config()
vggish = make_vggish_embedding_config()
whisper_emb = make_whisper_embedding_config()
clap_emb = make_clap_embedding_config()

feature_datasets = [opensmile, vggish, whisper_emb, clap_emb]

whisper_audio = make_whisper_audio_config()
clap_audio = make_clap_audio_config()

audio_datasets = [whisper_audio, clap_audio]

# --- Experiment ---

experiment = ExperimentConfig(
    name="vtc_vocalset_full",
    output_dir="./results",
    blocks=[
        # 1. Full VOCALSET_10, mixed gender
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        ),
        # 2. Full VOCALSET_10, female only
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="female",
                include_labels=VOCALSET_10,
            ),
        ),
        # 3. Full VOCALSET_10, male only
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="male",
                include_labels=VOCALSET_10,
            ),
        ),
        # 4. PMT, mixed gender
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=PMT,
            ),
        ),
        # 5. PMT, female only
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="female",
                include_labels=PMT,
            ),
        ),
        # 6. PMT, male only
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="male",
                include_labels=PMT,
            ),
        ),
        # 7. TMT, mixed gender
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=TMT,
            ),
        ),
        # 8. TMT, female only
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="female",
                include_labels=TMT,
            ),
        ),
        # 9. TMT, male only
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="male",
                include_labels=TMT,
            ),
        ),
        # 10. Raw audio fine-tuning, full VOCALSET_10, mixed gender
        ExperimentBlock(
            model_type="raw_audio",
            benchmark_config=raw_audio_input_benchmark,
            datasets=audio_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        ),
    ],
)
