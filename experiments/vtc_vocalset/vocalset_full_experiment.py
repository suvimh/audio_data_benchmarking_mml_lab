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
    make_mfcc_config
)
from experiments.vtc_vocalset.vocalset_raw_audio_data_config import (
    make_whisper_audio_config,
    make_clap_audio_config,
)
from experiments.vtc_vocalset.vocalset_benchmark_configs import (
    trad_ml_benchmark,
    whisper_finetuning_benchmark,
    clap_finetuning_benchmark,
    full_whisper_finetuning_benchmark,
    full_clap_finetuning_benchmark,
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
mfccs = make_mfcc_config()

feature_datasets = [opensmile, vggish, whisper_emb, clap_emb, mfccs]

whisper_audio = make_whisper_audio_config()
clap_audio = make_clap_audio_config()

FINETUNING_METRICS_PATH = "./metrics/vtc_vocalset_full/finetuning_metrics/metrics.csv"

# --- Experiment ---
vtc_experiment = ExperimentConfig(
    name="vtc_vocalset_full",
    output_dir="./metrics",
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
        # # 2. Full VOCALSET_10, female only
        # ExperimentBlock(
        #     model_type="features",
        #     benchmark_config=trad_ml_benchmark,
        #     datasets=feature_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="female",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
        # # 3. Full VOCALSET_10, male only
        # ExperimentBlock(
        #     model_type="features",
        #     benchmark_config=trad_ml_benchmark,
        #     datasets=feature_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="male",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
        # # 4. PMT, mixed gender
        # ExperimentBlock(
        #     model_type="features",
        #     benchmark_config=trad_ml_benchmark,
        #     datasets=feature_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=PMT,
        #     ),
        # ),
        # # 5. PMT, female only
        # ExperimentBlock(
        #     model_type="features",
        #     benchmark_config=trad_ml_benchmark,
        #     datasets=feature_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="female",
        #         include_labels=PMT,
        #     ),
        # ),
        # # 6. PMT, male only
        # ExperimentBlock(
        #     model_type="features",
        #     benchmark_config=trad_ml_benchmark,
        #     datasets=feature_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="male",
        #         include_labels=PMT,
        #     ),
        # ),
        # # 7. TMT, mixed gender
        # ExperimentBlock(
        #     model_type="features",
        #     benchmark_config=trad_ml_benchmark,
        #     datasets=feature_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=TMT,
        #     ),
        # ),
        # # 8. TMT, female only
        # ExperimentBlock(
        #     model_type="features",
        #     benchmark_config=trad_ml_benchmark,
        #     datasets=feature_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="female",
        #         include_labels=TMT,
        #     ),
        # ),
        # # 9. TMT, male only
        # ExperimentBlock(
        #     model_type="features",
        #     benchmark_config=trad_ml_benchmark,
        #     datasets=feature_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="male",
        #         include_labels=TMT,
        #     ),
        # ),
        # # # 10. Raw audio fine-tuning, full VOCALSET_10, mixed gender
        # ExperimentBlock(
        #     model_type="clap_frozen_encoder",
        #     benchmark_config=clap_finetuning_benchmark,
        #     datasets=[clap_audio],
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     metrics_path=FINETUNING_METRICS_PATH,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
        # ExperimentBlock(
        #     model_type="whisper_frozen_encoder",
        #     benchmark_config=whisper_finetuning_benchmark,
        #     datasets=[whisper_audio],
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     metrics_path=FINETUNING_METRICS_PATH,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
        # # 11. Raw audio fine-tuning, full VOCALSET_10, female only
        # ExperimentBlock(
        #     model_type="raw_audio",
        #     benchmark_config=finetuning_benchmark,
        #     datasets=audio_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="female",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
        # # 12. Raw audio fine-tuning, full VOCALSET_10, male only
        # ExperimentBlock(
        #     model_type="raw_audio",
        #     benchmark_config=finetuning_benchmark,
        #     datasets=audio_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="male",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
        # # 13. Raw audio fine-tuning, full PMT, mixed gender
        # ExperimentBlock(
        #     model_type="raw_audio",
        #     benchmark_config=finetuning_benchmark,
        #     datasets=audio_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=PMT,
        #     ),
        # ),
        # # 14. Raw audio fine-tuning, full PMT, female only
        # ExperimentBlock(
        #     model_type="raw_audio",
        #     benchmark_config=finetuning_benchmark,
        #     datasets=audio_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="female",
        #         include_labels=PMT,
        #     ),
        # ),
        # # 15. Raw audio fine-tuning, full PMT, male only
        # ExperimentBlock(
        #     model_type="raw_audio",
        #     benchmark_config=finetuning_benchmark,
        #     datasets=audio_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="male",
        #         include_labels=PMT,
        #     ),
        # ),
        # # 16. Raw audio fine-tuning, full TMT, mixed gender
        # ExperimentBlock(
        #     model_type="raw_audio",
        #     benchmark_config=finetuning_benchmark,
        #     datasets=audio_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=TMT,
        #     ),
        # ),
        # # 17. Raw audio fine-tuning, full TMT, female only
        # ExperimentBlock(
        #     model_type="raw_audio",
        #     benchmark_config=finetuning_benchmark,
        #     datasets=audio_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="female",
        #         include_labels=TMT,
        #     ),
        # ),
        # # 18. Raw audio fine-tuning, full TMT, male only
        # ExperimentBlock(
        #     model_type="raw_audio",
        #     benchmark_config=finetuning_benchmark,
        #     datasets=audio_datasets,
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     filters=ExperimentFilters(
        #         gender="male",
        #         include_labels=TMT,
        #     ),
        # ),
        # 19. Raw audio full fine-tuning, 10-way mixed
        # ExperimentBlock(
        #     model_type="clap_full_fine_tuning",
        #     benchmark_config=full_clap_finetuning_benchmark,
        #     datasets=[clap_audio],
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     metrics_path=FINETUNING_METRICS_PATH,
        #     n_repeats=10,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
        # ExperimentBlock(
        #     model_type="whisper_full_fine_tuning",
        #     benchmark_config=full_whisper_finetuning_benchmark,
        #     datasets=[whisper_audio],
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.25,
        #     metrics_path=FINETUNING_METRICS_PATH,
        #     n_repeats=10,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
    ],
)
