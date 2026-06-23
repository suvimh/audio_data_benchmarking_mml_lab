from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)
from experiments.tuni_emotion.tuni_experiments_constants import (
    TUNI_EMOTIONS,
    TRAIN_SINGERS,
    VAL_SINGERS,
)
from experiments.tuni_emotion.tuni_data_featureset_embedding_configs import (
    make_opensmile_embedding_config,
)
from experiments.tuni_emotion.tuni_benchmark_configs import trad_ml_benchmark

partition = PartitionConfig(
    name="tuni_singer_independent",
    train_singer_ids=TRAIN_SINGERS,
    test_singer_ids=VAL_SINGERS,
    singer_column="singer",
)

opensmile = make_opensmile_embedding_config(frame_duration=3.0)

experiment = ExperimentConfig(
    name="tuni_emotion_baseline",
    output_dir="./results",
    blocks=[
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=[opensmile],
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=TUNI_EMOTIONS,
            ),
        ),
    ],
)
