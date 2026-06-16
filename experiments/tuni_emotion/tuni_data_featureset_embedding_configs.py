from scripts.config.dataset_config import DatasetConfig
from experiments.tuni_emotion.tuni_experiments_constants import (
    TUNI_EMBEDDINGS_DATA_DIR,
    TUNI_OPENSMILE_PARQUET_PREFIX,
)


def _duration_to_suffix(frame_duration: float) -> str:
    return f"{frame_duration:.1f}s"


def make_data_config_for_tuni_embeddings(
    name: str = "opensmile",
    feature_type: str = "opensmile-compare-2016",
    data_dir: str = TUNI_EMBEDDINGS_DATA_DIR,
    parquet_file: str = "tuni_emotion_dataset_opensmile-compare-2016_3.0s.parquet",
):
    return DatasetConfig(
        name=name,
        data_type="features",
        feature_type=feature_type,
        feature_column="embedding",
        label_column="emotion",
        data_dir=data_dir,
        parquet_file=parquet_file,
    )


def make_opensmile_embedding_config(
    frame_duration: float = 3.0,
    opensmile_variant: str = "compare-2016",
) -> DatasetConfig:
    duration = _duration_to_suffix(frame_duration)
    feature_type = f"opensmile-{opensmile_variant}"
    parquet_file = f"{TUNI_OPENSMILE_PARQUET_PREFIX}_{feature_type}_{duration}.parquet"

    return make_data_config_for_tuni_embeddings(
        name="opensmile",
        feature_type=feature_type,
        parquet_file=parquet_file,
    )
