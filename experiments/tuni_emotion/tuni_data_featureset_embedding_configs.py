from scripts.config.dataset_config import DatasetConfig
from scripts.feature_selection.base import OPENSMILE_TOP_K_FEATURES
from experiments.tuni_emotion.tuni_experiments_constants import (
    TUNI_CLAP_DIR,
    TUNI_OPENSMILE_DIR,
    TUNI_OPENSMILE_PARQUET_PREFIX,
    TUNI_WHISPER_DIR,
)


def _duration_to_suffix(frame_duration: float) -> str:
    return f"{frame_duration:.1f}s"


def make_data_config_for_tuni_embeddings(
    name: str = "opensmile",
    feature_type: str = "opensmile-compare-2016",
    data_dir: str = TUNI_OPENSMILE_DIR,
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
    """OpenSMILE embeddings are reduced to top-k features during loading."""
    duration = _duration_to_suffix(frame_duration)
    feature_type = f"opensmile-{opensmile_variant}"
    parquet_file = f"{TUNI_OPENSMILE_PARQUET_PREFIX}_{feature_type}_{duration}.parquet"

    return make_data_config_for_tuni_embeddings(
        name="opensmile",
        feature_type=feature_type,
        data_dir=TUNI_OPENSMILE_DIR,
        parquet_file=parquet_file,
    )


OPENSMILE_FEATURE_COUNT = OPENSMILE_TOP_K_FEATURES


def make_whisper_embedding_config(frame_duration: float = 3.0) -> DatasetConfig:
    duration = _duration_to_suffix(frame_duration)
    feature_type = "whisper_whisper-base"
    parquet_file = f"{TUNI_OPENSMILE_PARQUET_PREFIX}_{feature_type}_{duration}.parquet"

    return make_data_config_for_tuni_embeddings(
        name="whisper",
        feature_type=feature_type,
        data_dir=TUNI_WHISPER_DIR,
        parquet_file=parquet_file,
    )


def make_clap_embedding_config(frame_duration: float = 3.0) -> DatasetConfig:
    duration = _duration_to_suffix(frame_duration)
    feature_type = "clap-2023"
    parquet_file = f"{TUNI_OPENSMILE_PARQUET_PREFIX}_{feature_type}_{duration}.parquet"

    return make_data_config_for_tuni_embeddings(
        name="clap",
        feature_type=feature_type,
        data_dir=TUNI_CLAP_DIR,
        parquet_file=parquet_file,
    )
