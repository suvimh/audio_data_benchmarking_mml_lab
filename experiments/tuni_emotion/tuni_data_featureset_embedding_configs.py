from pathlib import Path

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


def make_opensmile_cleaned_embedding_config(
    frame_duration: float = 3.0,
    aggressive: bool = False,
) -> DatasetConfig:
    duration = _duration_to_suffix(frame_duration)
    feature_type = "opensmile-compare-2016"
    subdir = "cleaned_aggressive" if aggressive else "cleaned"
    suffix = ".cleaned.aggressive" if aggressive else ".cleaned"
    parquet_file = (
        f"{TUNI_OPENSMILE_PARQUET_PREFIX}_{feature_type}_{duration}{suffix}.parquet"
    )
    data_dir = str(Path(TUNI_OPENSMILE_DIR) / subdir)

    return make_data_config_for_tuni_embeddings(
        name="opensmile_cleaned",
        feature_type=feature_type,
        data_dir=data_dir,
        parquet_file=parquet_file,
    )


# Additional OpenSMILE variants from Dogdu 2022 / INTERSPEECH 2009 revisited.
# Parquets must be extracted separately; configs are ready once files exist.
OPENSMILE_VARIANTS = {
    "compare-2016": "opensmile-compare-2016",
    "emobase": "opensmile-emobase",
    "is09": "opensmile-is09",
    "egemaps": "opensmile-egemapsv02",
}


def make_opensmile_variant_config(
    variant: str,
    frame_duration: float = 3.0,
    data_dir: str | None = None,
) -> DatasetConfig:
    if variant not in OPENSMILE_VARIANTS:
        raise ValueError(f"Unknown OpenSMILE variant: {variant}. Choose from {OPENSMILE_VARIANTS}")

    duration = _duration_to_suffix(frame_duration)
    feature_type = OPENSMILE_VARIANTS[variant]
    parquet_file = f"{TUNI_OPENSMILE_PARQUET_PREFIX}_{feature_type}_{duration}.parquet"
    resolved_dir = data_dir or TUNI_OPENSMILE_DIR

    return make_data_config_for_tuni_embeddings(
        name=f"opensmile_{variant.replace('-', '_')}",
        feature_type=feature_type,
        data_dir=resolved_dir,
        parquet_file=parquet_file,
    )
