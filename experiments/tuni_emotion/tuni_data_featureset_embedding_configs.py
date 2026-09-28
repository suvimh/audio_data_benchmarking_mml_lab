from scripts.config.dataset_config import DatasetConfig
from experiments.tuni_emotion.tuni_experiments_constants import EMBEDDINGS_DATA_DIR


def make_data_config_for_tuni_embeddings(
    name="opensmile",
    feature_type="opensmile-compare-2016",
    data_dir=EMBEDDINGS_DATA_DIR,
    parquet_file="tuni_emotion_dataset_opensmile-compare-2016_0.5s.parquet",
    singer_column="singer",
    label_column="emotion",
):
    return DatasetConfig(
        name=name,
        data_type="features",
        feature_type=feature_type,
        feature_column="embedding",
        label_column=label_column,
        data_dir=data_dir,
        parquet_file=parquet_file,
        singer_column=singer_column,
    )


def make_opensmile_embedding_config(
    parquet_file="tuni_emotion_dataset_opensmile-compare-2016_0.5s.parquet",
    data_dir=EMBEDDINGS_DATA_DIR,
    label_column="emotion",
):
    return make_data_config_for_tuni_embeddings(
        name="opensmile",
        feature_type="opensmile-compare-2016",
        parquet_file=parquet_file,
        data_dir=data_dir,
        label_column=label_column,
    )


def make_whisper_embedding_config(
    parquet_file="tuni_emotion_dataset_whisper_whisper-base_0.5s.parquet",
    data_dir=EMBEDDINGS_DATA_DIR,
    label_column="emotion",
):
    return make_data_config_for_tuni_embeddings(
        name="whisper",
        feature_type="whisper-base",
        parquet_file=parquet_file,
        data_dir=data_dir,
        label_column=label_column,
    )


def make_clap_embedding_config(
    parquet_file="tuni_emotion_dataset_clap-2023_0.5s.parquet",
    data_dir=EMBEDDINGS_DATA_DIR,
    label_column="emotion",
):
    return make_data_config_for_tuni_embeddings(
        name="clap",
        feature_type="clap-2023",
        parquet_file=parquet_file,
        data_dir=data_dir,
        label_column=label_column,
    )

