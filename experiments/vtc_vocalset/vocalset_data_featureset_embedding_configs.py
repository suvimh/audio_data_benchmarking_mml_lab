from scripts.config.dataset_config import DatasetConfig
from scripts.feature_selection.base import OPENSMILE_TOP_K_FEATURES
from experiments.vtc_vocalset.vocalset_experiments_constants import EMBEDDINGS_DATA_DIR


def make_data_config_for_vtc_embeddings(
        name="opensmile",
        feature_type="opensmile-compare-2016",
        data_dir=EMBEDDINGS_DATA_DIR,
        parquet_file="vocalset_dataset_opensmile-compare-2016_3.0s.parquet"
    ):
    return DatasetConfig(
        name=name,
        data_type="features",
        feature_type=feature_type,
        feature_column="embedding",
        label_column="vocal_technique",
        data_dir=data_dir,
        parquet_file=parquet_file,
    )


def make_opensmile_embedding_config():
    """OpenSMILE embeddings are reduced to top-k features during loading."""
    return make_data_config_for_vtc_embeddings(
        name="opensmile",
        feature_type="opensmile-compare-2016",
        parquet_file="vocalset_dataset_opensmile-compare-2016_3.0s.parquet",
    )


OPENSMILE_FEATURE_COUNT = OPENSMILE_TOP_K_FEATURES


def make_vggish_embedding_config():
    return make_data_config_for_vtc_embeddings(
        name="vggish",
        feature_type="vggish",
        parquet_file="vocalset_dataset_vocalset_dataset_vggish_3.0s.parquet",
    )


def make_whisper_embedding_config():
    return make_data_config_for_vtc_embeddings(
        name="whisper",
        feature_type="whisper-base",
        parquet_file="vocalset_dataset_whisper_whisper-base_3.0s.parquet",
    )


def make_clap_embedding_config():
    return make_data_config_for_vtc_embeddings(
        name="clap",
        feature_type="clap-2023",
        parquet_file="vocalset_dataset_vocalset_dataset_clap-2023_3.0s.parquet",
    )
