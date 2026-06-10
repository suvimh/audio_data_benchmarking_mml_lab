from scripts.config.dataset_config import DatasetConfig

dataset = DatasetConfig(
    name="vocalset_vggish",
    data_type="features",
    data_dir="output/vggish",
    parquet_file="vggish_window_3.0s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
)
