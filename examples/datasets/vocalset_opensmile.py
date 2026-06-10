from scripts.config.dataset_config import DatasetConfig

dataset = DatasetConfig(
    name="vocalset_opensmile",
    data_type="features",
    data_dir="output/opensmile",
    parquet_file="opensmile_egemapsv02_3.0s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
)
