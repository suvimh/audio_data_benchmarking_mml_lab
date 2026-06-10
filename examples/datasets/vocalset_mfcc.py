from scripts.config.dataset_config import DatasetConfig

dataset = DatasetConfig(
    name="vocalset_mfcc",
    data_type="features",
    data_dir="output/mfcc",
    parquet_file="mfcc_window_3.0s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
)
