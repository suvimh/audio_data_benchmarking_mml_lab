from scripts.config.dataset_config import DatasetConfig

dataset = DatasetConfig(
    name="vocalset_melspectrograms",
    data_type="images",
    image_dir="data/melspectrograms",
    image_size=(224, 224),
)
