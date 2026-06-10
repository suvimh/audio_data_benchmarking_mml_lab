from scripts.config.dataset_config import DatasetConfig
from experiments.vtc_vocalset.vocalset_experiments_constants import IMAGENET_DATA_DIR


def make_melspectrograms_config(data_dir=IMAGENET_DATA_DIR):
    return DatasetConfig(
        name="melspectrograms",
        data_type="images",
        image_dir=data_dir,
        image_size=(224, 224),
    )
