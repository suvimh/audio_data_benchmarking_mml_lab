from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from scripts.config.dataset_config import DatasetConfig


@dataclass
class ImageData:
    train_ds: object
    val_ds: object
    class_names: List[str]
    n_classes: int
    input_shape: tuple


def load_image_data(config: DatasetConfig) -> ImageData:
    import tensorflow as tf

    img_size = config.image_size
    batch_size = 32
    data_dir = Path(config.image_dir)

    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=None,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        validation_split=None,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return ImageData(
        train_ds=train_ds,
        val_ds=val_ds,
        class_names=class_names,
        n_classes=len(class_names),
        input_shape=(*img_size, 3),
    )
