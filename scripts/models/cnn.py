from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.evaluation.metrics import (
    compute_metrics,
    save_metrics_csv,
)
from scripts.evaluation.visualization import plot_training_history
from scripts.utils import ensure_dir, set_seed

MODEL_METADATA = {
    "name": "cnn",
    "supported_data_types": ["features"],
    "description": "3-layer CNN classifier on padded sequential feature data",
    "default_params": {
        "filters": [32, 64, 128],
        "kernel_size": 3,
        "dense_units": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
    },
}


def _build_model(
    input_shape: Tuple[int, int, int],
    n_classes: int,
    params: Dict[str, Any],
):
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for i, n_filters in enumerate(params["filters"]):
        x = layers.Conv2D(n_filters, params["kernel_size"], padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(params["dropout_rate"])(x)

    x = layers.Flatten()(x)
    x = layers.Dense(params["dense_units"], activation="relu")(x)
    x = layers.Dropout(params["dropout_rate"])(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(data, params: Dict[str, Any], n_classes: int, callbacks=None) -> tuple:
    import tensorflow as tf

    set_seed(42)
    n_timesteps = data.X_train.shape[1]
    input_shape = (n_timesteps, data.input_dim, 1)

    model = _build_model(input_shape, n_classes, params)

    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks or [],
        verbose=1,
    )
    return model, history


def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    from scripts.data.features import load_features_data

    import tensorflow as tf

    output_dir = ensure_dir(output_dir)
    dc = DatasetConfig(
        name=dataset_config.name,
        data_type=dataset_config.data_type,
        data_dir=dataset_config.data_dir,
        feature_type=dataset_config.feature_type,
        train_data_path=dataset_config.train_data_path,
        val_data_path=dataset_config.val_data_path,
        train_data_paths=dataset_config.train_data_paths,
        val_data_paths=dataset_config.val_data_paths,
        gender_split=dataset_config.gender_split,
        include_labels=dataset_config.include_labels,
        exclude_labels=dataset_config.exclude_labels,
        label_map=dataset_config.label_map,
        feature_column=dataset_config.feature_column,
        label_column=dataset_config.label_column,
        singer_column=dataset_config.singer_column,
        parquet_file=dataset_config.parquet_file,
        pad_sequences=True,
        add_channel_dim=True,
    )
    data = load_features_data(dc)

    params = MODEL_METADATA["default_params"].copy()
    params.update(bench_config.get_model_params("cnn"))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model, history = train(data, params, data.n_classes, callbacks)

    train_loss, train_acc = model.evaluate(data.X_train, data.y_train, verbose=0)
    val_loss, val_acc = model.evaluate(data.X_val, data.y_val, verbose=0)

    y_train_pred = np.argmax(model.predict(data.X_train, verbose=0), axis=1)
    y_val_pred = np.argmax(model.predict(data.X_val, verbose=0), axis=1)

    train_metrics = compute_metrics(data.y_train, y_train_pred, None, data.class_names)
    val_metrics = compute_metrics(data.y_val, y_val_pred, None, data.class_names)

    train_metrics["accuracy"] = float(train_acc)
    train_metrics["loss"] = float(train_loss)
    val_metrics["accuracy"] = float(val_acc)
    val_metrics["loss"] = float(val_loss)


    metrics_path = Path(dataset_config.metrics_path or output_dir / "metrics.csv")
    save_metrics_csv(val_metrics, metrics_path, "cnn", {"split": "validation"})
    save_metrics_csv(train_metrics, metrics_path, "cnn", {"split": "train"})

    hist_dict = history.history if history else {}
    plot_training_history(
        hist_dict,
        title="CNN",
        save_path=str(output_dir / "training_history.png"),
    )

    model.save(str(output_dir / "final_model.keras"))

    return {
        "model_name": "cnn",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": {k: [float(v) for v in vals] for k, vals in hist_dict.items()},
    }
