from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

if platform.system() == "Darwin":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_tf_configured = False


def _configure_tensorflow() -> None:
    global _tf_configured
    if _tf_configured:
        return
    import tensorflow as tf

    if platform.system() == "Darwin":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    _tf_configured = True


if platform.system() == "Darwin":
    _configure_tensorflow()

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.evaluation.metrics import (
    compute_metrics,
    print_metrics_summary,
    save_metrics_csv,
)
from scripts.utils import ensure_dir, set_seed

MODEL_METADATA = {
    "name": "cnn",
    "supported_data_types": ["features"],
    "description": "3-layer CNN classifier on sequential or flat feature data",
    "default_params": {
        "filters": [32, 64, 128],
        "kernel_size": 3,
        "dense_units": 64,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
        "max_sequence_length": 128,
    },
}


def _scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    train_shape = X_train.shape
    val_shape = X_val.shape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(X_train.shape[0], -1)
    ).reshape(train_shape).astype(np.float32)
    X_val_scaled = scaler.transform(
        X_val.reshape(X_val.shape[0], -1)
    ).reshape(val_shape).astype(np.float32)
    return X_train_scaled, X_val_scaled


def _input_shape_from_data(X: np.ndarray) -> Tuple[int, ...]:
    if X.ndim == 3:
        return (X.shape[1], X.shape[2])
    if X.ndim == 4:
        return (X.shape[1], X.shape[2], X.shape[3])
    raise ValueError(f"Expected 3D or 4D CNN input, got shape {X.shape}")


def _build_model(
    input_shape: Tuple[int, ...],
    n_classes: int,
    params: Dict[str, Any],
):
    _configure_tensorflow()
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if len(input_shape) == 2:
        max_length = params.get("max_sequence_length", 512)
        if input_shape[0] > max_length:
            pool_size = int(np.ceil(input_shape[0] / max_length))
            x = layers.AveragePooling1D(pool_size=pool_size, strides=pool_size)(x)

        for n_filters in params["filters"]:
            x = layers.Conv1D(
                n_filters, params["kernel_size"], padding="same", activation="relu"
            )(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(params["dropout_rate"])(x)
        x = layers.GlobalAveragePooling1D()(x)
    else:
        for n_filters in params["filters"]:
            x = layers.Conv2D(
                n_filters, params["kernel_size"], padding="same", activation="relu"
            )(x)
            x = layers.MaxPooling2D(pool_size=(2, 1))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(params["dropout_rate"])(x)
        x = layers.GlobalAveragePooling2D()(x)

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


def _make_debug_callback():
    # #region agent log
    import json as _json
    import time as _time

    import tensorflow as _tf

    _log_path = Path("/Users/pubert/audio_new/audio_data_benchmarking_mml_lab/.cursor/debug-4b4427.log")

    def _emit(hypothesis_id: str, location: str, message: str, data: dict) -> None:
        try:
            with _log_path.open("a", encoding="utf-8") as _f:
                _f.write(_json.dumps({
                    "sessionId": "4b4427",
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(_time.time() * 1000),
                }) + "\n")
        except OSError:
            pass

    class _DbgCallback(_tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            _emit("A", "cnn.py:on_train_begin", "fit started", {})
            self._t_epoch = _time.time()

        def on_epoch_begin(self, epoch, logs=None):
            self._t_epoch = _time.time()
            self._t_batch = _time.time()
            _emit("C", "cnn.py:on_epoch_begin", "epoch begin", {"epoch": epoch})

        def on_train_batch_end(self, batch, logs=None):
            now = _time.time()
            if batch < 3 or batch % 10 == 0:
                _emit("A/B", "cnn.py:on_train_batch_end", "train batch end", {
                    "batch": batch,
                    "dt_s": round(now - self._t_batch, 3),
                })
            self._t_batch = now

        def on_test_begin(self, logs=None):
            self._t_val = _time.time()
            _emit("D", "cnn.py:on_test_begin", "validation begin", {})

        def on_test_end(self, logs=None):
            _emit("D", "cnn.py:on_test_end", "validation end", {
                "dt_s": round(_time.time() - self._t_val, 3),
            })

        def on_epoch_end(self, epoch, logs=None):
            _emit("C/E", "cnn.py:on_epoch_end", "epoch end", {
                "epoch": epoch,
                "epoch_s": round(_time.time() - self._t_epoch, 3),
                "logs": {k: float(v) for k, v in (logs or {}).items()},
            })

    return _DbgCallback()
    # #endregion


def train(data, params: Dict[str, Any], n_classes: int, callbacks=None) -> tuple:
    _configure_tensorflow()
    set_seed(42)
    X_train = data.X_train
    X_val = data.X_val
    if not data.preprocessed:
        X_train, X_val = _scale_features(X_train, X_val)

    input_shape = _input_shape_from_data(X_train)
    model = _build_model(input_shape, n_classes, params)

    fit_callbacks = list(callbacks or [])
    fit_callbacks.append(_make_debug_callback())  # agent log

    history = model.fit(
        X_train,
        data.y_train,
        validation_data=(X_val, data.y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=fit_callbacks,
        verbose=1,
    )
    return model, history, X_train, X_val


def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    from scripts.data.features import load_features_data

    _configure_tensorflow()
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
        train_singer_ids=dataset_config.train_singer_ids,
        val_singer_ids=dataset_config.val_singer_ids,
        parquet_file=dataset_config.parquet_file,
        pad_sequences=True,
        add_channel_dim=True,
    )
    data = load_features_data(dc, bench_config)

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

    model, history, X_train, X_val = train(data, params, data.n_classes, callbacks)

    train_loss, train_acc = model.evaluate(X_train, data.y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, data.y_val, verbose=0)

    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    train_metrics = compute_metrics(data.y_train, y_train_pred, None, data.class_names)
    val_metrics = compute_metrics(data.y_val, y_val_pred, None, data.class_names)

    train_metrics["accuracy"] = float(train_acc)
    train_metrics["loss"] = float(train_loss)
    val_metrics["accuracy"] = float(val_acc)
    val_metrics["loss"] = float(val_loss)

    print_metrics_summary(train_metrics, "CNN (train)")
    print_metrics_summary(val_metrics, "CNN (validation)")

    metrics_path = output_dir / "metrics.csv"
    save_metrics_csv(val_metrics, metrics_path, "cnn", {"split": "validation"})
    save_metrics_csv(train_metrics, metrics_path, "cnn", {"split": "train"})

    hist_dict = history.history if history else {}
    from scripts.evaluation.visualization import plot_training_history

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
