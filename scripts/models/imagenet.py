from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.evaluation.metrics import (
    compute_metrics,
    print_metrics_summary,
    save_metrics_csv,
)
from scripts.evaluation.visualization import plot_training_history
from scripts.utils import ensure_dir

MODEL_METADATA = {
    "name": "imagenet",
    "supported_data_types": ["images"],
    "description": "ImageNet transfer learning (ResNet50, MobileNetV2, EfficientNetB0)",
    "default_params": {
        "backbone": "efficientnet",
        "learning_rate": 0.001,
        "dense_units": 128,
        "dropout_rate": 0.5,
        "epochs": 100,
        "batch_size": 32,
        "patience": 15,
    },
    "available_backbones": ["resnet50", "mobilenet", "efficientnet"],
}


def _build_model(input_shape, n_classes, params):
    import tensorflow as tf
    from tensorflow.keras import applications, layers

    backbone_name = params.get("backbone", "efficientnet")
    weights = "imagenet"

    if backbone_name == "resnet50":
        backbone = applications.ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    elif backbone_name == "mobilenet":
        backbone = applications.MobileNetV2(weights=weights, include_top=False, input_shape=input_shape)
    elif backbone_name == "efficientnet":
        backbone = applications.EfficientNetB0(weights=weights, include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.imagenet_utils.preprocess_input(inputs)
    x = backbone(x, training=False)
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


def train(data, params, n_classes, callbacks=None):
    import tensorflow as tf

    model = _build_model(data.input_shape, n_classes, params)

    history = model.fit(
        data.train_ds,
        validation_data=data.val_ds,
        epochs=params["epochs"],
        callbacks=callbacks or [],
        verbose=1,
    )
    return model, history


def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    from scripts.data.images import load_image_data

    import tensorflow as tf

    output_dir = ensure_dir(output_dir)
    data = load_image_data(dataset_config)

    params = MODEL_METADATA["default_params"].copy()
    params.update(bench_config.get_model_params("imagenet"))
    backbone = params.get("backbone", "efficientnet")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=params["patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / f"best_model_{backbone}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model, history = train(data, params, data.n_classes, callbacks)

    train_metrics = _evaluate_dataset(model, data.train_ds, data.class_names, "train")
    val_metrics = _evaluate_dataset(model, data.val_ds, data.class_names, "validation")

    print_metrics_summary(train_metrics, f"{backbone} (train)")
    print_metrics_summary(val_metrics, f"{backbone} (validation)")

    metrics_path = output_dir / "metrics.csv"
    save_metrics_csv(val_metrics, metrics_path, f"imagenet_{backbone}", {"split": "validation"})
    save_metrics_csv(train_metrics, metrics_path, f"imagenet_{backbone}", {"split": "train"})

    hist_dict = history.history if history else {}
    plot_training_history(
        hist_dict,
        title=f"{backbone}",
        save_path=str(output_dir / f"training_history_{backbone}.png"),
    )

    model.save(str(output_dir / f"final_model_{backbone}.keras"))

    return {
        "model_name": f"imagenet_{backbone}",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": {k: [float(v) for v in vals] for k, vals in hist_dict.items()},
    }


def _evaluate_dataset(model, dataset, class_names, split_name):
    all_preds, all_labels = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred, None, class_names)
    loss, acc = model.evaluate(dataset, verbose=0)
    metrics["accuracy"] = float(acc)
    metrics["loss"] = float(loss)
    return metrics
