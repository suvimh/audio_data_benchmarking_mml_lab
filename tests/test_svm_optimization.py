"""Tests for SVM optimization utilities."""
from __future__ import annotations

import numpy as np

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.data.concat_features import load_concat_features_data
from scripts.feature_selection.wrapper import build_wrapper_selector
from scripts.tuning.svm_grid import search_svm_hyperparams


def test_build_wrapper_selector_accepts_svm_estimator() -> None:
    selector = build_wrapper_selector(
        {"n_features_to_select": 5, "estimator": "svm", "step": 2}
    )
    X = np.random.randn(40, 20).astype(np.float32)
    y = np.array([0, 1] * 20, dtype=np.int32)
    selector.fit(X, y)
    assert selector.transform(X).shape[1] == 5


def test_search_svm_hyperparams_returns_params() -> None:
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(60, 8)).astype(np.float32)
    X_val = rng.normal(size=(20, 8)).astype(np.float32)
    y_train = np.array([0, 1, 2] * 20, dtype=np.int32)
    y_val = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int32)

    params, score = search_svm_hyperparams(
        X_train,
        y_train,
        X_val,
        y_val,
        param_grid={"C": [0.1, 1.0], "gamma": ["scale"], "kernel": ["linear"]},
    )
    assert "C" in params
    assert score >= 0.0


def test_load_concat_features_data_smoke() -> None:
    from experiments.tuni_emotion.tuni_data_featureset_embedding_configs import (
        make_clap_embedding_config,
        make_opensmile_embedding_config,
        make_whisper_embedding_config,
    )
    from experiments.tuni_emotion.tuni_experiments_constants import (
        TRAIN_SINGERS,
        VAL_SINGERS,
        TUNI_EMOTIONS,
    )

    base_kwargs = {
        "include_labels": TUNI_EMOTIONS,
        "label_column": "emotion",
        "train_singer_ids": TRAIN_SINGERS,
        "val_singer_ids": VAL_SINGERS,
        "singer_column": "singer",
    }
    configs = []
    for factory in (
        make_opensmile_embedding_config,
        make_whisper_embedding_config,
        make_clap_embedding_config,
    ):
        cfg = factory(frame_duration=3.0)
        for key, value in base_kwargs.items():
            setattr(cfg, key, value)
        configs.append(cfg)

    bench = BenchmarkConfig(
        name="test",
        models=["svm"],
        feature_selection=None,
    )
    data = load_concat_features_data(configs, bench)
    assert data.X_train.ndim == 2
    assert data.X_train.shape[0] > 0
    assert data.X_val.shape[0] > 0


if __name__ == "__main__":
    test_build_wrapper_selector_accepts_svm_estimator()
    test_search_svm_hyperparams_returns_params()
    test_load_concat_features_data_smoke()
    print("All SVM optimization tests passed.")
