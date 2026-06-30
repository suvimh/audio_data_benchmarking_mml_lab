from __future__ import annotations

import numpy as np

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.data.features import _maybe_apply_feature_selection
from scripts.feature_selection.base import (
    OPENSMILE_TOP_K_FEATURES,
    FeatureSelectionConfig,
    is_opensmile_dataset,
    resolve_feature_selection,
)


def test_is_opensmile_dataset_detects_feature_type_and_name() -> None:
    assert is_opensmile_dataset("opensmile-compare-2016", "opensmile")
    assert is_opensmile_dataset("opensmile-egemapsv02", "custom")
    assert not is_opensmile_dataset("whisper-base", "whisper")
    assert is_opensmile_dataset(None, "opensmile")


def test_resolve_feature_selection_defaults_for_opensmile() -> None:
    resolved = resolve_feature_selection("opensmile-compare-2016", "opensmile", None)
    assert resolved is not None
    method, params = resolved
    assert method == "filter"
    assert params["k"] == OPENSMILE_TOP_K_FEATURES


def test_resolve_feature_selection_respects_explicit_disable() -> None:
    fs = FeatureSelectionConfig(enabled=False)
    resolved = resolve_feature_selection("opensmile-compare-2016", "opensmile", fs)
    assert resolved is None


def test_resolve_feature_selection_skips_non_opensmile() -> None:
    assert resolve_feature_selection("whisper-base", "whisper", None) is None


def test_maybe_apply_feature_selection_reduces_opensmile_to_top_k() -> None:
    rng = np.random.default_rng(42)
    n_samples = 40
    n_features = 6373
    X_train = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    X_val = rng.standard_normal((10, n_features), dtype=np.float32)
    y_train = np.array([0, 1] * (n_samples // 2), dtype=np.int32)

    dataset_config = DatasetConfig(
        name="opensmile",
        data_type="features",
        feature_type="opensmile-compare-2016",
    )

    X_train_sel, X_val_sel, fs_info, preprocessed, input_dim = _maybe_apply_feature_selection(
        X_train,
        X_val,
        y_train,
        dataset_config,
        None,
        n_features,
    )

    assert preprocessed is True
    assert input_dim == OPENSMILE_TOP_K_FEATURES
    assert X_train_sel.shape == (n_samples, OPENSMILE_TOP_K_FEATURES)
    assert X_val_sel.shape == (10, OPENSMILE_TOP_K_FEATURES)
    assert fs_info is not None
    assert fs_info["n_features_before"] == n_features
    assert fs_info["n_features_after"] == OPENSMILE_TOP_K_FEATURES
    assert len(fs_info["selected_feature_indices"]) == OPENSMILE_TOP_K_FEATURES


def test_maybe_apply_feature_selection_leaves_non_opensmile_unchanged() -> None:
    rng = np.random.default_rng(7)
    n_features = 512
    X_train = rng.standard_normal((20, n_features), dtype=np.float32)
    X_val = rng.standard_normal((5, n_features), dtype=np.float32)
    y_train = np.zeros(20, dtype=np.int32)

    dataset_config = DatasetConfig(
        name="whisper",
        data_type="features",
        feature_type="whisper-base",
    )

    X_train_out, X_val_out, fs_info, preprocessed, input_dim = _maybe_apply_feature_selection(
        X_train,
        X_val,
        y_train,
        dataset_config,
        BenchmarkConfig(name="trad_ml", models=["knn"]),
        n_features,
    )

    assert preprocessed is False
    assert input_dim == n_features
    assert fs_info is None
    assert X_train_out.shape == X_train.shape
    assert X_val_out.shape == X_val.shape


if __name__ == "__main__":
    test_is_opensmile_dataset_detects_feature_type_and_name()
    test_resolve_feature_selection_defaults_for_opensmile()
    test_resolve_feature_selection_respects_explicit_disable()
    test_resolve_feature_selection_skips_non_opensmile()
    test_maybe_apply_feature_selection_reduces_opensmile_to_top_k()
    test_maybe_apply_feature_selection_leaves_non_opensmile_unchanged()
    print("All OpenSMILE feature-selection tests passed.")
