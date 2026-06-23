from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.feature_selection.embedded import build_embedded_selector
from scripts.feature_selection.filter import build_filter_selector
from scripts.feature_selection.wrapper import build_wrapper_selector

_BUILDERS = {
    "filter": build_filter_selector,
    "wrapper": build_wrapper_selector,
    "embedded": build_embedded_selector,
}


def build_selector(method: str, params: Dict[str, Any]):
    if method not in _BUILDERS:
        raise ValueError(f"Unknown feature selection method: {method}. Choose from {_BUILDERS}")
    return _BUILDERS[method](params)


def apply_feature_selection(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    method: str,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)

    n_before = X_train.shape[1]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    selector = build_selector(method, params)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_val_sel = selector.transform(X_val_scaled)

    selected_indices = _extract_selected_indices(selector, n_before)
    info = {
        "method": method,
        "params": params,
        "n_features_before": n_before,
        "n_features_after": X_train_sel.shape[1],
        "selected_feature_indices": selected_indices,
    }
    return X_train_sel.astype(np.float32), X_val_sel.astype(np.float32), info


def _extract_selected_indices(selector, n_features: int) -> list[int]:
    if hasattr(selector, "get_support"):
        support = selector.get_support(indices=True)
        return support.tolist()

    if hasattr(selector, "named_steps"):
        for step in reversed(list(selector.named_steps.values())):
            if hasattr(step, "get_support"):
                mask = np.zeros(n_features, dtype=bool)
                if hasattr(selector.named_steps.get("variance", None), "get_support"):
                    var_mask = selector.named_steps["variance"].get_support()
                    reduced_indices = step.get_support(indices=True)
                    full_indices = np.where(var_mask)[0][reduced_indices]
                    return full_indices.tolist()
                return step.get_support(indices=True).tolist()

    return list(range(n_features))
