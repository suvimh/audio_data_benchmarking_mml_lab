from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_SVM_PARAM_GRID: Dict[str, List[Any]] = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "kernel": ["rbf", "linear"],
}

FAST_SVM_PARAM_GRID: Dict[str, List[Any]] = {
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", 0.01],
    "kernel": ["rbf", "linear"],
}


def _expand_grid(param_grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(param_grid.keys())
    for values in product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, values))


def search_svm_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    param_grid: Dict[str, List[Any]] | None = None,
    preprocessed: bool = False,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float]:
    """Grid-search SVM hyperparameters using validation macro-F1."""
    grid = param_grid or DEFAULT_SVM_PARAM_GRID
    best_score = -1.0
    best_params: Dict[str, Any] = {}

    for params in _expand_grid(grid):
        svm_params = {
            **params,
            "probability": True,
            "class_weight": "balanced",
            "random_state": seed,
        }
        steps = []
        if not preprocessed:
            steps.append(("scaler", StandardScaler()))
        steps.append(("svm", SVC(**svm_params)))
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        score = float(f1_score(y_val, y_pred, average="macro"))
        if score > best_score:
            best_score = score
            best_params = dict(params)

    return best_params, best_score
