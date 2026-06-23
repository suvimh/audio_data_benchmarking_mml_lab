from __future__ import annotations

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline


def build_filter_selector(params: dict) -> Pipeline:
    variance_threshold = params.get("variance_threshold", 0.0)
    k = params.get("k", 500)
    return Pipeline([
        ("variance", VarianceThreshold(threshold=variance_threshold)),
        ("select_k", SelectKBest(score_func=f_classif, k=k)),
    ])
