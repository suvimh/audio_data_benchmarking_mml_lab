from __future__ import annotations

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC


def build_embedded_selector(params: dict) -> SelectFromModel:
    C = params.get("C", 0.1)
    max_features = params.get("max_features")
    random_state = params.get("random_state", 42)

    estimator = LinearSVC(
        penalty="l1",
        dual=False,
        C=C,
        class_weight="balanced",
        random_state=random_state,
        max_iter=params.get("max_iter", 5000),
    )

    kwargs = {"estimator": estimator}
    if max_features is not None:
        kwargs["max_features"] = max_features
    else:
        kwargs["threshold"] = params.get("threshold", "mean")

    return SelectFromModel(**kwargs)
