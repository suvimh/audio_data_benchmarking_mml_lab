from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def build_wrapper_selector(params: dict) -> RFE:
    n_features = params.get("n_features_to_select", 128)
    estimator_name = params.get("estimator", "logistic")
    random_state = params.get("random_state", 42)

    if estimator_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 50),
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        estimator = LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            random_state=random_state,
            class_weight="balanced",
        )

    return RFE(estimator=estimator, n_features_to_select=n_features, step=params.get("step", 1))
