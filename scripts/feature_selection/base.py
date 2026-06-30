from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

OPENSMILE_TOP_K_FEATURES = 100
OPENSMILE_DEFAULT_FILTER_PARAMS: Dict[str, Any] = {
    "variance_threshold": 0.0,
    "k": OPENSMILE_TOP_K_FEATURES,
}


@dataclass
class FeatureSelectionConfig:
    enabled: bool = False
    method: str = "filter"
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSelectionConfig:
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def opensmile_default(cls) -> FeatureSelectionConfig:
        return cls(
            enabled=True,
            method="filter",
            params=dict(OPENSMILE_DEFAULT_FILTER_PARAMS),
        )


def is_opensmile_dataset(feature_type: Optional[str], name: Optional[str] = None) -> bool:
    if feature_type and feature_type.startswith("opensmile"):
        return True
    return name == "opensmile"


def resolve_feature_selection(
    feature_type: Optional[str],
    name: Optional[str],
    benchmark_feature_selection: Optional[FeatureSelectionConfig],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if benchmark_feature_selection is not None:
        if benchmark_feature_selection.enabled:
            return benchmark_feature_selection.method, benchmark_feature_selection.params
        return None

    if is_opensmile_dataset(feature_type, name):
        return "filter", dict(OPENSMILE_DEFAULT_FILTER_PARAMS)

    return None
