from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


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
