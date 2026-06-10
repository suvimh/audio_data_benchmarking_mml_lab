from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class PartitionConfig:
    name: str
    train_singer_ids: List[str]
    test_singer_ids: List[str]
    singer_column: str = "singer"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PartitionConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save_yaml(self, path: str | Path) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> PartitionConfig:
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def save_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> PartitionConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
