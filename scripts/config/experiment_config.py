from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.config.partition_config import PartitionConfig


@dataclass
class ExperimentFilters:
    gender: Optional[str] = None
    include_labels: Optional[List[str]] = None
    exclude_labels: Optional[List[str]] = None
    label_map: Optional[Dict[str, int]] = None
    pad_sequences: Optional[bool] = None
    add_channel_dim: Optional[bool] = None


@dataclass
class ExperimentBlock:
    model_type: str
    benchmark_config: BenchmarkConfig
    datasets: List[DatasetConfig]
    partition: PartitionConfig
    filters: ExperimentFilters = field(default_factory=ExperimentFilters)
    frame_duration: float = 3.0
    overlap: float = 0.25


@dataclass
class ExperimentConfig:
    name: str
    blocks: List[ExperimentBlock]
    output_dir: str = "./results"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
