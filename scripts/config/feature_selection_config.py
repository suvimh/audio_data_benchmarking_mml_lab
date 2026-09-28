'''
    Feature-selection variant of the traditional ML benchmark config.

    Wraps embedding (feature-vector) datasets with RFE or SelectKBest,
    sweeping over a list of target feature counts, then runs a set of
    traditional-ML classifiers on the reduced features.  Designed to
    follow the standard benchmark: run the original experiment first,
    then use this config to investigate which features matter and how
    performance scales with dimensionality.
'''
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.config.dataset_config import DatasetConfig
from scripts.config.experiment_config import ExperimentFilters
from scripts.config.partition_config import PartitionConfig


@dataclass
class ModelEmbeddingPair:
    """An embedding + a set of models to run against it."""
    dataset: DatasetConfig
    models: List[str]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset.to_dict(),
            "models": list(self.models),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelEmbeddingPair:
        ds = d.get("dataset")
        if isinstance(ds, DatasetConfig):
            dataset = ds
        elif isinstance(ds, dict):
            dataset = DatasetConfig(**ds)
        else:
            raise ValueError("ModelEmbeddingPair.from_dict: dataset must be dict or DatasetConfig")
        return cls(dataset=dataset, models=list(d.get("models", [])))


@dataclass
class FeatureSelectionExperimentConfig:
    """Configuration for a feature-selection experiment run.

    Each pair declares an embedding dataset and the models to evaluate.
    For every (pair, model, method, n_features) combination the experiment
    fits the selector on the training set only, transforms both splits,
    trains the classifier, and writes a CSV row (appended between runs).

    Attributes
    ----------
    name : str
        Human-readable experiment name; used as a subfolder and in CSV filenames.
    pairs : list[ModelEmbeddingPair]
        Embedding × model pairs to evaluate.  Edit this list (or subset
        it) to save compute — only declared pairs are run.
    output_dir : str
        Root directory; the CSV and plots go into ``<output_dir>/<name>/``.
    partition / filters / frame_duration / overlap
        Same semantics as ``ExperimentBlock``; merged onto every dataset
        before loading data.
    model_params : dict, optional
        Per-model keyword overrides, keyed by model name.  These are
        merged over each model module's ``MODEL_METADATA["default_params"]``
        at run time, exactly like ``BenchmarkConfig.get_model_params``.
    feature_selection_methods : list[str]
        Accepted values (case-insensitive): ``"rfe"``, ``"select_k_best"``.
    n_features_list : list[int]
        Candidate feature counts.  Counts ``>= total_features`` for a
        given embedding are silently skipped at run time.
    rfe_estimator : str
        Estimator used inside RFE.  ``"rf"`` (default) →
        ``RandomForestClassifier``; ``"svm"`` → ``LinearSVC``.
    rfe_estimator_params : dict, optional
        Keyword overrides passed to the RFE estimator constructor.
    rfe_step : float
        Fraction of features to remove per RFE iteration (default 0.1).
        Larger values are faster but may be less precise.
    seed : int
        Global random seed applied before every (model, method, n)
        combination.
    """

    name: str
    pairs: List[ModelEmbeddingPair]
    output_dir: str = "./metrics"
    partition: Optional[PartitionConfig] = None
    filters: ExperimentFilters = field(default_factory=ExperimentFilters)
    frame_duration: float = 3.0
    overlap: float = 0.25

    model_params: Optional[Dict[str, Dict[str, Any]]] = None
    feature_selection_methods: List[str] = field(
        default_factory=lambda: ["rfe", "select_k_best"]
    )
    n_features_list: List[int] = field(
        default_factory=lambda: [5, 10, 20, 40, 80, 160, 320, 640]
    )
    rfe_estimator: str = "rf"
    rfe_estimator_params: Optional[Dict[str, Any]] = None
    rfe_step: float = 0.1
    seed: int = 42

    # ------------------------------------------------------------------
    # Serialisation helpers (mirrors BenchmarkConfig / ExperimentConfig)
    # ------------------------------------------------------------------

    def get_model_params(self, model_name: str) -> dict:
        """Return per-model param overrides (or empty dict)."""
        if self.model_params and model_name in self.model_params:
            return dict(self.model_params[model_name])
        return {}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairs"] = [p.to_dict() for p in self.pairs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSelectionExperimentConfig:
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        # Reconstruct nested dataclasses when loaded from raw dicts
        if d.get("filters") and not isinstance(d["filters"], ExperimentFilters):
            d["filters"] = ExperimentFilters(**d["filters"])
        if d.get("partition") and not isinstance(d["partition"], PartitionConfig):
            d["partition"] = PartitionConfig(**d["partition"])
        if d.get("pairs"):
            d["pairs"] = [
                ModelEmbeddingPair.from_dict(p) if isinstance(p, dict) else p
                for p in d["pairs"]
            ]
        return cls(**d)

    def save_yaml(self, path: str | Path) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> FeatureSelectionExperimentConfig:
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def save_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> FeatureSelectionExperimentConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
