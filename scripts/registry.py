from __future__ import annotations

from typing import Any, Callable, Dict, List

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig

ModelRunFn = Callable[[DatasetConfig, BenchmarkConfig, str], Dict[str, Any]]


_registry: Dict[str, Dict[str, Any]] = {}


def register_model(name: str, supported_data_types: List[str], run_fn: ModelRunFn, metadata: dict = None):
    _registry[name] = {
        "name": name,
        "supported_data_types": supported_data_types,
        "run": run_fn,
        "metadata": metadata or {},
    }


def get_model(name: str) -> Dict[str, Any]:
    if name not in _registry:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list_models()}"
        )
    return _registry[name]


def list_models() -> List[Dict[str, Any]]:
    return [
        {
            "name": info["name"],
            "supported_data_types": info["supported_data_types"],
            "description": info["metadata"].get("description", ""),
        }
        for info in _registry.values()
    ]


def supports_data_type(model_name: str, data_type: str) -> bool:
    model = get_model(model_name)
    return data_type in model["supported_data_types"]


def _auto_register():
    try:
        from scripts.models import knn
        _register_model_from_module("knn", knn)
    except ImportError:
        pass
    try:
        from scripts.models import rf
        _register_model_from_module("rf", rf)
    except ImportError:
        pass
    try:
        from scripts.models import mlp
        _register_model_from_module("mlp", mlp)
    except ImportError:
        pass
    try:
        from scripts.models import cnn
        _register_model_from_module("cnn", cnn)
    except ImportError:
        pass
    try:
        from scripts.models import imagenet
        _register_model_from_module("imagenet", imagenet)
    except ImportError:
        pass
    try:
        from scripts.models import whisper
        _register_model_from_module("whisper", whisper)
    except ImportError:
        pass
    try:
        from scripts.models import clap
        _register_model_from_module("clap", clap)
    except ImportError:
        pass
    try:
        from scripts.models import svm
        _register_model_from_module("svm", svm)
    except ImportError:
        pass


def _register_model_from_module(name: str, module):
    meta = getattr(module, "MODEL_METADATA", {})
    supported = meta.get("supported_data_types", [])
    run_fn = getattr(module, "run", None)
    if run_fn:
        register_model(name, supported, run_fn, meta)


_auto_register()
