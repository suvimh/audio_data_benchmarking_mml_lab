"""
metrics_utils.py

Hardware info, parameter counts, and FLOP estimation utilities for the
VTC benchmarking pipeline. Designed so each model's results row can be
enriched with a consistent, hardware-independent complexity metric
(params, FLOPs) alongside the hardware-dependent wall-clock timing.

Install (uni machine, once):
    pip install psutil thop fvcore

Usage pattern in your experiment loop:

    from metrics_utils import get_hardware_info, get_model_complexity

    hw = get_hardware_info()   # call once per session, store in a run-level log

    complexity = get_model_complexity(
        model_type="sklearn_knn", model=knn_model, n_features=39, n_train=1200
    )
    row = {**your_existing_metrics_dict, **complexity, **hw}
    results.append(row)
"""

import platform
import subprocess
import threading
import time

import psutil


# ---------------------------------------------------------------------------
# 1. Hardware info
# ---------------------------------------------------------------------------
def get_hardware_info() -> dict:
    """Collects CPU/RAM/GPU/software info. Call once per run and log it
    alongside (not repeated per-model-row, to avoid clutter) — or merge
    into every row if you want it queryable per-experiment."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
    }

    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        info["gpu"] = gpu_query.stdout.strip()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        info["gpu"] = "none_detected"

    return info


# ---------------------------------------------------------------------------
# 2. PyTorch models (Whisper, custom torch modules)
# ---------------------------------------------------------------------------
def count_torch_params_flops(model, input_shape, device="cpu"):
    """
    input_shape: shape of a SINGLE example, e.g. (1, 80, 3000) for Whisper
                 log-mel input (no batch dim needed — thop adds it).
    Requires: pip install thop
    """
    import torch
    from thop import profile

    model = model.to(device).eval()
    dummy_input = torch.randn(1, *input_shape).to(device)

    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    return {
        "total_params": int(params),
        "flops": int(macs * 2),  # thop reports MACs; 1 MAC = 2 FLOPs
        "macs": int(macs),
    }


# ---------------------------------------------------------------------------
# 3. Keras / TF models (VGGish CNN, ResNet-50, MobileNetV2, EfficientNetB0)
# ---------------------------------------------------------------------------
def count_keras_params_flops(model):
    """
    Params: read directly from the model.
    FLOPs: computed via TF's graph profiler (no extra install needed
           beyond tensorflow itself).
    """
    import tensorflow as tf

    trainable = int(
        sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    )
    non_trainable = int(
        sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    )

    # FLOPs via concrete function + profiler
    input_shape = model.input_shape
    if input_shape[0] is None:
        input_shape = (1,) + input_shape[1:]

    @tf.function
    def forward(x):
        return model(x)

    concrete = forward.get_concrete_function(
        tf.TensorSpec(input_shape, model.input.dtype)
    )
    frozen_func = (
        tf.python.framework.convert_to_constants.convert_variables_to_constants_v2(
            concrete
        )
    )
    graph = frozen_func.graph

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, options=opts)

    return {
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "total_params": trainable + non_trainable,
        "flops": int(flops.total_float_ops) if flops is not None else None,
    }


# ---------------------------------------------------------------------------
# 4. Sklearn models (KNN, RF, MLP) — "FLOPs" isn't standard here, so we
#    report the closest meaningful analogue per model type, plus an
#    inference-time FLOP estimate where it's well-defined.
# ---------------------------------------------------------------------------
def estimate_sklearn_complexity(
    model_type: str, model, n_features: int, n_train: int = None
):
    """
    model_type: one of "knn", "random_forest", "mlp"

    KNN has no trained parameters; cost is at inference time (distance
    computation against the stored training set), so we report that as
    "flops" and n_train (stored samples) as the size proxy.

    RF's natural complexity proxy is total number of nodes across all
    trees, since that is what's actually "learned" and stored.

    MLP FLOPs are computed analytically from layer sizes: a dense layer
    with in_features -> out_features costs 2 * in * out FLOPs (multiply +
    add) per forward pass, plus a bias add (out FLOPs).
    """
    if model_type == "knn":
        if n_train is None:
            n_train = getattr(model, "n_samples_fit_", None)
        # one inference call: distance to every stored point
        flops_per_query = 2 * n_features * n_train if n_train else None
        return {
            "total_params": 0,  # KNN stores data, not learned params
            "stored_samples": n_train,
            "flops_per_inference": flops_per_query,
        }

    elif model_type == "random_forest":
        total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
        max_depth = max(tree.tree_.max_depth for tree in model.estimators_)
        return {
            "total_params": total_nodes,  # proxy: nodes stored
            "n_estimators": len(model.estimators_),
            "max_tree_depth": max_depth,
            "flops_per_inference": len(model.estimators_)
            * max_depth,  # traversal cost, rough
        }

    elif model_type == "mlp":
        # sklearn MLPClassifier: model.coefs_ is a list of weight matrices
        total_params = sum(w.size for w in model.coefs_) + sum(
            b.size for b in model.intercepts_
        )
        flops = 0
        for w in model.coefs_:
            in_f, out_f = w.shape
            flops += 2 * in_f * out_f + out_f  # matmul + bias add
        return {
            "total_params": int(total_params),
            "flops_per_inference": int(flops),
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# 5. Single dispatcher for convenience
# ---------------------------------------------------------------------------
def get_model_complexity(model_type: str, model, **kwargs) -> dict:
    """
    model_type in {"torch", "keras", "knn", "random_forest", "mlp"}
    kwargs passed through to the relevant function
    (e.g. input_shape=..., n_features=..., n_train=...)
    """
    if model_type == "torch":
        return count_torch_params_flops(model, **kwargs)
    elif model_type == "keras":
        return count_keras_params_flops(model)
    elif model_type in ("knn", "random_forest", "mlp"):
        return estimate_sklearn_complexity(model_type, model, **kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# ---------------------------------------------------------------------------
# 6. Resource tracking during training
# ---------------------------------------------------------------------------
class ResourceTracker:
    """Tracks CPU %, RSS, and peak GPU memory during a training run.

    Usage::

        tracker = ResourceTracker()
        tracker.start()
        # ... training loop ...
        stats = tracker.stop()
        # stats = {"Avg CPU %": ..., "Peak RSS (GB)": ..., "Peak GPU Memory (GB)": ...}
    """

    def __init__(self, sample_interval: float = 1.0):
        self._sample_interval = sample_interval
        self._process = psutil.Process()
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._peak_gpu_mem: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self):
        self._process.cpu_percent(None)
        while not self._stop_event.is_set():
            self._cpu_samples.append(self._process.cpu_percent())
            mem = self._process.memory_info()
            self._rss_samples.append(mem.rss / (1024 ** 3))
            self._stop_event.wait(self._sample_interval)

    def start(self):
        self._cpu_samples = []
        self._rss_samples = []
        self._peak_gpu_mem = 0.0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        try:
            import torch
            if torch.cuda.is_available():
                self._peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except ImportError:
            pass

        avg_cpu = round(sum(self._cpu_samples) / len(self._cpu_samples), 1) if self._cpu_samples else 0.0
        peak_rss = round(max(self._rss_samples), 2) if self._rss_samples else 0.0

        return {
            "Avg CPU %": avg_cpu,
            "Peak RSS (GB)": peak_rss,
            "Peak GPU Memory (GB)": round(self._peak_gpu_mem, 2),
        }
