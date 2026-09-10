from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from scripts.evaluation.metrics_utils import get_hardware_info, get_model_complexity

# Hardware doesn't change mid-run, so query it once and reuse it for every
# row rather than re-shelling out to nvidia-smi/psutil per model.
_HARDWARE_INFO_CACHE: Optional[Dict] = None


def _cached_hardware_info() -> Dict:
    global _HARDWARE_INFO_CACHE
    if _HARDWARE_INFO_CACHE is None:
        _HARDWARE_INFO_CACHE = get_hardware_info()
    return _HARDWARE_INFO_CACHE


def log_environment_snapshot(output_path: str | Path) -> None:
    """Writes `pip freeze` output to disk. Call this once at the start of
    every experiment session and commit the file alongside your results —
    this is what was missing last time and caused the untraceable VGGish
    discrepancy."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    output_path.write_text(freeze.stdout)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }

    if class_names is not None:
        metrics["class_names"] = class_names
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        metrics["per_class_accuracy"] = {
            name: float(acc) for name, acc in zip(class_names, per_class_acc)
        }

    if y_prob is not None:
        top2 = np.mean(
            np.sum(np.argsort(y_prob, axis=1)[:, -2:] == y_true[:, None], axis=1)
        )
        top3 = np.mean(
            np.sum(np.argsort(y_prob, axis=1)[:, -3:] == y_true[:, None], axis=1)
        )
        metrics["top2_accuracy"] = float(top2)
        metrics["top3_accuracy"] = float(top3)

    return metrics


def _get_report_f1(report: dict) -> float | str:
    if not isinstance(report, dict):
        return ""
    for key in ("macro avg", "weighted avg", "accuracy"):
        if isinstance(report.get(key), dict):
            value = report[key].get("f1-score")
            if value is not None:
                return float(value)
    return ""


def _metric_value(metrics: Optional[Dict], key: str):
    if not metrics:
        return ""
    value = metrics.get(key, "")
    if value == "" or value is None:
        return ""
    return float(value)


def save_metrics_csv(
    output_path: str | Path,
    model_name: str,
    input_data: str,
    train_metrics: Optional[Dict] = None,
    val_metrics: Optional[Dict] = None,
    extra: Optional[Dict] = None,
    complexity: Optional[Dict] = None,
    include_hardware: bool = True,
) -> None:
    """
    complexity: output of metrics_utils.get_model_complexity(...), e.g.
        complexity = get_model_complexity(
            "random_forest", rf_model, n_features=128, n_train=len(X_train)
        )
        save_metrics_csv(..., complexity=complexity)
    Params/FLOPs are hardware-independent, so they stay comparable even if
    later rows are produced on a different machine than earlier ones.

    include_hardware: attaches cached CPU/GPU/RAM/platform info to the row
    (set False if you're logging hardware separately at the run level
    instead of per-row).
    """
    row = {
        "Model": model_name,
        "Input Data": input_data,
        "Train Accuracy": _metric_value(train_metrics, "accuracy"),
        "Train Balanced Accuracy": _metric_value(train_metrics, "balanced_accuracy"),
        "Train F1": (
            _get_report_f1(train_metrics.get("classification_report", {}))
            if train_metrics
            else ""
        ),
        "Train Top-2": _metric_value(train_metrics, "top2_accuracy"),
        "Train Top-3": _metric_value(train_metrics, "top3_accuracy"),
        "Val Accuracy": _metric_value(val_metrics, "accuracy"),
        "Val Balanced Accuracy": _metric_value(val_metrics, "balanced_accuracy"),
        "Val F1": (
            _get_report_f1(val_metrics.get("classification_report", {}))
            if val_metrics
            else ""
        ),
        "Val Top-2": _metric_value(val_metrics, "top2_accuracy"),
        "Val Top-3": _metric_value(val_metrics, "top3_accuracy"),
        "Train+Eval Time (s)": extra.get("Train+Eval Time (s)", "") if extra else "",
        "Epochs": extra.get("Epochs", "") if extra else "",
        "Batch Size": extra.get("Batch Size", "") if extra else "",
        "n_samples train": _metric_value(train_metrics, "n_samples"),
        "n_samples_val": _metric_value(val_metrics, "n_samples"),
    }

    for split, metrics in (("Train", train_metrics), ("Val", val_metrics)):
        for class_name, accuracy in (metrics or {}).get("per_class_accuracy", {}).items():
            row[f"{split} {class_name} Accuracy"] = accuracy

    if complexity:
        # prefix keys so they're unambiguous in the CSV (e.g. Total Params,
        # FLOPs) regardless of which model-type branch produced them
        for key, value in complexity.items():
            column = key.replace("_", " ").title()
            row[column] = value

    if include_hardware:
        hw = _cached_hardware_info()
        row["CPU"] = hw.get("processor", "")
        row["RAM (GB)"] = hw.get("ram_gb", "")
        row["GPU"] = hw.get("gpu", "")
        row["Platform"] = hw.get("platform", "")

    if extra:
        for key, value in extra.items():
            if key not in row:
                row[key] = value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if {"Model", "Input Data"}.issubset(existing.columns):
            existing = existing[
                ~(
                    (existing["Model"] == model_name)
                    & (existing["Input Data"] == input_data)
                )
            ]
        else:
            existing = existing.copy()
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(output_path, index=False)


def classification_report_to_df(report: dict) -> pd.DataFrame:
    rows = []
    for label, scores in report.items():
        if isinstance(scores, dict):
            rows.append(
                {
                    "class": label,
                    "precision": scores.get("precision", ""),
                    "recall": scores.get("recall", ""),
                    "f1-score": scores.get("f1-score", ""),
                    "support": scores.get("support", ""),
                }
            )
    return pd.DataFrame(rows)
