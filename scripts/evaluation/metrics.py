from __future__ import annotations

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
        top2 = np.mean(np.sum(np.argsort(y_prob, axis=1)[:, -2:] == y_true[:, None], axis=1))
        top3 = np.mean(np.sum(np.argsort(y_prob, axis=1)[:, -3:] == y_true[:, None], axis=1))
        metrics["top2_accuracy"] = float(top2)
        metrics["top3_accuracy"] = float(top3)

    return metrics


def save_metrics_csv(
    metrics: Dict,
    output_path: str | Path,
    model_name: str,
    extra: Optional[Dict] = None,
) -> None:
    report = metrics.get("classification_report", {})
    macro = report.get("macro avg", {})

    row = {
        "model": model_name,
        "accuracy": metrics.get("accuracy", ""),
        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "macro_precision": macro.get("precision", ""),
        "macro_recall": macro.get("recall", ""),
        "macro_f1": macro.get("f1-score", ""),
        "top2_accuracy": metrics.get("top2_accuracy", ""),
        "top3_accuracy": metrics.get("top3_accuracy", ""),
        "n_samples": metrics.get("n_samples", ""),
    }
    if extra:
        row.update(extra)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([row])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(output_path, index=False)


def print_metrics_summary(metrics: Dict, model_name: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"Model: {model_name}")
    print(f"{'=' * 50}")
    print(f"Accuracy:          {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A'):.4f}")
    print(f"Top-2 Accuracy:    {metrics.get('top2_accuracy', 'N/A'):.4f}")
    print(f"Top-3 Accuracy:    {metrics.get('top3_accuracy', 'N/A'):.4f}")
    print(f"Samples:           {metrics.get('n_samples', 'N/A')}")

    per_class = metrics.get("per_class_accuracy")
    if per_class:
        print(f"\nPer-class accuracy:")
        for cls, acc in per_class.items():
            print(f"  {cls:20s}: {acc:.4f}")


def classification_report_to_df(report: dict) -> pd.DataFrame:
    rows = []
    for label, scores in report.items():
        if isinstance(scores, dict):
            rows.append({
                "class": label,
                "precision": scores.get("precision", ""),
                "recall": scores.get("recall", ""),
                "f1-score": scores.get("f1-score", ""),
                "support": scores.get("support", ""),
            })
    return pd.DataFrame(rows)
