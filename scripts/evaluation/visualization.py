from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
    figsize=(10, 8),
) -> None:
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_training_history(
    history: dict,
    title: str = "Training History",
    save_path: Optional[str | Path] = None,
) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric in zip(axes, ["loss", "accuracy"]):
        train_vals = history.get(metric, [])
        val_vals = history.get(f"val_{metric}", [])
        if train_vals:
            ax.plot(epochs[: len(train_vals)], train_vals, "b-", label=f"train_{metric}")
        if val_vals:
            ax.plot(epochs[: len(val_vals)], val_vals, "r-", label=f"val_{metric}")
        ax.set_title(f"{title} - {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
