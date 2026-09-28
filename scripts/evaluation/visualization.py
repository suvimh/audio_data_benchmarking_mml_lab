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
    plt.figure(figsize=figsize)
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
    plt.show()


def plot_feature_selection_curves(
    csv_path: str | Path,
    output_dir: str | Path,
    metric: str = "Val Balanced Accuracy",
    log_x: bool = True,
) -> None:
    """Line plots of a metric vs the number of selected features.

    Reads the cumulative CSV written by ``run_feature_selection`` and
    produces, for every embedding ("Input Data"):
      - one figure: one curve per model, line style per selection method.
      - a combined overview figure with one subplot per embedding.

    Args:
        csv_path: cumulative feature-selection metrics CSV.
        output_dir: where the generated PNGs are written.
        metric: CSV column to plot on the y-axis.
        log_x: log-scale the x-axis (feature counts double each step).
    """
    import pandas as pd

    from scripts.utils import safe_filename

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = {"Model", "Input Data", "Feature Selection Method", "N Features", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing columns required for plotting: {sorted(missing)}"
        )

    df = df.drop_duplicates(subset=["Model", "Input Data", "Feature Selection Method", "N Features"])
    df["N Features"] = pd.to_numeric(df["N Features"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["N Features", metric])

    if df.empty:
        print(f"  No numeric rows to plot in {csv_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = list(dict.fromkeys(df["Input Data"].astype(str)))
    metric_slug = safe_filename(metric).replace(" ", "_").lower()

    method_order = list(dict.fromkeys(df["Feature Selection Method"].astype(str)))
    model_order = list(dict.fromkeys(df["Model"].astype(str)))

    formatter = plt.FuncFormatter(lambda val, _pos: f"{int(val):g}")

    def _decorate(ax, sub, title):
        sns.lineplot(
            data=sub,
            x="N Features",
            y=metric,
            hue="Model",
            style="Feature Selection Method",
            markers=True,
            dashes=True,
            markersize=6,
            ax=ax,
        )
        if log_x:
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(sub["N Features"].dropna().unique().tolist()))
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Number of features")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    for emb in embeddings:
        sub = df[df["Input Data"].astype(str) == emb].copy()
        sub["Model"] = pd.Categorical(sub["Model"], categories=model_order)
        sub["Feature Selection Method"] = pd.Categorical(
            sub["Feature Selection Method"], categories=method_order
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        _decorate(ax, sub, f"{emb} — {metric} vs number of features")
        ax.legend(
            title="", fontsize=8, frameon=True, ncol=2, loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"feature_selection_{safe_filename(emb)}_{metric_slug}.png",
            dpi=200, bbox_inches="tight",
        )
        plt.close(fig)

    if len(embeddings) > 1:
        ncols = 2
        nrows = int(np.ceil(len(embeddings) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(13, 4.5 * nrows), squeeze=False
        )
        for ax, emb in zip(axes.ravel(), embeddings):
            sub = df[df["Input Data"].astype(str) == emb].copy()
            sub["Model"] = pd.Categorical(sub["Model"], categories=model_order)
            sub["Feature Selection Method"] = pd.Categorical(
                sub["Feature Selection Method"], categories=method_order
            )
            _decorate(ax, sub, emb)

        for ax in axes.ravel()[len(embeddings):]:
            ax.set_visible(False)

        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, fontsize=9, frameon=True, ncol=2,
            loc="upper center", bbox_to_anchor=(0.5, 1.0),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(
            output_dir / f"feature_selection_overview_{metric_slug}.png",
            dpi=200, bbox_inches="tight",
        )
        plt.close(fig)


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
    plt.show()
