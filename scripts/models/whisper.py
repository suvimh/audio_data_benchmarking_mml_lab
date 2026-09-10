from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.evaluation.metrics import (
    compute_metrics,
    save_metrics_csv,
)
from scripts.utils import ensure_dir, set_seed, load_pickle, save_pickle

MODEL_METADATA = {
    "name": "whisper",
    "supported_data_types": ["raw_audio"],
    "description": "Whisper encoder + linear classifier (HuggingFace transformers)",
    "default_params": {
        "model_name": "openai/whisper-base",
        "learning_rate": 5e-5,
        "encoder_learning_rate": 1e-6,
        "batch_size": 16,
        "epochs": 50,
        "patience": 10,
        "frame_duration": 3.0,
        "overlap": 0.25,
        "sample_rate": 16000,
        "freeze_encoder": True,
    },
}


class WhisperClassifier(nn.Module):
    def __init__(self, model_name: str, n_classes: int, freeze_encoder: bool = True):
        super().__init__()
        from transformers import WhisperModel

        self.whisper = WhisperModel.from_pretrained(model_name)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.whisper.parameters():
                param.requires_grad = False

        hidden_size = self.whisper.config.hidden_size
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_features):
        outputs = self.whisper.encoder(input_features)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


def _make_windows(audio: np.ndarray, frame_len: int, hop_len: int) -> list[np.ndarray]:
    if len(audio) < frame_len:
        return [np.pad(audio, (0, frame_len - len(audio)))]
    starts = list(range(0, len(audio) - frame_len + 1, hop_len))
    last_full_start = len(audio) - frame_len
    if not starts or starts[-1] != last_full_start:
        starts.append(last_full_start)
    return [audio[s : s + frame_len] for s in starts]


class WhisperDataset(Dataset):
    """Splits each raw audio source file into overlapping frame_duration-second
    windows and produces one training sample per window.

    All windows and their labels are precomputed in __init__ so that each
    __getitem__ returns a fixed-length feature tensor for a single window.
    """

    def __init__(
        self,
        filepaths: List[str],
        labels: List[int],
        n_classes: int,
        model_name: str,
        sample_rate: int = 16000,
        frame_duration: float = 3.0,
        overlap: float = 0.25,
    ):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.overlap = overlap

        from transformers import WhisperFeatureExtractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        self._items = []
        for filepath, label in zip(filepaths, labels):
            self._items.extend(self._windows_for_file(filepath, label))

    def _windows_for_file(self, filepath: str, label: int) -> List[tuple]:
        import librosa

        audio, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)
        audio = np.asarray(audio, dtype=np.float32)

        frame_len = int(self.sample_rate * self.frame_duration)
        hop_len = max(1, int(frame_len * (1 - self.overlap)))
        windows = _make_windows(audio, frame_len, hop_len)

        items = []
        for window in windows:
            inputs = self.feature_extractor(
                window,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.squeeze(0)
            items.append((input_features, label))
        return items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        input_features, label = self._items[idx]
        return input_features, label


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for input_features, labels in dataloader:
        input_features = input_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_features)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    with torch.no_grad():
        for input_features, labels in dataloader:
            input_features = input_features.to(device)
            labels = labels.to(device)

            outputs = model(input_features)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return (
        total_loss / len(dataloader),
        np.array(all_preds),
        np.array(all_labels),
        np.concatenate(all_probs, axis=0) if all_probs else None,
    )


def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = MODEL_METADATA["default_params"].copy()
    params.update(bench_config.get_model_params("whisper"))

    params["frame_duration"] = dataset_config.frame_duration
    params["overlap"] = dataset_config.overlap
    params["sample_rate"] = dataset_config.sample_rate

    freeze_encoder = params.get("freeze_encoder", True)

    from scripts.data.raw_audio import scan_audio_files

    singer_map = _get_singer_map(dataset_config)
    file_df = scan_audio_files(
        audio_dir=dataset_config.audio_dir or dataset_config.data_dir,
        train_singers=singer_map["train"],
        val_singers=singer_map["val"],
        gender_filter=dataset_config.gender_split,
        include_labels=dataset_config.include_labels,
        exclude_labels=dataset_config.exclude_labels,
        level_names=dataset_config.level_names,
        label_level=dataset_config.label_level,
    )

    classes = _resolve_classes(file_df, dataset_config)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = file_df["train"]
    val_df = file_df["val"]
    train_df = train_df[train_df["label"].isin(classes)]
    val_df = val_df[val_df["label"].isin(classes)]

    train_dataset = WhisperDataset(
        filepaths=train_df["filepath"].tolist(),
        labels=[class_to_idx[l] for l in train_df["label"]],
        n_classes=len(classes),
        model_name=params["model_name"],
        sample_rate=params["sample_rate"],
        frame_duration=params["frame_duration"],
        overlap=params["overlap"],
    )

    val_dataset = WhisperDataset(
        filepaths=val_df["filepath"].tolist(),
        labels=[class_to_idx[l] for l in val_df["label"]],
        n_classes=len(classes),
        model_name=params["model_name"],
        sample_rate=params["sample_rate"],
        frame_duration=params["frame_duration"],
        overlap=params["overlap"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = WhisperClassifier(
        model_name=params["model_name"],
        n_classes=len(classes),
        freeze_encoder=freeze_encoder,
    ).to(device)

    if freeze_encoder:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params["learning_rate"],
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.whisper.parameters(), "lr": params.get("encoder_learning_rate", 1e-6)},
            {"params": model.classifier.parameters(), "lr": params["learning_rate"]},
        ])

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"loss": [], "val_loss": [], "val_accuracy": []}

    from scripts.evaluation.metrics_utils import ResourceTracker
    resource_tracker = ResourceTracker()
    resource_tracker.start()

    actual_epochs = 0
    for epoch in range(params["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_preds, val_labels, _ = validate(model, val_loader, device)
        val_acc = (val_preds == val_labels).mean()
        actual_epochs = epoch + 1

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(val_acc))

        print(f"Epoch {epoch + 1}/{params['epochs']} - "
              f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(output_dir / "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    resource_stats = resource_tracker.stop()

    model.load_state_dict(torch.load(str(output_dir / "best_model.pt")))
    model.eval()

    _, train_preds, train_labels, train_probs = validate(model, train_loader, device)
    _, val_preds, val_labels, val_probs = validate(model, val_loader, device)

    train_metrics = compute_metrics(train_labels, train_preds, train_probs, classes)
    val_metrics = compute_metrics(val_labels, val_preds, val_probs, classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    complexity = {
        "Backbone": f"Whisper ({params['model_name']})",
        "Classifier Head": f"Linear({model.whisper.config.hidden_size}, {len(classes)})",
        "Total Params": total_params,
        "Trainable Params": trainable_params,
        "Non-trainable Params": non_trainable_params,
        "Encoder Frozen": freeze_encoder,
    }

    metrics_path = Path(dataset_config.metrics_path or output_dir / "metrics.csv")
    save_metrics_csv(
        metrics_path,
        "whisper_full" if not freeze_encoder else "whisper_head_only",
        dataset_config.name,
        train_metrics,
        val_metrics,
        extra={
            "Epochs": actual_epochs,
            "Max Epochs": params.get("epochs", bench_config.epochs),
            "Batch Size": params.get("batch_size", bench_config.batch_size),
            "Train+Eval Time (s)": round(time.perf_counter() - start_time, 4),
            "frame_duration": dataset_config.frame_duration,
            "overlap": dataset_config.overlap,
            **resource_stats,
        },
        complexity=complexity,
    )

    return {
        "model_name": "whisper_full" if not freeze_encoder else "whisper_head_only",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
        "complexity": complexity,
        "resource_stats": resource_stats,
        "actual_epochs": actual_epochs,
        "max_epochs": params.get("epochs", bench_config.epochs),
        "batch_size": params.get("batch_size", bench_config.batch_size),
        "train_time_s": round(time.perf_counter() - start_time, 4),
    }


def _get_singer_map(dataset_config: DatasetConfig) -> Dict[str, List[str]]:
    if dataset_config.train_singer_ids and dataset_config.val_singer_ids:
        return {
            "train": dataset_config.train_singer_ids,
            "val": dataset_config.val_singer_ids,
        }
    return {"train": None, "val": None}


def _resolve_classes(file_df, dataset_config):
    if dataset_config.label_map:
        return sorted(dataset_config.label_map.keys(), key=lambda c: dataset_config.label_map[c])
    all_labels = set(file_df["train"]["label"].unique()) | set(file_df["val"]["label"].unique())
    return sorted(all_labels)
