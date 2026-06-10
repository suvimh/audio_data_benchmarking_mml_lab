from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.evaluation.metrics import (
    compute_metrics,
    print_metrics_summary,
    save_metrics_csv,
)
from scripts.utils import ensure_dir

MODEL_METADATA = {
    "name": "clap",
    "supported_data_types": ["raw_audio"],
    "description": "CLAP embeddings + classifier (head_only or full finetuning)",
    "default_params": {
        "finetune_mode": "head_only",
        "learning_rate": 0.001,
        "encoder_learning_rate": 1e-6,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
        "frame_duration": 3.0,
        "overlap": 0.25,
        "sample_rate": 44100,
        "model_name": "laion/clap-htsat-fused",
    },
}

# ---------------------------------------------------------------------------
# Head-only: extract embeddings via msclap => train a linear classifier
# ---------------------------------------------------------------------------

def _extract_embeddings_msclap(
    audio_dir: str,
    file_df,
    sample_rate: int,
    frame_duration: float,
    overlap: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    from msclap import CLAP

    clap_model = CLAP(version="2023", use_cuda=device.type == "cuda")

    all_embeddings = []
    all_labels = []

    for _, row in tqdm(file_df.iterrows(), desc="Extracting CLAP embeddings", total=len(file_df)):
        import librosa

        audio, _ = librosa.load(row["filepath"], sr=sample_rate, mono=True)
        frame_len = int(sample_rate * frame_duration)
        hop_len = int(frame_len * (1 - overlap))

        if len(audio) < frame_len:
            audio = np.pad(audio, (0, frame_len - len(audio)))

        frames = []
        for start in range(0, len(audio) - frame_len + 1, hop_len):
            frames.append(audio[start: start + frame_len])

        for frame in frames:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, frame, sample_rate)
                tmp_path = tmp.name

            try:
                emb = clap_model.get_audio_embedding([tmp_path], resample=sample_rate)
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                if emb.ndim == 2:
                    emb = emb[0]
                all_embeddings.append(emb)
                all_labels.append(row["label"])
            finally:
                os.unlink(tmp_path)

    return {"embeddings": np.array(all_embeddings), "labels": np.array(all_labels)}


def _load_or_extract_embeddings(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    from scripts.data.raw_audio import scan_audio_files

    singer_map = _get_singer_map(dataset_config)
    file_df = scan_audio_files(
        audio_dir=dataset_config.audio_dir or dataset_config.data_dir,
        train_singers=singer_map["train"],
        val_singers=singer_map["val"],
        gender_filter=dataset_config.gender_split,
        include_labels=dataset_config.include_labels,
        exclude_labels=dataset_config.exclude_labels,
    )

    classes = _resolve_classes(file_df, dataset_config)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    params = MODEL_METADATA["default_params"].copy()
    params.update(bench_config.get_model_params("clap"))

    cache_path = output_dir / "embeddings.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        train_data = cached["train_embeddings"]
        train_labels = cached["train_labels"]
        val_data = cached["val_embeddings"]
        val_labels = cached["val_labels"]
    else:
        train_df = file_df["train"]
        val_df = file_df["val"]
        train_df = train_df[train_df["label"].isin(classes)]
        val_df = val_df[val_df["label"].isin(classes)]

        print("Extracting training embeddings...")
        train_result = _extract_embeddings_msclap(
            dataset_config.audio_dir or dataset_config.data_dir,
            train_df,
            params["sample_rate"],
            params["frame_duration"],
            params["overlap"],
            device,
        )
        print("Extracting validation embeddings...")
        val_result = _extract_embeddings_msclap(
            dataset_config.audio_dir or dataset_config.data_dir,
            val_df,
            params["sample_rate"],
            params["frame_duration"],
            params["overlap"],
            device,
        )

        train_data = train_result["embeddings"]
        train_labels = np.array([class_to_idx[l] for l in train_result["labels"]])
        val_data = val_result["embeddings"]
        val_labels = np.array([class_to_idx[l] for l in val_result["labels"]])

        np.savez_compressed(
            cache_path,
            train_embeddings=train_data,
            train_labels=train_labels,
            val_embeddings=val_data,
            val_labels=val_labels,
        )

    return {
        "train_embeddings": train_data,
        "train_labels": train_labels,
        "val_embeddings": val_data,
        "val_labels": val_labels,
        "class_names": classes,
        "n_classes": len(classes),
    }


class ClapLinearHead(nn.Module):
    def __init__(self, embedding_dim: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        return self.classifier(x)


def _run_head_only(
    dataset_config, bench_config, output_dir, device, params
) -> Dict[str, Any]:
    data = _load_or_extract_embeddings(dataset_config, bench_config, output_dir, device)

    embedding_dim = data["train_embeddings"].shape[1]

    train_tensor = torch.FloatTensor(data["train_embeddings"])
    train_labels = torch.LongTensor(data["train_labels"])
    val_tensor = torch.FloatTensor(data["val_embeddings"])
    val_labels = torch.LongTensor(data["val_labels"])

    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset = TensorDataset(val_tensor, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    model = ClapLinearHead(
        embedding_dim=embedding_dim,
        n_classes=data["n_classes"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_preds, val_true = [], []
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_acc = (np.array(val_preds) == np.array(val_true)).mean()

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

    model.load_state_dict(torch.load(str(output_dir / "best_model.pt")))
    model.eval()

    train_preds, train_true = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(y.numpy())

    val_preds, val_true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(y.numpy())

    train_metrics = compute_metrics(np.array(train_true), np.array(train_preds), None, data["class_names"])
    val_metrics = compute_metrics(np.array(val_true), np.array(val_preds), None, data["class_names"])

    print_metrics_summary(train_metrics, "CLAP head_only (train)")
    print_metrics_summary(val_metrics, "CLAP head_only (validation)")

    metrics_path = output_dir / "metrics.csv"
    save_metrics_csv(val_metrics, metrics_path, "clap_head_only",
                     {"split": "validation",
                      "frame_duration": dataset_config.frame_duration,
                      "overlap": dataset_config.overlap})
    save_metrics_csv(train_metrics, metrics_path, "clap_head_only",
                     {"split": "train",
                      "frame_duration": dataset_config.frame_duration,
                      "overlap": dataset_config.overlap})

    return {
        "model_name": "clap_head_only",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Full fine-tuning: HuggingFace CLAP, train encoder + classifier end-to-end
# ---------------------------------------------------------------------------

class ClapForClassification(nn.Module):
    def __init__(self, model_name: str, n_classes: int):
        super().__init__()
        from transformers import ClapModel

        self.clap = ClapModel.from_pretrained(model_name)
        hidden_size = self.clap.config.audio_config.hidden_size
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_values):
        outputs = self.clap.audio_encoder(input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


class ClapFinetuneDataset(Dataset):
    def __init__(
        self,
        filepaths: List[str],
        labels: List[int],
        model_name: str,
        sample_rate: int = 48000,
        frame_duration: float = 3.0,
        overlap: float = 0.25,
    ):
        self.filepaths = filepaths
        self.labels = labels
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.overlap = overlap

        from transformers import ClapProcessor
        self.processor = ClapProcessor.from_pretrained(model_name)
        self._cache = {}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        import librosa

        filepath = self.filepaths[idx]
        audio, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)

        frame_len = int(self.sample_rate * self.frame_duration)
        hop_len = int(frame_len * (1 - self.overlap))

        if len(audio) < frame_len:
            audio = np.pad(audio, (0, frame_len - len(audio)))

        inputs = self.processor(
            audios=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = inputs.input_values.squeeze(0)

        label = self.labels[idx]
        result = (input_values, label)
        self._cache[idx] = result
        return result


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for input_values, labels in dataloader:
        input_values = input_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_values)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for input_values, labels in dataloader:
            input_values = input_values.to(device)
            labels = labels.to(device)

            outputs = model(input_values)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels)


def _run_full_finetune(
    dataset_config, bench_config, output_dir, device, params
) -> Dict[str, Any]:
    from scripts.data.raw_audio import scan_audio_files

    singer_map = _get_singer_map(dataset_config)
    file_df = scan_audio_files(
        audio_dir=dataset_config.audio_dir or dataset_config.data_dir,
        train_singers=singer_map["train"],
        val_singers=singer_map["val"],
        gender_filter=dataset_config.gender_split,
        include_labels=dataset_config.include_labels,
        exclude_labels=dataset_config.exclude_labels,
    )

    classes = _resolve_classes(file_df, dataset_config)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = file_df["train"]
    val_df = file_df["val"]
    train_df = train_df[train_df["label"].isin(classes)]
    val_df = val_df[val_df["label"].isin(classes)]

    clap_sr = 48000

    train_dataset = ClapFinetuneDataset(
        filepaths=train_df["filepath"].tolist(),
        labels=[class_to_idx[l] for l in train_df["label"]],
        model_name=params["model_name"],
        sample_rate=clap_sr,
        frame_duration=params["frame_duration"],
        overlap=params["overlap"],
    )

    val_dataset = ClapFinetuneDataset(
        filepaths=val_df["filepath"].tolist(),
        labels=[class_to_idx[l] for l in val_df["label"]],
        model_name=params["model_name"],
        sample_rate=clap_sr,
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

    model = ClapForClassification(
        model_name=params["model_name"],
        n_classes=len(classes),
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.clap.audio_encoder.parameters(), "lr": params.get("encoder_learning_rate", 1e-6)},
        {"params": model.classifier.parameters(), "lr": params["learning_rate"]},
    ])

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(params["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_preds, val_labels = validate(model, val_loader, device)
        val_acc = (val_preds == val_labels).mean()

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

    model.load_state_dict(torch.load(str(output_dir / "best_model.pt")))
    model.eval()

    _, train_preds, train_labels = validate(model, train_loader, device)
    _, val_preds, val_labels = validate(model, val_loader, device)

    train_metrics = compute_metrics(train_labels, train_preds, None, classes)
    val_metrics = compute_metrics(val_labels, val_preds, None, classes)

    print_metrics_summary(train_metrics, "CLAP full_finetune (train)")
    print_metrics_summary(val_metrics, "CLAP full_finetune (validation)")

    metrics_path = output_dir / "metrics.csv"
    save_metrics_csv(val_metrics, metrics_path, "clap_full",
                     {"split": "validation",
                      "frame_duration": dataset_config.frame_duration,
                      "overlap": dataset_config.overlap})
    save_metrics_csv(train_metrics, metrics_path, "clap_full",
                     {"split": "train",
                      "frame_duration": dataset_config.frame_duration,
                      "overlap": dataset_config.overlap})

    return {
        "model_name": "clap_full",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = MODEL_METADATA["default_params"].copy()
    params.update(bench_config.get_model_params("clap"))

    params["frame_duration"] = dataset_config.frame_duration
    params["overlap"] = dataset_config.overlap

    finetune_mode = params.get("finetune_mode", "head_only")

    if finetune_mode == "head_only":
        return _run_head_only(dataset_config, bench_config, output_dir, device, params)
    elif finetune_mode == "full":
        return _run_full_finetune(dataset_config, bench_config, output_dir, device, params)
    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}. Use 'head_only' or 'full'.")


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
