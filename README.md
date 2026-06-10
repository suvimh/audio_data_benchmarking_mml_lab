# Audio Data Benchmarking

Reusable model implementations for audio classification used by the MML Lab at Universitat Pompeu Fabra. Supports traditional ML (KNN, SVM, RF, MLP, CNN on feature vectors), transfer learning (Whisper, CLAP fine-tuning on raw audio), and image-based models (ResNet50, MobileNetV2, EfficientNetB0 on mel-spectrograms).

## Repository Structure

```
scripts/
  cli.py, __main__.py           # CLI entry point (python -m scripts <command>)
  run_benchmark.py               # Experiment orchestrator + config resolution
  registry.py                    # Model auto-registration
  utils.py / validate_config.py  # Shared utilities
  config/
    dataset_config.py            # Pure data definition (no partitioning)
    partition_config.py          # Singer-independent train/test split
    benchmark_config.py          # Model-type-specific parameters
    experiment_config.py         # Orchestration: datasets x models x filters
  data/
    features.py                  # Feature data loader (parquet + pkl)
    raw_audio.py                 # Audio scanning and loading
    images.py                    # Image data loader (mel-spectrograms)
  models/
    knn.py                       # K-Nearest Neighbors
    svm.py                       # Support Vector Machine
    rf.py                        # Random Forest
    mlp.py                       # Multi-Layer Perceptron
    cnn.py                       # 3-layer CNN on padded sequential features
    imagenet.py                  # ResNet50 / MobileNetV2 / EfficientNetB0
    whisper.py                   # Whisper encoder + classifier (head or full)
    clap.py                      # CLAP embeddings (head_only) or full fine-tuning
  evaluation/
    metrics.py                   # Metrics computation and CSV export
    visualization.py             # Confusion matrices and training plots
examples/
  datasets/                      # Pure dataset definitions
  partitions/                    # Train/test split configs
  benchmarks/                    # Model-type-specific configs
  experiments/                   # Full experiment configs
experiments/
  vtc_vocalset/                  # VocalSet-specific configs + experiment
```

## Configuration System

Experiments use four config types, each with a single responsibility:

### 1. DatasetConfig — data definition only

Describes *where the data lives* and *what columns to use*, with no partitioning or filtering baked in:

```python
opensmile = DatasetConfig(
    name="opensmile",
    data_type="features",
    data_dir="output/opensmile",
    parquet_file="opensmile_egemapsv02_3.0s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
)
```

### 2. PartitionConfig — train/test split

Defines which singers/participants go into training vs testing. This is **shared across all experiments** to maintain singer-independent evaluation:

```python
partition = PartitionConfig(
    name="singer_independent",
    train_singer_ids=[f"f{i}" for i in range(1,10)] + [f"m{i}" for i in range(1,7)],
    test_singer_ids=["f10","f11","m7","m8","m9"],
    singer_column="singer",
)
```

### 3. BenchmarkConfig — model parameters

Declares which models to run and their hyperparameters, grouped by input data type:

```python
trad_ml = BenchmarkConfig(
    name="trad_ml",
    models=["knn", "svm", "rf", "mlp", "cnn"],
    model_params={
        "knn": {"n_neighbors": 5},
        "svm": {"kernel": "rbf", "C": 1.0},
        "cnn": {"filters": [32, 64, 128], "epochs": 100},
    },
)
```

### 4. ExperimentConfig — orchestration

Ties datasets, partitions, benchmarks, and filters together into runnable blocks. Audio-level parameters (`frame_duration`, `overlap`) are set here so they are **consistent across all models and datasets in a block**:

```python
experiment = ExperimentConfig(
    name="vtc_vocalset_full",
    blocks=[
        ExperimentBlock(
            model_type="raw_audio",
            benchmark_config=finetune,
            datasets=[whisper_audio, clap_audio],
            partition=partition,
            frame_duration=3.0,
            overlap=0.25,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        ),
    ],
)
```

The orchestrator **resolves** each block by merging the dataset definition + partition + filters + frame parameters into a full DatasetConfig, then runs each model in the benchmark config against it. The `frame_duration` and `overlap` values are propagated to every model's metrics output.

## Usage

### CLI (recommended for batch runs)

```bash
# List available models
python -m scripts list-models

# Run a full experiment (new recommended way)
python -m scripts run --experiment experiments/vtc_vocalset/vocalset_full_experiment.py

# Validate an experiment config
python -m scripts validate --experiment experiments/vtc_vocalset/vocalset_full_experiment.py

# Run a single dataset + benchmark pair (legacy mode still works)
python -m scripts run --dataset-config dataset.py --benchmark-config bench.py
```

### Notebook (interactive / ad-hoc)

Import the package directly in a Jupyter/Colab notebook and orchestrate experiments with full Python control:

```python
# Import configs and orchestrator
from scripts.config.dataset_config import DatasetConfig
from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import ExperimentConfig, ExperimentBlock, ExperimentFilters
from scripts.run_benchmark import run_benchmark

# Load or define configs inline
dataset = DatasetConfig(...)
benchmark = BenchmarkConfig(...)
partition = PartitionConfig(...)

# Option A: single run (legacy style)
metrics = run_benchmark(dataset, benchmark, output_dir="results/my_experiment")

# Option B: resolve and run with custom filters
from scripts.run_benchmark import resolve_dataset
resolved = resolve_dataset(dataset, partition, ExperimentFilters(gender="male"))
metrics = run_benchmark(resolved, benchmark, output_dir="results/male_only")
```

This is useful for interactive exploration, custom pre/post-processing, or chaining experiments with Python logic that would be awkward in a static config.

## Models and Data Type Compatibility

| Model      | Features | Raw Audio | Images |
|------------|----------|-----------|--------|
| KNN        | yes      |           |        |
| SVM        | yes      |           |        |
| RF         | yes      |           |        |
| MLP        | yes      |           |        |
| CNN        | yes      |           |        |
| ImageNet   |          |           | yes    |
| Whisper    |          | yes       |        |
| CLAP       |          | yes       |        |

## Designing an Experiment

1. **Define datasets** in a config file: path, parquet file, column mapping (pure data, no splits)
2. **Define the partition** with train/test singer IDs (shared across experiments)
3. **Define benchmark configs** with model lists and parameters
4. **Create an experiment config** that connects datasets, partition, benchmarks, and filters into blocks

Audio-level settings (`frame_duration`, `overlap`) are specified on each `ExperimentBlock` — this ensures every model in a block uses the same framing, making results comparable. These values are recorded in the metrics CSV outputs for traceability.

Filters control label subsets (e.g. VOCALSET\_10, PMT, TMT), gender filtering, and data reshaping. Filtering is applied **separately to train and test data** with no leakage.

Example VocalSet experiment:
```bash
python -m scripts run --experiment experiments/vtc_vocalset/vocalset_full_experiment.py
```

This runs trad ML (KNN, SVM, RF, MLP, CNN) on OpenSMILE/VGGish/MFCC, ImageNet models on mel-spectrograms, and Whisper/CLAP on raw audio — all with the same singer-independent partition.

## Data

Pre-generated data for VocalSet experiments is available at:
https://drive.google.com/drive/folders/1o9jL03t5q60fptKuO95Gaw18gPoM1sun

## Adding a New Model

1. Create `scripts/models/your_model.py`
2. Export `MODEL_METADATA` (name, supported\_data\_types, default\_params)
3. Export `run(dataset_config, benchmark_config, output_dir)` returning a metrics dict
4. The registry auto-registers it on import

## Authors

Suvi Haara (https://github.com/suvimh)
