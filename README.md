# Vocal Technique Classification of VocalSet using Traditional Machine Learning and Transfer Learning

Overview
--------

Companion repository for the paper "A Systematic Benchmarking Across Feature-Model Pairings for Vocal Technique Classification" by Suvi Haara and Rafael Ramirez (Universitat Pompeu Fabra). Benchmarks Vocal Technique Classification (VTC) on the VocalSet dataset across multiple feature-model pairings.

The original VocalSet dataset is available at https://zenodo.org/records/1193957

Repository Structure
--------------------

```
scripts/                          # Main benchmarking framework
  __main__.py                     # python -m scripts <command>
  cli.py                          # CLI entry point
  run_benchmark.py                # Experiment orchestrator
  registry.py                     # Model registration and lookup
  validate_config.py              # Config validation
  utils.py                        # Shared utilities
  config/
    dataset_config.py             # Dataset configuration dataclass
    benchmark_config.py           # Experiment configuration dataclass
  data/
    loader.py                     # Data loading dispatcher
    features.py                   # Feature data loader (MFCC, VGGish, etc.)
    raw_audio.py                  # Raw audio scanning and loading
    images.py                     # Image data loader (mel-spectrograms)
  models/
    knn.py                        # K-Nearest Neighbors classifier
    rf.py                         # Random Forest classifier
    mlp.py                        # Multi-Layer Perceptron classifier
    cnn_vggish.py                 # 3-layer CNN on VGGish features
    imagenet.py                   # ImageNet transfer learning (ResNet/MobileNet/EfficientNet)
    whisper.py                    # Whisper encoder + linear classifier
    clap.py                       # CLAP embeddings + MLP classifier
  evaluation/
    metrics.py                    # Metrics computation and CSV export
    visualization.py              # Confusion matrices and training plots
examples/                         # Example config files (Python + YAML)
experiments/                      # Dataset-specific experiment configs
  vocalset_features.py            # VocalSet experiments on pre-extracted features
  vocalset_audio.py               # VocalSet experiments on raw audio
  vocalset_images.py              # VocalSet experiments on mel-spectrograms
results/                          # Experiment outputs (metrics, models, plots)
models/                           # Saved model weights
data/                             # Data directory
```

Each model architecture (KNN, RF, MLP, CNN, ResNet, MobileNet, EfficientNet, Whisper, CLAP) has its own script under `scripts/models/` with a consistent interface (MODEL_METADATA + run() function).

Usage
-----

```bash
# List available models and their supported data types
python -m scripts list-models

# Validate configs without running
python -m scripts validate -d configs/dataset.py -b configs/benchmark.py

# Run a benchmark experiment
python -m scripts run -d configs/dataset.py -b configs/benchmark.py
```

Configuration
-------------

Experiments are defined by two config files:

**Dataset config** describes the data source:
- `data_type`: "features" (pre-extracted feature vectors), "raw_audio" (audio files), or "images" (spectrogram images)
- Classification setup: `include_labels`, `label_map`, `gender_split`
- Data paths for train/val splits

**Benchmark config** describes the experiment:
- `models`: which models to run
- `model_params`: per-model parameter overrides
- Training hyperparameters and output directory

Classification tasks (label subsets) are defined in the dataset config via `include_labels` or `label_map`, making the system dataset-agnostic.

Models and Data Type Compatibility
----------------------------------

| Model         | Features | Raw Audio | Images |
|---------------|----------|-----------|--------|
| KNN           | yes      |           |        |
| RF            | yes      |           |        |
| MLP           | yes      |           |        |
| CNN-VGGish    | yes      |           |        |
| ImageNet      |          |           | yes    |
| Whisper       |          | yes       |        |
| CLAP          |          | yes       |        |

Data
----

Pre-generated data for VocalSet experiments (MFCC .pkl, VGGish .pkl, mel-spectrogram .png, Whisper .pt, CLAP .npz) is available at:
https://drive.google.com/drive/folders/1o9jL03t5q60fptKuO95Gaw18gPoM1sun

Authors
-------

Suvi Haara (https://github.com/suvimh)
