'''
    Feature-selection experiment config for TUNI emotion (traditional-ML stages).

    Run the ORIGINAL benchmark first (tuni_full_experiment.py); then use the
    results to pick which embedding × model pairs are worth this sweep —
    edit the ``pairs`` list below to save compute/time.

    Outputs (in ./metrics/tuni_emotion_feature_selection/):
      - tuni_emotion_feature_selection_feature_selection_metrics.csv
      - feature_selection_<embedding>_val_balanced_accuracy.png
      - feature_selection_overview_*.png
      - model.pkl / feature_selector.pkl per combination.

    Usage:
        python -m scripts run --feature-selection-experiment \
            experiments/tuni_emotion/tuni_feature_selection_experiment.py
'''
from scripts.config.feature_selection_config import (
    FeatureSelectionExperimentConfig,
    ModelEmbeddingPair,
)
from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import ExperimentFilters

from experiments.tuni_emotion.tuni_experiments_constants import (
    EMOTIONS,
    TRAIN_SINGERS,
    VAL_SINGERS,
)
from experiments.tuni_emotion.tuni_data_featureset_embedding_configs import (
    make_opensmile_embedding_config,
    make_whisper_embedding_config,
    make_clap_embedding_config,
)

# --- Singer-independent partition (same as the full experiment) ---

partition = PartitionConfig(
    name="tuni_emotion_singer_independent",
    train_singer_ids=TRAIN_SINGERS,
    test_singer_ids=VAL_SINGERS,
    singer_column="singer",
)

# --- Embeddings ---

opensmile = make_opensmile_embedding_config()   # ~6373 features
whisper_emb = make_whisper_embedding_config()   # 512 features
clap_emb = make_clap_embedding_config()         # 512 features

# ---------------------------------------------------------------------------
# EMBEDDING × MODEL PAIRS — edit this list based on previous benchmark runs
# ---------------------------------------------------------------------------
# Current top trad-ML pairs by Val Balanced Accuracy
# (experiments/tuni_emotion/metrics/tuni_emotion_full_metrics_outline.csv):
#   svm/opensmile 0.362 · svm/clap 0.350 · rf/opensmile 0.336
#   svm/whisper 0.331 · rf/whisper 0.331 · rf/clap 0.326
# Feature counts >= an embedding's full size are skipped automatically.
pairs = [
    ModelEmbeddingPair(dataset=opensmile, models=["svm", "rf"]),
    ModelEmbeddingPair(dataset=clap_emb, models=["svm"]),
    ModelEmbeddingPair(dataset=whisper_emb, models=["svm", "rf"]),
]

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

feature_selection_experiment = FeatureSelectionExperimentConfig(
    name="tuni_emotion_feature_selection",
    output_dir="./metrics",
    pairs=pairs,
    partition=partition,
    filters=ExperimentFilters(
        include_labels=EMOTIONS,
    ),
    frame_duration=0.5,  # TUNI embeddings use 0.5 s frames
    overlap=0.25,
    # Same hyper-parameters as the original trad_ml benchmark config
    model_params={
        "knn": {"n_neighbors": 5},
        "rf": {"n_estimators": 100, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500, "random_state": 42},
        "svm": {"kernel": "rbf", "C": 1.0},
    },
    feature_selection_methods=["rfe", "select_k_best"],
    n_features_list=[5, 10, 20, 40, 80, 160, 320, 640],
    seed=42,
)