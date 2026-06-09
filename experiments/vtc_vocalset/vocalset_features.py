from scripts.config.dataset_config import DatasetConfig
from scripts.config.benchmark_config import BenchmarkConfig

VOCALSET_10 = ["vibrato", "straight", "belt", "breathy", "lip_trill",
               "inhaled", "spoken", "trill", "trillo", "vocal_fry"]
PITCH = ["vibrato", "straight", "trill", "trillo"]
TIMBRE = ["vocal_fry", "belt", "breathy", "inhaled", "spoken"]

TRAIN_SINGERS = [f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 7)]
VAL_SINGERS = ["f10", "f11", "m7", "m8", "m9"]

BASE_DATA_DIR = "path/to/vggish_or_mfcc_pkl_files"
GENDERS = ["female", "male"]

# ---------------------------------------------------------------------------
# Old-style .pkl files (Features + Class columns)
# ---------------------------------------------------------------------------

def make_vocalset_features_pkl_config(task_labels, task_name, gender="mixed"):
    return DatasetConfig(
        name=f"vocalset_{task_name}",
        data_type="features",
        feature_type="vggish",
        feature_column="Features",
        label_column="Class",
        train_data_paths={g: f"{BASE_DATA_DIR}/train_{g}.pkl" for g in GENDERS},
        val_data_paths={g: f"{BASE_DATA_DIR}/val_{g}.pkl" for g in GENDERS},
        gender_split=gender,
        include_labels=task_labels,
        pad_sequences=False,
    )

# ---------------------------------------------------------------------------
# New-style Parquet files (from the embedding extraction pipeline)
# ---------------------------------------------------------------------------

def make_vocalset_features_parquet_config(task_labels, task_name, gender="mixed",
                                          parquet_dir="output/parquet",
                                          parquet_file="vggish_window_3.0s.parquet"):
    return DatasetConfig(
        name=f"vocalset_{task_name}",
        data_type="features",
        feature_type="vggish",
        feature_column="embedding",
        label_column="vocal_technique",
        singer_column="singer",
        data_dir=parquet_dir,
        parquet_file=parquet_file,
        train_singer_ids=TRAIN_SINGERS,
        val_singer_ids=VAL_SINGERS,
        include_labels=task_labels,
        pad_sequences=False,
    )

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Old .pkl version
dataset_pkl = make_vocalset_features_pkl_config(VOCALSET_10, "vocalset_10", gender="mixed")

# New parquet version
dataset_parquet = make_vocalset_features_parquet_config(VOCALSET_10, "vocalset_10_parquet")

benchmark = BenchmarkConfig(
    name="vocalset_features_full",
    models=["knn", "rf", "mlp", "cnn_vggish"],
    model_params={
        "knn": {"n_neighbors": 5},
        "rf": {"n_estimators": 100, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500, "random_state": 42},
        "cnn_vggish": {"filters": [32, 64, 128], "epochs": 100, "patience": 10},
    },
    output_dir="./results/vocalset_features",
    seed=42,
)
