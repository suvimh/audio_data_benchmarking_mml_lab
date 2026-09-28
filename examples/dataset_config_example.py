from scripts.config.dataset_config import DatasetConfig

# --- Old-style .pkl files (Features + Class columns) ---

features_dataset_pkl = DatasetConfig(
    name="vocalset_vggish",
    data_type="features",
    feature_type="vggish",
    train_data_paths={"female": "data/vggish/train_female.pkl", "male": "data/vggish/train_male.pkl"},
    val_data_paths={"female": "data/vggish/val_female.pkl", "male": "data/vggish/val_male.pkl"},
    gender_split="mixed",
    include_labels=["vibrato", "straight", "belt", "breathy", "lip_trill",
                    "inhaled", "spoken", "trill", "trillo", "vocal_fry"],
)

# --- New-style Parquet files (from embedding extraction pipeline) ---
# The parquet file has an "embedding" column (bytes) and label columns
# from the folder hierarchy (e.g. "vocal_technique", "singer", "exercise_type").

features_dataset_parquet = DatasetConfig(
    name="vocalset_vggish",
    data_type="features",
    feature_type="vggish",
    data_dir="output/parquet",
    parquet_file="vggish_window_3.0s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
    singer_column="singer",
    train_singer_ids=[f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 7)],
    val_singer_ids=["f10", "f11", "m7", "m8", "m9"],
    include_labels=["vibrato", "straight", "belt", "breathy", "lip_trill",
                    "inhaled", "spoken", "trill", "trillo", "vocal_fry"],
    pad_sequences=False,
)

# The label column is auto-detected when using parquet files:
# any column that is not embedding metadata (embedding, embedding_dtype,
# embedding_shape, filename, filepath, frame_index, extractor) is treated
# as a candidate label column.  You can also set label_level to pick one.

audio_dataset = DatasetConfig(
    name="vocalset_audio",
    data_type="raw_audio",
    audio_dir="data/VocalSet/audio",
    sample_rate=16000,
    frame_duration=3.0,
    overlap=0.25,
    level_names=["singer_id", "technique", "exercise"],
    label_level="technique",
    gender_split="mixed",
    train_singer_ids=[f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 7)],
    val_singer_ids=["f10", "f11", "m7", "m8", "m9"],
    include_labels=["vibrato", "straight", "belt", "breathy", "lip_trill",
                    "inhaled", "spoken", "trill", "trillo", "vocal_fry"],
)

image_dataset = DatasetConfig(
    name="vocalset_melspectrograms",
    data_type="images",
    image_dir="data/melspectrograms",
    image_size=(224, 224),
    gender_split="mixed",
)
