from scripts.config.dataset_config import DatasetConfig
from scripts.config.benchmark_config import BenchmarkConfig

VOCALSET_10 = ["vibrato", "straight", "belt", "breathy", "lip_trill",
               "inhaled", "spoken", "trill", "trillo", "vocal_fry"]
PITCH = ["vibrato", "straight", "trill", "trillo"]
TIMBRE = ["vocal_fry", "belt", "breathy", "inhaled", "spoken"]

TRAIN_SINGERS = [f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 7)]
VAL_SINGERS = ["f10", "f11", "m7", "m8", "m9"]

def make_vocalset_audio_config(task_labels, task_name, gender="mixed"):
    singer_ids = TRAIN_SINGERS + VAL_SINGERS
    if gender == "female":
        singer_ids = [s for s in singer_ids if s.startswith("f")]
    elif gender == "male":
        singer_ids = [s for s in singer_ids if s.startswith("m")]

    train_ids = [s for s in singer_ids if s in TRAIN_SINGERS]
    val_ids = [s for s in singer_ids if s in VAL_SINGERS]

    return DatasetConfig(
        name=f"vocalset_audio_{task_name}",
        data_type="raw_audio",
        audio_dir="path/to/VocalSet/audio",
        sample_rate=16000,
        frame_duration=3.0,
        overlap=0.25,
        level_names=["singer_id", "technique", "exercise"],
        label_level="technique",
        train_singer_ids=train_ids if gender != "mixed" else TRAIN_SINGERS,
        val_singer_ids=val_ids if gender != "mixed" else VAL_SINGERS,
        include_labels=task_labels,
    )

dataset = make_vocalset_audio_config(VOCALSET_10, "vocalset_10", gender="mixed")

benchmark = BenchmarkConfig(
    name="vocalset_audio_full",
    models=["whisper", "clap"],
    model_params={
        "whisper": {
            "model_name": "openai/whisper-base",
            "learning_rate": 5e-5,
            "batch_size": 16,
            "epochs": 50,
            "patience": 10,
        },
        "clap": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,
            "hidden_dim": 512,
        },
    },
    output_dir="./results/vocalset_audio",
    seed=42,
)
