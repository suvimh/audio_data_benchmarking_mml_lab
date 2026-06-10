from scripts.config.dataset_config import DatasetConfig

dataset = DatasetConfig(
    name="vocalset_audio",
    data_type="raw_audio",
    audio_dir="data/VocalSet/audio",
    sample_rate=16000,
    frame_duration=3.0,
    overlap=0.25,
    level_names=["singer_id", "technique", "exercise"],
    label_level="technique",
)
