from scripts.config.dataset_config import DatasetConfig
from experiments.vtc_vocalset.vocalset_experiments_constants import RAW_AUDIO_DATA_DIR


def make_whisper_audio_config(audio_dir=RAW_AUDIO_DATA_DIR):
    return DatasetConfig(
        name="whisper_audio",
        data_type="raw_audio",
        audio_dir=audio_dir,
        sample_rate=16000,
        frame_duration=3.0,
        overlap=0.25,
        level_names=["singer_id", "technique", "exercise"],
        label_level="technique",
    )


def make_clap_audio_config(audio_dir=RAW_AUDIO_DATA_DIR):
    return DatasetConfig(
        name="clap_audio",
        data_type="raw_audio",
        audio_dir=audio_dir,
        sample_rate=44100,
        frame_duration=3.0,
        overlap=0.25,
        level_names=["singer_id", "technique", "exercise"],
        label_level="technique",
    )
