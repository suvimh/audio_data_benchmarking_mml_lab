from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TUNI_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "gentleness",
    "neutral",
]

TUNI_SINGERS = [
    "Anu_Mattila",
    "Anniina_Honkala",
    "Bianca_Hösli",
    "Elina_Lahtinen",
    "Liisi_Petterson",
    "Maarit_Aura",
    "Marja_Erdogan",
    "Saga_Ohlsson_1",
    "Sanna_Vähälä",
    "Tero_Ikävalko",
    "Tommi_Grönberg",
    "Tua_Hakanpää",
    "Veera_Tapanainen",
]

MALE_SINGERS = ["Tero_Ikävalko", "Tommi_Grönberg"]
FEMALE_SINGERS = [s for s in TUNI_SINGERS if s not in MALE_SINGERS]
SINGER_GENDER_MAP = {s: "female" for s in FEMALE_SINGERS} | {
    s: "male" for s in MALE_SINGERS
}

POP_ONLY_SINGERS = [
    "Saga_Ohlsson_1",
    "Tero_Ikävalko",
    "Tua_Hakanpää",
    "Veera_Tapanainen",
]
CLASSICAL_ONLY_SINGERS = [
    "Anniina_Honkala",
    "Tommi_Grönberg",
]

# Singer-independent train/val split (9 train / 4 val):
# - 1 male in train (Tero, pop), 1 male in val (Tommi, classical)
# - val: pop-only females (Saga, Veera), classical-only female (Anniina), classical male (Tommi)
# - train includes dual-genre singers since the dataset is pop-heavy
TRAIN_SINGERS = [
    "Anu_Mattila",
    "Bianca_Hösli",
    "Elina_Lahtinen",
    "Liisi_Petterson",
    "Maarit_Aura",
    "Marja_Erdogan",
    "Sanna_Vähälä",
    "Tua_Hakanpää",
    "Tero_Ikävalko",
]
VAL_SINGERS = [
    "Saga_Ohlsson_1",
    "Tommi_Grönberg",
    "Anniina_Honkala",
    "Veera_Tapanainen",
]

RAW_AUDIO_DATA_DIR = "/Users/pubert/Downloads/SkyNote/OTHER_DATASETS/TUNI_emotion_dataset"
TUNI_EMBEDDINGS_DATA_DIR = str(REPO_ROOT / "embeddings" / "tuni_emotion")
TUNI_OPENSMILE_DIR = str(REPO_ROOT / "embeddings" / "tuni_emotion" / "opensmile")
TUNI_WHISPER_DIR = str(REPO_ROOT / "embeddings" / "tuni_emotion" / "whisper")
TUNI_CLAP_DIR = str(REPO_ROOT / "embeddings" / "tuni_emotion" / "clap")
TUNI_OPENSMILE_PARQUET_PREFIX = "tuni_emotion_dataset"
