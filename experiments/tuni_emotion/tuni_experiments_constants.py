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

TRAIN_SINGERS = [
    "Anu_Mattila",
    "Anniina_Honkala",
    "Bianca_Hösli",
    "Elina_Lahtinen",
    "Liisi_Petterson",
    "Maarit_Aura",
    "Marja_Erdogan",
    "Sanna_Vähälä",
    "Tua_Hakanpää",
]
VAL_SINGERS = ["Saga_Ohlsson_1", "Tero_Ikävalko"]
TEST_SINGERS = ["Veera_Tapanainen", "Tommi_Grönberg"]

RAW_AUDIO_DATA_DIR = "/Users/pubert/Downloads/SkyNote/OTHER_DATASETS/TUNI_emotion_dataset"
TUNI_EMBEDDINGS_DATA_DIR = str(REPO_ROOT / "embeddings" / "tuni_emotion")
TUNI_OPENSMILE_PARQUET_PREFIX = "tuni_emotion_dataset"
