VOCALSET_10 = [
    "vibrato",
    "straight",
    "belt",
    "breathy",
    "lip_trill",
    "inhaled",
    "spoken",
    "trill",
    "trillo",
    "vocal_fry",
]
PMT = ["vibrato", "straight", "trill", "trillo"]
TMT = ["vocal_fry", "belt", "breathy", "inhaled", "spoken"]

TRAIN_SINGERS = [f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 7)]
VAL_SINGERS = ["f10", "f11", "m7", "m8", "m9"]

GENDERS = ["female", "male"]

RAW_AUDIO_DATA_DIR = "/Users/pubert/Downloads/SkyNote/OTHER_DATASETS/VOCALSET_data_versions/original_split/VocalSet_fixed/FULL"
EMBEDDINGS_DATA_DIR = "/home/suvihaara/Documents/PhD/DATA/VocalSet/VocalSet_fixed/embeddings/"
IMAGENET_DATA_DIR = "/home/suvihaara/Documents/PhD/DATA/VocalSet/VocalSet_fixed/FM_split_mel_spectrograms"