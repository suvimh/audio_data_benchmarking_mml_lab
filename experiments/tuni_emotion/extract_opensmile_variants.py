"""
Extract additional OpenSMILE feature sets for TUNI emotion (phase 3).

Requires the external extraction repo:
  /Users/pubert/Downloads/SkyNote/audio_repo/audio_embeddings_and_feature_extraction_from_audio_dataset

Usage:
    python -m experiments.tuni_emotion.extract_opensmile_variants --variants emobase is09 egemaps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.tuni_emotion.tuni_experiments_constants import (
    TUNI_OPENSMILE_DIR,
    TUNI_OPENSMILE_PARQUET_PREFIX,
)
from experiments.tuni_emotion.tuni_data_featureset_embedding_configs import OPENSMILE_VARIANTS

EXTRACTION_REPO = Path(
    "/Users/pubert/Downloads/SkyNote/audio_repo/"
    "audio_embeddings_and_feature_extraction_from_audio_dataset"
)
OPENSMILE_OUTPUT = EXTRACTION_REPO / "TUNI_emotion_data_extraction" / "embeddings" / "OpenSmile"
LOCAL_OPENSMILE_DIR = Path(TUNI_OPENSMILE_DIR)


def _duration_suffix(frame_duration: float) -> str:
    return f"{frame_duration:.1f}s"


def _expected_parquet(variant: str, frame_duration: float) -> Path:
    feature_type = OPENSMILE_VARIANTS[variant]
    return OPENSMILE_OUTPUT / (
        f"{TUNI_OPENSMILE_PARQUET_PREFIX}_{feature_type}_{_duration_suffix(frame_duration)}.parquet"
    )


def _link_parquet(variant: str, frame_duration: float) -> Path:
    source = _expected_parquet(variant, frame_duration)
    if not source.exists():
        raise FileNotFoundError(f"Missing extracted parquet: {source}")

    LOCAL_OPENSMILE_DIR.mkdir(parents=True, exist_ok=True)
    target = LOCAL_OPENSMILE_DIR / source.name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
    return target


def extract_variants(
    variants: list[str],
    frame_durations: list[float] | None = None,
) -> dict:
    frame_durations = frame_durations or [3.0]
    results = {"extracted": [], "linked": [], "missing": []}

    extract_script = EXTRACTION_REPO / "scripts" / "extract.py"
    if not extract_script.exists():
        results["note"] = (
            "Extraction script not found. Run TUNI_emotion_data_extraction notebook "
            "with opensmile variants configured, then re-run this linker."
        )

    for variant in variants:
        if variant not in OPENSMILE_VARIANTS:
            raise ValueError(f"Unknown variant {variant}. Choose from {OPENSMILE_VARIANTS}")
        for duration in frame_durations:
            parquet = _expected_parquet(variant, duration)
            if not parquet.exists() and extract_script.exists():
                cmd = [
                    sys.executable,
                    str(extract_script),
                    "--dataset",
                    "tuni_emotion",
                    "--extractor",
                    "opensmile",
                    "--opensmile-variant",
                    variant,
                    "--window-length",
                    str(duration),
                ]
                subprocess.run(cmd, check=False, cwd=str(EXTRACTION_REPO))
            if parquet.exists():
                linked = _link_parquet(variant, duration)
                results["linked"].append(str(linked))
            else:
                results["missing"].append(str(parquet))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract/link OpenSMILE variants for TUNI")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["emobase", "is09", "egemaps"],
        choices=list(OPENSMILE_VARIANTS.keys()),
    )
    parser.add_argument("--frame-durations", nargs="+", type=float, default=[3.0])
    args = parser.parse_args()

    results = extract_variants(args.variants, args.frame_durations)
    print(results)
    if results.get("missing"):
        print(
            "\nSome parquets are still missing. Configure the extraction repo to emit "
            "opensmile-emobase, opensmile-is09, and opensmile-egemapsv02, then rerun."
        )


if __name__ == "__main__":
    main()
