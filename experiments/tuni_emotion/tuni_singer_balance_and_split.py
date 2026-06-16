from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from experiments.tuni_emotion.tuni_experiments_constants import (
    RAW_AUDIO_DATA_DIR,
    SINGER_GENDER_MAP,
    TEST_SINGERS,
    TRAIN_SINGERS,
    VAL_SINGERS,
)


def _count_files_per_singer(audio_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for singer_dir in sorted(p for p in audio_root.iterdir() if p.is_dir()):
        counts[singer_dir.name] = sum(
            1 for wav in singer_dir.rglob("*") if wav.is_file() and wav.suffix.lower() == ".wav"
        )
    return counts


def _validate_disjoint_splits(train: List[str], val: List[str], test: List[str]) -> List[str]:
    issues: List[str] = []
    sets = {
        "train": set(train),
        "val": set(val),
        "test": set(test),
    }
    overlaps = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]
    for left, right in overlaps:
        overlap = sorted(sets[left].intersection(sets[right]))
        if overlap:
            issues.append(f"{left} and {right} overlap: {overlap}")
    return issues


def _validate_gender_coverage(split_name: str, singers: List[str]) -> str:
    genders = [SINGER_GENDER_MAP.get(singer, "unknown") for singer in singers]
    counts = Counter(genders)
    has_female = counts.get("female", 0) > 0
    has_male = counts.get("male", 0) > 0
    if has_female and has_male:
        return f"{split_name}: OK (female={counts['female']}, male={counts['male']})"
    return (
        f"{split_name}: FAIL (female={counts.get('female', 0)}, "
        f"male={counts.get('male', 0)}, unknown={counts.get('unknown', 0)})"
    )


def _build_summary(audio_root: Path) -> dict:
    per_singer_counts = _count_files_per_singer(audio_root)

    split_issues = _validate_disjoint_splits(TRAIN_SINGERS, VAL_SINGERS, TEST_SINGERS)
    val_gender_msg = _validate_gender_coverage("val", VAL_SINGERS)
    test_gender_msg = _validate_gender_coverage("test", TEST_SINGERS)

    return {
        "audio_root": str(audio_root),
        "per_singer_counts": per_singer_counts,
        "split_validation": {
            "disjoint_issues": split_issues,
            "val_gender_coverage": val_gender_msg,
            "test_gender_coverage": test_gender_msg,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Tuni singer balance and validate singer-independent splits."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path(RAW_AUDIO_DATA_DIR),
        help="Path to original Tuni wav directory (default from constants).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write JSON summary.",
    )
    args = parser.parse_args()

    audio_root = args.audio_root.expanduser().resolve()
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root does not exist: {audio_root}")

    summary = _build_summary(audio_root)

    print("=== Tuni Singer File Counts ===")
    for singer, count in sorted(summary["per_singer_counts"].items()):
        print(f"{singer}: {count}")

    print("\n=== Split Validation ===")
    split_validation = summary["split_validation"]
    if split_validation["disjoint_issues"]:
        for issue in split_validation["disjoint_issues"]:
            print(f"DISJOINT FAIL: {issue}")
    else:
        print("DISJOINT OK")
    print(split_validation["val_gender_coverage"])
    print(split_validation["test_gender_coverage"])

    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nWrote summary to: {out_path}")


if __name__ == "__main__":
    main()
