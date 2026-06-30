from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List

from experiments.tuni_emotion.tuni_experiments_constants import (
    CLASSICAL_ONLY_SINGERS,
    POP_ONLY_SINGERS,
    RAW_AUDIO_DATA_DIR,
    SINGER_GENDER_MAP,
    TUNI_SINGERS,
    TRAIN_SINGERS,
    VAL_SINGERS,
)


def _normalize_name(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def _normalize_counts(per_singer_counts: Dict[str, int]) -> Dict[str, int]:
    return {_normalize_name(name): count for name, count in per_singer_counts.items()}


def _lookup_count(per_singer_counts: Dict[str, int], singer: str) -> int:
    return per_singer_counts.get(_normalize_name(singer), 0)


def _resolve_singer_root(audio_root: Path) -> Path:
    normalized_singers = {_normalize_name(s) for s in TUNI_SINGERS}
    direct_children = [_normalize_name(p.name) for p in audio_root.iterdir() if p.is_dir()]
    if any(name in normalized_singers for name in direct_children):
        return audio_root

    wav_root = audio_root / "wav"
    if wav_root.exists() and wav_root.is_dir():
        wav_children = [_normalize_name(p.name) for p in wav_root.iterdir() if p.is_dir()]
        if any(name in normalized_singers for name in wav_children):
            return wav_root

    return audio_root


def _count_files_per_singer(singer_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for singer_dir in sorted(p for p in singer_root.iterdir() if p.is_dir()):
        counts[singer_dir.name] = sum(
            1 for wav in singer_dir.rglob("*") if wav.is_file() and wav.suffix.lower() == ".wav"
        )
    return counts


def _validate_disjoint_splits(train: List[str], val: List[str]) -> List[str]:
    issues: List[str] = []
    overlap = sorted(set(train).intersection(val))
    if overlap:
        issues.append(f"train and val overlap: {overlap}")
    return issues


def _validate_complete_coverage(train: List[str], val: List[str]) -> List[str]:
    issues: List[str] = []
    assigned = set(train) | set(val)
    missing = sorted(set(TUNI_SINGERS) - assigned)
    extra = sorted(assigned - set(TUNI_SINGERS))
    if missing:
        issues.append(f"singers not assigned to train or val: {missing}")
    if extra:
        issues.append(f"unknown singers in split: {extra}")
    if len(train) + len(val) != len(TUNI_SINGERS):
        issues.append(
            f"expected {len(TUNI_SINGERS)} singers total, got {len(train)} train + {len(val)} val"
        )
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


def _split_file_totals(
    per_singer_counts: Dict[str, int],
    singers: List[str],
) -> dict:
    normalized_counts = _normalize_counts(per_singer_counts)
    total = sum(_lookup_count(normalized_counts, s) for s in singers)
    return {
        "singers": len(singers),
        "files": total,
        "per_singer": {s: _lookup_count(normalized_counts, s) for s in singers},
    }


def _build_summary(audio_root: Path) -> dict:
    singer_root = _resolve_singer_root(audio_root)
    per_singer_counts = _normalize_counts(_count_files_per_singer(singer_root))

    disjoint_issues = _validate_disjoint_splits(TRAIN_SINGERS, VAL_SINGERS)
    coverage_issues = _validate_complete_coverage(TRAIN_SINGERS, VAL_SINGERS)

    train_totals = _split_file_totals(per_singer_counts, TRAIN_SINGERS)
    val_totals = _split_file_totals(per_singer_counts, VAL_SINGERS)
    all_files = sum(per_singer_counts.values())
    used_files = train_totals["files"] + val_totals["files"]

    val_has_pop_only_female = any(
        s in POP_ONLY_SINGERS and SINGER_GENDER_MAP.get(s) == "female"
        for s in VAL_SINGERS
    )
    val_has_classical_only_female = any(
        s in CLASSICAL_ONLY_SINGERS and SINGER_GENDER_MAP.get(s) == "female"
        for s in VAL_SINGERS
    )

    return {
        "audio_root": str(audio_root),
        "singer_root": str(singer_root),
        "per_singer_counts": per_singer_counts,
        "splits": {
            "train": train_totals,
            "val": val_totals,
            "train_file_pct": round(100 * train_totals["files"] / all_files, 1) if all_files else 0,
            "val_file_pct": round(100 * val_totals["files"] / all_files, 1) if all_files else 0,
        },
        "split_validation": {
            "disjoint_issues": disjoint_issues,
            "coverage_issues": coverage_issues,
            "train_gender_coverage": _validate_gender_coverage("train", TRAIN_SINGERS),
            "val_gender_coverage": _validate_gender_coverage("val", VAL_SINGERS),
            "val_has_pop_only_female": val_has_pop_only_female,
            "val_has_classical_only_female": val_has_classical_only_female,
            "all_files_accounted_for": used_files == all_files,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Tuni singer balance and validate singer-independent train/val splits."
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

    splits = summary["splits"]
    print("\n=== Train / Val Split ===")
    print(f"Train: {splits['train']['singers']} singers, {splits['train']['files']} files ({splits['train_file_pct']}%)")
    print(f"Val:   {splits['val']['singers']} singers, {splits['val']['files']} files ({splits['val_file_pct']}%)")

    print("\n=== Split Validation ===")
    split_validation = summary["split_validation"]
    all_issues = split_validation["disjoint_issues"] + split_validation["coverage_issues"]
    if all_issues:
        for issue in all_issues:
            print(f"FAIL: {issue}")
    else:
        print("DISJOINT OK")
        print("COVERAGE OK (all singers assigned to train or val)")
    print(split_validation["train_gender_coverage"])
    print(split_validation["val_gender_coverage"])
    pop_only_female = "OK" if split_validation["val_has_pop_only_female"] else "FAIL"
    print(f"val pop-only female: {pop_only_female}")
    classical_only_female = "OK" if split_validation["val_has_classical_only_female"] else "FAIL"
    print(f"val classical-only female: {classical_only_female}")
    files_accounted = "OK" if split_validation["all_files_accounted_for"] else "FAIL"
    print(f"all files accounted for: {files_accounted}")

    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nWrote summary to: {out_path}")


if __name__ == "__main__":
    main()
