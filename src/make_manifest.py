"""
make_manifest.py — Stage 1.

Walks data/positive and data/negative and writes data/manifest.csv with one
row per clip: (filename, label, source_id).

Why a manifest at all: every downstream step (split, training, evaluation)
reads this CSV instead of crawling directories itself. That makes the dataset
one explicit, version-controllable object — you can see exactly what the model
was trained on, and a bad file shows up as a bad row rather than a mystery.

source_id: because every clip is a distinct spoken take, each file is its own
source, so source_id == the filename stem. The column looks redundant today.
It is deliberate insurance: if you later expand the dataset by generating
several augmented copies from one clean take, those copies share a source_id,
and data.py can switch to a grouped split without any schema change here.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Folder name -> label string written into the manifest.
CLASS_DIRS = {
    "positive": "positive",
    "negative": "negative",
}
AUDIO_EXT = ".wav"


def build_manifest(data_root: Path) -> list[dict]:
    rows: list[dict] = []
    for folder, label in CLASS_DIRS.items():
        class_dir = data_root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}\n"
                f"Layout should be {data_root}/positive/*.wav and "
                f"{data_root}/negative/*.wav"
            )
        clips = sorted(class_dir.glob(f"*{AUDIO_EXT}"))
        if not clips:
            print(f"  warning: no {AUDIO_EXT} files in {class_dir}")
        for clip in clips:
            rows.append({
                # path relative to data_root, so the manifest is portable
                "filename": str(clip.relative_to(data_root)).replace("\\", "/"),
                "label": label,
                "source_id": clip.stem,   # each take is its own source
            })
    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate data/manifest.csv")
    parser.add_argument(
        "--data-root", default="data",
        help="Folder containing positive/ and negative/ subfolders (default: data)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path (default: <data-root>/manifest.csv)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out) if args.out else data_root / "manifest.csv"

    rows = build_manifest(data_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "source_id"])
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(r["label"] == "positive" for r in rows)
    n_neg = sum(r["label"] == "negative" for r in rows)
    print(f"Wrote {out_path} — {len(rows)} clips ({n_pos} positive, {n_neg} negative)")


if __name__ == "__main__":
    main()
