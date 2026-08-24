"""
data.py — Stage 2.

Loads data/manifest.csv and produces reproducible train / val / test splits.

Split choice: stratified random. We ruled out leakage — every positive is a
distinct spoken take, so there are no correlated near-duplicates that must be
kept together. Stratification just keeps the positive/negative ratio steady
across all three splits, which matters here because the classes are close but
not identical in size (338 vs 365).

The split is grouped-ready but not grouped today. Because source_id is unique
per file, grouping would be a no-op. If you later add augmented copies that
share a source_id, switch `make_splits` to StratifiedGroupKFold (noted inline)
so augmented children never straddle the train/test boundary.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

# positive is the phrase we want to detect -> the "1" class.
LABEL_TO_INT = {"negative": 0, "positive": 1}

RANDOM_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        def line(name: str, df: pd.DataFrame) -> str:
            pos = int((df["y"] == 1).sum())
            neg = int((df["y"] == 0).sum())
            return f"  {name:<5} {len(df):>4} clips  ({pos} pos / {neg} neg)"
        return "\n".join([
            line("train", self.train),
            line("val", self.val),
            line("test", self.test),
        ])


def load_manifest(manifest_path: str | Path = "data/manifest.csv") -> pd.DataFrame:
    """Read the manifest and attach an integer label column `y`."""
    df = pd.read_csv(manifest_path)
    missing = {"filename", "label", "source_id"} - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    unknown = set(df["label"]) - set(LABEL_TO_INT)
    if unknown:
        raise ValueError(f"manifest has unknown labels: {sorted(unknown)}")
    df = df.copy()
    df["y"] = df["label"].map(LABEL_TO_INT).astype(int)
    return df


def make_splits(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> Splits:
    """Stratified train/val/test split with a fixed seed for reproducibility.

    Done in two steps: first peel off the test set, then peel val out of the
    remainder. Stratifying on `y` each time keeps the class ratio stable.

    (If you move to grouped splitting later: replace both train_test_split
    calls with sklearn.model_selection.StratifiedGroupKFold using
    groups=df["source_id"], and take one fold as test, one as val.)
    """
    train_val, test = train_test_split(
        df,
        test_size=test_fraction,
        stratify=df["y"],
        random_state=seed,
        shuffle=True,
    )
    # val_fraction is expressed relative to the whole set, so rescale it to the
    # remaining train_val portion.
    val_relative = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val["y"],
        random_state=seed,
        shuffle=True,
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


if __name__ == "__main__":
    frame = load_manifest()
    splits = make_splits(frame)
    print(f"Loaded {len(frame)} clips from manifest.")
    print(splits.summary())
