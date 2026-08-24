"""
export_model.py — Stage 6: produce the deployable model artifact.

Every training script so far retrained from scratch each run. A serving
container can't do that — it must LOAD a pre-trained model and must not carry
the training data. This script trains the final model once and saves it as a
self-contained artifact (models/model.joblib) plus a small metadata file.

Which model? The HARD-NEGATIVE-MINED logistic regression. The plain baseline
looked good on the test set but false-triggered ~300x/hour on real audio; mining
cut that ~96%. So the mined model is the only defensible thing to deploy.

Difference from hard_negative_mining.py: that script held out Recording 3 to
*prove* the fix honestly. Here we're not proving anything — we've already done
that — so we mine from the ENTIRE cold stream to give the shipped model as many
real-world hard negatives as possible. (If no cold stream is present, it trains
without mining and warns loudly that FA/hr will be poor.)

The saved artifact is a fitted sklearn Pipeline (scaler + logreg). The container
loads it and feeds it MFCC vectors from the SAME features.py used here — so no
train/serve skew.

Run it:
    python src/export_model.py
    # -> models/model.joblib, models/model_meta.json
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    N_MFCC, POOLING, MFCC_FEATURE_LEN, TARGET_SR, CLIP_SECONDS,
)
from data import load_manifest, make_splits  # noqa: E402
from train_baseline import build_matrix, make_model  # noqa: E402
from fa_per_hour import load_stream, window_probabilities  # noqa: E402
from hard_negative_mining import mine_hard_negatives, make_model_balanced  # noqa: E402

DEFAULT_THRESHOLD = 0.5   # post-mining, FA/hr is already low at 0.5 (see journey doc)
MINING_THRESHOLD = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the deployable model artifact")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--stream-dir", default="data/cold_stream")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="operating threshold stored in metadata (default 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- base training data (train+val; test is never used for the shipped model)
    df = load_manifest(args.manifest)
    splits = make_splits(df, seed=args.seed)
    X_tr, y_tr = build_matrix(splits.train, data_root)
    X_va, y_va = build_matrix(splits.val, data_root)
    X_fit = np.concatenate([X_tr, X_va])
    y_fit = np.concatenate([y_tr, y_va])
    print(f"Base training clips (train+val): {len(X_fit)}")

    # --- mine hard negatives from the whole cold stream (if present) -------
    stream_dir = Path(args.stream_dir)
    mined = []
    if stream_dir.is_dir():
        audio = sorted(p for p in stream_dir.iterdir()
                       if p.suffix.lower() in {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac"})
        if audio:
            # need a base model to find its own false positives
            base = make_model("logreg", args.seed)
            base.fit(X_fit, y_fit)
            print(f"Mining hard negatives from {len(audio)} cold-stream file(s)...")
            from features import mfcc_features
            for p in audio:
                w = mine_hard_negatives(p, base, MINING_THRESHOLD)
                print(f"  {p.name}: {len(w)} hard negatives")
                mined.extend(w)
            if mined:
                X_mined = np.stack([mfcc_features(w) for w in mined])
                X_fit = np.concatenate([X_fit, X_mined])
                y_fit = np.concatenate([y_fit, np.zeros(len(X_mined), dtype=int)])
    if not mined:
        print("WARNING: no hard negatives mined (no cold stream). The shipped model "
              "will have POOR false-accepts-per-hour. Add data/cold_stream/ and re-run.")

    n_pos = int((y_fit == 1).sum()); n_neg = int((y_fit == 0).sum())
    print(f"\nFinal training set: {len(X_fit)} clips ({n_pos} pos / {n_neg} neg)")

    # --- fit the shipped model (balanced to absorb the extra negatives) ----
    model = make_model_balanced(args.seed) if mined else make_model("logreg", args.seed)
    model.fit(X_fit, y_fit)

    # --- save artifact + metadata -----------------------------------------
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    meta = {
        "model": "logistic_regression",
        "trained_on": "train+val" + ("+hard_negatives" if mined else ""),
        "hard_negatives_added": len(mined),
        "class_weight": "balanced" if mined else None,
        "operating_threshold": args.threshold,
        "features": {
            "type": "mfcc",
            "n_mfcc": N_MFCC,
            "pooling": POOLING,
            "vector_length": MFCC_FEATURE_LEN,
            "sample_rate": TARGET_SR,
            "clip_seconds": CLIP_SECONDS,
        },
        "labels": {"0": "negative", "1": "positive (like a Bosch)"},
    }
    meta_path = out_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  {meta_path}")
    print(f"Operating threshold stored: {args.threshold}")


if __name__ == "__main__":
    main()
