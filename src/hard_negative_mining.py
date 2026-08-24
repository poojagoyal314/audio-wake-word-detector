"""
hard_negative_mining.py — Stage 5, part 3: close the loop.

The FA/hr result proved the model fires on real-world sounds it never saw as
negatives during training. Hard negative mining fixes exactly that: the windows
that FALSELY fire are the negatives the model most needs to learn from. We
capture them, add them to the training negatives, retrain, and re-measure.

Leakage guard (the crux)
------------------------
If we mined hard negatives from the whole cold stream and then measured the
improved FA/hr on that same stream, FA/hr would drop just because the model
memorized those exact windows — a fake improvement. So we SPLIT the stream:

    mine  from the "mine" files   (Recording_1, Recording_2)
    report FA/hr on the held-out  (Recording_3)  the model never trained on.

We report before/after FA/hr on the held-out stream only. That number is honest.

Trade-off to watch
------------------
Teaching the model that certain sounds are negative makes it more conservative,
which can raise the miss rate (FRR) on real positives. So we re-check the test
set too: FA/hr should fall AND test recall should not collapse. Measuring both
is the whole discipline.

Run it (after clips + cold stream are in place):
    python src/hard_negative_mining.py
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import TARGET_SR, CLIP_SAMPLES  # noqa: E402
from data import load_manifest, make_splits  # noqa: E402
from train_baseline import build_matrix, make_model, extract_mfcc_features  # noqa: E402
from fa_per_hour import (  # noqa: E402
    load_stream, window_probabilities, count_false_accepts,
    mfcc_vector_from_signal, HOP_SAMPLES, HOP_SECONDS, DEBOUNCE_SECONDS,
)

# Which cold-stream files to MINE from vs HOLD OUT for honest FA/hr reporting.
MINE_FILES = {"Recording_1", "Recording_2"}
HELDOUT_FILES = {"Recording_3"}
MINING_THRESHOLD = 0.5   # mine everything that fires at the loosest useful cut


def make_model_balanced(seed):
    """Logreg with class_weight='balanced' to absorb the added negatives without
    faking positives — up-weights the rarer (positive) class in the loss."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def mine_hard_negatives(stream_path: Path, model, threshold: float):
    """Return the raw audio windows (as arrays) that falsely fire, debounced so
    we save one clip per event, not one per overlapping window."""
    y = load_stream(stream_path)
    if len(y) < CLIP_SAMPLES:
        return []
    starts = list(range(0, len(y) - CLIP_SAMPLES + 1, HOP_SAMPLES))
    probs = window_probabilities(y, model)
    debounce_windows = int(round(DEBOUNCE_SECONDS / HOP_SECONDS))
    mined, cooldown = [], 0
    for i, p in enumerate(probs):
        if cooldown > 0:
            cooldown -= 1
            continue
        if p >= threshold:
            s = starts[i]
            mined.append(y[s:s + CLIP_SAMPLES].copy())  # the offending window
            cooldown = debounce_windows
    return mined


def fa_per_hour_on(files, model, thresholds):
    """FA/hr at each threshold over a set of stream files (held-out)."""
    all_probs, total_sec = [], 0.0
    for f in files:
        y = load_stream(f)
        total_sec += len(y) / TARGET_SR
        all_probs.append(window_probabilities(y, model))
    probs = np.concatenate(all_probs)
    hours = total_sec / 3600.0
    return {t: count_false_accepts(probs, t) / hours for t in thresholds}, hours


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard negative mining round")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--stream-dir", default="data/cold_stream")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", default="0.5,0.639,0.79")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    stream_dir = Path(args.stream_dir)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    def find(names):
        out = []
        for p in sorted(stream_dir.iterdir()):
            if p.suffix.lower() in {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac"} \
               and p.stem in names:
                out.append(p)
        return out

    mine_paths = find(MINE_FILES)
    heldout_paths = find(HELDOUT_FILES)
    if not mine_paths or not heldout_paths:
        raise FileNotFoundError(
            f"Need mine files {MINE_FILES} and held-out {HELDOUT_FILES} in {stream_dir}. "
            f"Found mine={[p.name for p in mine_paths]}, held-out={[p.name for p in heldout_paths]}"
        )

    # --- data + baseline model (train+val) --------------------------------
    df = load_manifest(args.manifest)
    splits = make_splits(df, seed=args.seed)
    X_tr, y_tr = build_matrix(splits.train, data_root)
    X_va, y_va = build_matrix(splits.val, data_root)
    X_te, y_te = build_matrix(splits.test, data_root)
    X_fit = np.concatenate([X_tr, X_va]); y_fit = np.concatenate([y_tr, y_va])

    base = make_model("logreg", args.seed)
    base.fit(X_fit, y_fit)

    # --- BEFORE: FA/hr on held-out stream + test metrics ------------------
    before_fa, held_hours = fa_per_hour_on(heldout_paths, base, thresholds)
    before_recall = recall_score(y_te, (base.predict_proba(X_te)[:, 1] >= 0.5).astype(int))
    before_auc = roc_auc_score(y_te, base.predict_proba(X_te)[:, 1])

    # --- MINE hard negatives from the mine files --------------------------
    print(f"Mining hard negatives from {[p.name for p in mine_paths]} "
          f"at threshold {MINING_THRESHOLD}...")
    mined_windows = []
    for p in mine_paths:
        w = mine_hard_negatives(p, base, MINING_THRESHOLD)
        print(f"  {p.name}: {len(w)} hard negatives")
        mined_windows.extend(w)
    print(f"Total mined: {len(mined_windows)} hard-negative clips\n")

    if not mined_windows:
        print("No hard negatives mined — nothing to add. Stopping.")
        return

    # featurize mined windows and append as NEGATIVES (label 0)
    X_mined = np.stack([mfcc_vector_from_signal(w) for w in mined_windows])
    y_mined = np.zeros(len(X_mined), dtype=int)

    X_fit2 = np.concatenate([X_fit, X_mined])
    y_fit2 = np.concatenate([y_fit, y_mined])
    n_pos = int((y_fit2 == 1).sum()); n_neg = int((y_fit2 == 0).sum())
    print(f"Retraining on {len(X_fit2)} clips  ({n_pos} pos / {n_neg} neg) "
          f"with class_weight='balanced'")

    # --- retrain (balanced) + re-measure ----------------------------------
    improved = make_model_balanced(args.seed)
    improved.fit(X_fit2, y_fit2)

    after_fa, _ = fa_per_hour_on(heldout_paths, improved, thresholds)
    after_recall = recall_score(y_te, (improved.predict_proba(X_te)[:, 1] >= 0.5).astype(int))
    after_auc = roc_auc_score(y_te, improved.predict_proba(X_te)[:, 1])

    # --- report ------------------------------------------------------------
    print(f"\n=== FA/hr on HELD-OUT stream ({', '.join(p.name for p in heldout_paths)}, "
          f"{held_hours*60:.1f} min) ===")
    print(f"  {'threshold':>9}   {'before':>8}   {'after':>8}   {'change':>8}")
    for t in thresholds:
        b, a = before_fa[t], after_fa[t]
        print(f"  {t:>9.3f}   {b:>8.1f}   {a:>8.1f}   {a-b:>+8.1f}")

    print(f"\n=== side-effect check on TEST set (must not collapse) ===")
    print(f"  recall  before {before_recall:.3f}   after {after_recall:.3f}   "
          f"change {after_recall-before_recall:+.3f}")
    print(f"  auc     before {before_auc:.3f}   after {after_auc:.3f}   "
          f"change {after_auc-before_auc:+.3f}")

    print("\nRead it: FA/hr should DROP on the held-out stream (the model learned")
    print("its real-world blind spots) while test recall stays healthy. If recall")
    print("fell a lot, the model got too conservative — that's the trade to weigh.")


if __name__ == "__main__":
    main()
