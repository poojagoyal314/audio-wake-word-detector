"""
train_baseline.py — Stage 3.

The number to beat. Before reaching for a CNN, we establish how far a simple
classical model gets: MFCC features pooled over time into a fixed-length
vector, fed to logistic regression (or a small random forest).

Design notes
------------
* Same audio front-end as the CNN. We call features.load_audio, so the baseline
  hears exactly the same 16 kHz / 3 s / mono signal the CNN will. Only the
  featurization differs (MFCC vector here vs log-Mel image for the CNN). That
  keeps the eventual comparison honest.

* Stateful normalization is fine HERE. StandardScaler standardizes against
  training-set statistics — the opposite of the stateless scheme in features.py.
  That is deliberate: this baseline is an offline comparison model, never a live
  server, so the "hidden state to sync at serve time" cost doesn't apply. The
  scaler lives inside a Pipeline so it is fit on train only (no leakage).

* Evaluated on VALIDATION, not test. Model selection (baseline vs CNN) is a
  decision, and decisions are made on val. The test set stays untouched until
  the final evaluation in Stage 5.

* Everything is logged to MLflow (local ./mlruns). Feature settings, classifier
  hyperparameters, val metrics, and a confusion-matrix image all land in one
  run, so `mlflow ui` gives you a comparable record of every model you try.

Run it:
    python src/train_baseline.py --model logreg
    python src/train_baseline.py --model rf
Then compare the two runs in the MLflow UI:
    mlflow ui        # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")  # headless: render plots to file, no display needed
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# make src/ importable whether run from repo root or elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_audio, TARGET_SR, CLIP_SECONDS  # noqa: E402
from data import load_manifest, make_splits  # noqa: E402
from tracking import setup_mlflow, EXPERIMENT_NAME  # noqa: E402

# --- baseline feature settings (logged to MLflow so runs are comparable) ----
N_MFCC = 40          # number of MFCC coefficients
POOLING = "mean+std"  # temporal pooling: one vector per clip regardless of length


def extract_mfcc_features(path: str) -> np.ndarray:
    """One clip -> fixed-length MFCC feature vector (length 2 * N_MFCC).

    MFCCs give an (N_MFCC, n_frames) matrix — still time-varying. A classical
    model needs a fixed-length vector, so we pool across time by taking the mean
    and standard deviation of each coefficient. Mean captures the average
    spectral shape; std captures how much it moves over the clip.
    """
    y = load_audio(path)  # 16 kHz, mono, fixed to CLIP_SAMPLES — same as the CNN
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def build_matrix(df, data_root: Path):
    """Turn a split DataFrame into (X, y) arrays."""
    X = np.stack([extract_mfcc_features(str(data_root / fn)) for fn in df["filename"]])
    y = df["y"].to_numpy()
    return X, y


def make_model(kind: str, seed: int) -> Pipeline:
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"unknown model '{kind}' (use 'logreg' or 'rf')")
    # Scaler is meaningful for logreg and a harmless no-op for the tree model;
    # keeping the pipeline uniform means one code path for both.
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def plot_confusion(cm, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["negative", "positive"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MFCC classical baseline")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # --- data -------------------------------------------------------------
    df = load_manifest(args.manifest)
    splits = make_splits(df, seed=args.seed)
    print("Split sizes:")
    print(splits.summary())

    print("\nExtracting MFCC features...")
    X_train, y_train = build_matrix(splits.train, data_root)
    X_val, y_val = build_matrix(splits.val, data_root)
    print(f"  feature vector length: {X_train.shape[1]} "
          f"(= 2 x {N_MFCC} from {POOLING} pooling)")

    # --- train + log ------------------------------------------------------
    setup_mlflow()  # points at sqlite:///mlflow.db, experiment "like-a-bosch"
    with mlflow.start_run(run_name=f"baseline-{args.model}"):
        mlflow.set_tag("stage", "3-baseline")
        # feature contract + model config, so runs are comparable in the UI
        mlflow.log_params({
            "model_family": "classical-baseline",
            "classifier": args.model,
            "features": "mfcc",
            "n_mfcc": N_MFCC,
            "pooling": POOLING,
            "sample_rate": TARGET_SR,
            "clip_seconds": CLIP_SECONDS,
            "seed": args.seed,
        })

        model = make_model(args.model, args.seed)
        model.fit(X_train, y_train)

        # evaluate on VALIDATION (test stays untouched until Stage 5)
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, val_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, val_pred, average="binary", pos_label=1, zero_division=0)
        auc = roc_auc_score(y_val, val_proba)

        mlflow.log_metrics({
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_auc": auc,
        })

        # artifacts: confusion matrix image + text report
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cm = confusion_matrix(y_val, val_pred, labels=[0, 1])
            plot_confusion(cm, tmp / "val_confusion.png",
                           f"baseline-{args.model} (val)")
            (tmp / "val_report.txt").write_text(
                classification_report(y_val, val_pred,
                                      target_names=["negative", "positive"],
                                      zero_division=0))
            mlflow.log_artifacts(str(tmp), artifact_path="eval")

        mlflow.sklearn.log_model(model, name="model")

        # --- print results so you SEE them, not just log them -------------
        print(f"\n=== baseline-{args.model} — VALIDATION results ===")
        print(f"  accuracy   {acc:.3f}")
        print(f"  precision  {prec:.3f}   (of clips called positive, how many were)")
        print(f"  recall     {rec:.3f}   (of real positives, how many caught)")
        print(f"  f1         {f1:.3f}")
        print(f"  auc        {auc:.3f}")
        print("  confusion matrix [[TN FP][FN TP]]:")
        print("   ", cm.tolist())
        print("\nLogged to MLflow experiment "
              f"'{EXPERIMENT_NAME}'. View with:  mlflow ui")


if __name__ == "__main__":
    main()
