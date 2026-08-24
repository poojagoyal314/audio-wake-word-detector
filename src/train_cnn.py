"""
train_cnn.py — Stage 4.

A small, purpose-built CNN on log-Mel spectrograms. Deliberately tiny (~tens of
thousands of parameters, not VGG16's 138 million) because the dataset is small
and the baseline is already strong — a big model would overfit and would be the
wrong thing to deploy on a live mic.

Its real job in this project: put a *measured* number next to the baseline so the
"is the heavier model worth it?" decision rests on evidence, not assertion. It is
evaluated on the SAME validation split as the baseline (identical seed + manifest),
so the two land in the same MLflow charts and are directly comparable. The test
set stays sealed until Stage 5.

Run it:
    python src/train_cnn.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import os
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF's info spam
import tensorflow as tf

import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import wav_to_logmel, FEATURE_SHAPE, TARGET_SR, N_MELS, N_FRAMES  # noqa: E402
from data import load_manifest, make_splits  # noqa: E402
from tracking import setup_mlflow, EXPERIMENT_NAME  # noqa: E402


def featurize(df, data_root: Path):
    """Split DataFrame -> (X, y) where X is (N, n_mels, n_frames, 1)."""
    feats = [wav_to_logmel(str(data_root / fn)) for fn in df["filename"]]
    X = np.stack(feats).astype(np.float32)[..., np.newaxis]  # add channel dim
    y = df["y"].to_numpy().astype(np.float32)
    return X, y


def build_cnn(input_shape, seed: int) -> tf.keras.Model:
    """A compact 3-block conv net.

    Each block: Conv -> BatchNorm -> ReLU -> MaxPool, widening 16 -> 32 -> 64.
    GlobalAveragePooling collapses the feature map to one vector per clip (far
    fewer parameters than a Flatten + big Dense, which is what keeps this small
    and overfitting-resistant). Dropout before the single sigmoid output adds a
    little regularization. Binary problem -> one output unit.
    """
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        # momentum=0.9 (not the 0.99 default): with few batches per epoch the
        # running mean/var that BatchNorm uses at inference converge slowly,
        # which can leave val predictions miscalibrated. A lower momentum lets
        # those running stats track the data faster on a small dataset.
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs, name="logmel_cnn")


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
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_history(history, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_title("loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history["auc"], label="train")
    ax2.plot(history["val_auc"], label="val")
    ax2.set_title("AUC"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the log-Mel CNN")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # --- data (SAME split as the baseline: same seed, same manifest) ------
    df = load_manifest(args.manifest)
    splits = make_splits(df, seed=args.seed)
    print("Split sizes:")
    print(splits.summary())

    print("\nExtracting log-Mel features...")
    X_train, y_train = featurize(splits.train, data_root)
    X_val, y_val = featurize(splits.val, data_root)
    input_shape = (N_MELS, N_FRAMES, 1)
    print(f"  X_train {X_train.shape}  X_val {X_val.shape}")

    # --- model ------------------------------------------------------------
    model = build_cnn(input_shape, args.seed)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    n_params = model.count_params()
    print(f"\nCNN parameters: {n_params:,}  (vs VGG16's ~138,000,000)")

    # Monitor val_LOSS, not val_auc. AUC rewards ranking; loss rewards
    # calibration. We threshold probabilities at 0.5, so we want the epoch with
    # the best-calibrated probabilities, not the one that merely ranks best
    # (which on easy data is an early, undertrained epoch whose probabilities
    # all sit below 0.5 — perfect AUC, useless at a 0.5 cut). restore_best_weights
    # keeps that best-loss epoch rather than the last.
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10, restore_best_weights=True)

    # --- train + log ------------------------------------------------------
    setup_mlflow()
    with mlflow.start_run(run_name="cnn-logmel"):
        mlflow.set_tag("stage", "4-cnn")
        mlflow.log_params({
            "model_family": "cnn",
            "architecture": "3xConv(16,32,64)+GAP",
            "features": "log-mel",
            "input_shape": f"{input_shape}",
            "sample_rate": TARGET_SR,
            "n_mels": N_MELS,
            "n_frames": N_FRAMES,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "n_params": n_params,
            "seed": args.seed,
        })

        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early],
            verbose=2,
        )

        # log the per-epoch curves so overfitting is visible in the UI
        for epoch in range(len(hist.history["loss"])):
            mlflow.log_metrics({
                "epoch_train_loss": hist.history["loss"][epoch],
                "epoch_val_loss": hist.history["val_loss"][epoch],
                "epoch_train_auc": hist.history["auc"][epoch],
                "epoch_val_auc": hist.history["val_auc"][epoch],
            }, step=epoch)

        # --- evaluate on val, SAME metrics/threshold as the baseline ------
        val_proba = model.predict(X_val, verbose=0).ravel()
        val_pred = (val_proba >= 0.5).astype(int)

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
            "epochs_trained": len(hist.history["loss"]),
        })

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cm = confusion_matrix(y_val, val_pred, labels=[0, 1])
            plot_confusion(cm, tmp / "val_confusion.png", "cnn-logmel (val)")
            plot_history(hist.history, tmp / "training_curves.png")
            (tmp / "val_report.txt").write_text(
                classification_report(y_val, val_pred,
                                      target_names=["negative", "positive"],
                                      zero_division=0))
            model.save(tmp / "model.keras")
            mlflow.log_artifacts(str(tmp), artifact_path="eval")

        print("\n=== cnn-logmel — VALIDATION results ===")
        print(f"  accuracy   {acc:.3f}")
        print(f"  precision  {prec:.3f}")
        print(f"  recall     {rec:.3f}")
        print(f"  f1         {f1:.3f}")
        print(f"  auc        {auc:.3f}")
        print(f"  epochs trained (early stop): {len(hist.history['loss'])}")
        print("  confusion matrix [[TN FP][FN TP]]:", cm.tolist())
        print(f"\nLogged to MLflow experiment '{EXPERIMENT_NAME}'.")
        print("Compare against the baselines:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
