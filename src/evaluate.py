"""
evaluate.py — Stage 5, part 1: the honest final evaluation.

Two things happen here that haven't happened before:

1. We break the seal on the TEST set. It was untouched through all of model
   selection (stages 3-4 only ever looked at val), so the number it gives is an
   unbiased estimate of how the chosen model generalizes. The chosen model is
   the logistic-regression baseline — it matched or beat the CNN on every val
   metric at a fraction of the cost, so it is what we evaluate and ship.

   The final model is refit on train + val (all non-test data). Model selection
   is done, so val's held-out job is finished and folding it back into training
   just gives the final model more data. Test stays genuinely unseen.

2. We sweep the decision threshold. Every metric so far used a 0.5 cutoff, which
   is arbitrary for a detector. Sweeping the threshold shows how false alarms
   (firing on non-keywords) trade against misses (ignoring real keywords), and
   lets us name a few sensible operating points instead of pretending 0.5 is
   special.

NOTE on false-accepts-per-hour: the false-positive RATE reported here is measured
on discrete, balanced test clips. That is NOT the same as false accepts per hour
on a continuous microphone stream, which has a wildly different base rate. That
number needs a continuous keyword-free recording and is Stage 5 part 2.

Run it:
    python src/evaluate.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_manifest, make_splits  # noqa: E402
from train_baseline import build_matrix, make_model, N_MFCC, POOLING  # noqa: E402
from tracking import setup_mlflow, EXPERIMENT_NAME  # noqa: E402


def sweep_thresholds(y_true, proba, grid=None):
    """At each threshold, compute precision, recall, FPR, and FRR.

    FRR (false reject rate) = fraction of real keywords missed  = FN / positives.
    FPR (false positive rate) = fraction of non-keywords fired on = FP / negatives.
    These are the two error types a detector trades off; the threshold is the dial.
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = n_pos - tp
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        frr = fn / n_pos if n_pos else 0.0
        rows.append((t, precision, recall, fpr, frr))
    return np.array(rows)  # columns: t, precision, recall, fpr, frr


def equal_error_rate(y_true, proba):
    """The threshold where FPR == FRR (the classic single-number detector summary)."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    frr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - frr)))
    eer = (fpr[idx] + frr[idx]) / 2
    return eer, float(thr[idx])


def high_precision_threshold(sweep, target=0.95):
    """Lowest threshold that reaches `target` precision (fewest misses at that precision)."""
    ok = sweep[sweep[:, 1] >= target]
    if len(ok) == 0:
        return None
    return float(ok[0, 0])  # smallest threshold meeting the precision target


def plot_roc(y_true, proba, auc, out_path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (recall)")
    ax.set_title("ROC — test set")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_threshold_curves(sweep, out_path):
    t, prec, rec, fpr, frr = sweep.T
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, prec, label="precision")
    ax.plot(t, rec, label="recall (1 - FRR)")
    ax.plot(t, fpr, label="FPR (false-alarm rate)", linestyle="--")
    ax.axvline(0.5, color="grey", linewidth=1, alpha=0.6)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("rate")
    ax.set_title("How the threshold trades misses vs false alarms")
    ax.legend(loc="center left")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final test-set evaluation + threshold sweep")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf"],
                        help="the chosen model to evaluate (default: logreg)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # --- data: same deterministic split; refit on train+val, test sealed ---
    df = load_manifest(args.manifest)
    splits = make_splits(df, seed=args.seed)
    print("Split sizes:")
    print(splits.summary())

    print("\nExtracting MFCC features...")
    X_train, y_train = build_matrix(splits.train, data_root)
    X_val, y_val = build_matrix(splits.val, data_root)
    X_test, y_test = build_matrix(splits.test, data_root)

    X_fit = np.concatenate([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    print(f"  final model trains on train+val: {len(X_fit)} clips; "
          f"test held out: {len(X_test)} clips")

    # --- fit + evaluate ONCE on test --------------------------------------
    model = make_model(args.model, args.seed)
    model.fit(X_fit, y_fit)

    proba = model.predict_proba(X_test)[:, 1]
    pred_05 = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred_05)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred_05, average="binary", pos_label=1, zero_division=0)
    auc = roc_auc_score(y_test, proba)
    eer, eer_thr = equal_error_rate(y_test, proba)

    sweep = sweep_thresholds(y_test, proba)
    hp_thr = high_precision_threshold(sweep, target=0.95)

    # --- log + report -----------------------------------------------------
    setup_mlflow()
    with mlflow.start_run(run_name=f"final-eval-{args.model}"):
        mlflow.set_tag("stage", "5-final-eval")
        mlflow.log_params({
            "model": args.model,
            "trained_on": "train+val",
            "features": "mfcc",
            "n_mfcc": N_MFCC,
            "pooling": POOLING,
            "seed": args.seed,
        })
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_auc": auc,
            "test_eer": eer,
        })

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cm = confusion_matrix(y_test, pred_05, labels=[0, 1])
            plot_roc(y_test, proba, auc, tmp / "test_roc.png")
            plot_threshold_curves(sweep, tmp / "threshold_curves.png")
            (tmp / "test_report.txt").write_text(
                classification_report(y_test, pred_05,
                                      target_names=["negative", "positive"],
                                      zero_division=0))
            # operating-points table
            lines = ["threshold,precision,recall,fpr,frr,note"]
            def row(t, note):
                pr = precision_score(y_test, (proba >= t).astype(int), zero_division=0)
                rc = recall_score(y_test, (proba >= t).astype(int), zero_division=0)
                fp = int(((proba >= t) & (y_test == 0)).sum())
                fn = int(((proba < t) & (y_test == 1)).sum())
                npos = int((y_test == 1).sum()); nneg = int((y_test == 0).sum())
                lines.append(f"{t:.3f},{pr:.3f},{rc:.3f},{fp/nneg:.3f},{fn/npos:.3f},{note}")
            row(0.5, "default")
            row(eer_thr, "equal-error-rate")
            if hp_thr is not None:
                row(hp_thr, "precision>=0.95")
            (tmp / "operating_points.csv").write_text("\n".join(lines))
            mlflow.log_artifacts(str(tmp), artifact_path="final_eval")

        print(f"\n=== FINAL TEST-SET results ({args.model}, threshold 0.5) ===")
        print(f"  accuracy   {acc:.3f}")
        print(f"  precision  {prec:.3f}")
        print(f"  recall     {rec:.3f}")
        print(f"  f1         {f1:.3f}")
        print(f"  auc        {auc:.3f}")
        print(f"  EER        {eer:.3f}  (at threshold {eer_thr:.3f})")
        print(f"  confusion  [[TN FP][FN TP]] = {cm.tolist()}")

        print("\n--- candidate operating points ---")
        for ln in lines:
            print("  " + ln)

        print("\nNOTE: the FPR above is on balanced test clips, NOT false accepts")
        print("per hour on a live stream. That needs a continuous keyword-free")
        print("recording — Stage 5 part 2.")
        print(f"\nLogged to MLflow experiment '{EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
