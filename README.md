# Like a Bosch — a wake-word detector

A spoken-phrase detector that listens for "like a Bosch" and ignores everything else. The model is deliberately modest; the point of the project is doing the thing properly — honest evaluation, a defensible model choice, and a containerized, database-backed service you can actually run.

> The headline finding: the model scored **0.91 accuracy / 0.97 AUC** on a held-out test set — but when evaluated on a real hour of keyword-free audio, it false-triggered **~300 times per hour**, which is unusable. Diagnosing why (the base-rate problem) and fixing it with **hard negative mining** (cutting false accepts **~96%** with no loss of recall) is the core of the project. Clean offline metrics can hide a model that fails in deployment; this project measures the gap instead of assuming it away.

---

## What it does

Speech in → `like a Bosch` or `not` out, as a probability plus a decision. It runs as a two-container application: a FastAPI inference service and a Postgres database that logs every prediction. A thin microphone client streams live audio to the service for real-time detection.

## Why it's interesting

Most audio-classification demos stop at "I got 92% accuracy." This one goes further and is honest about where models actually break:

- **A measured model choice, not an assumed one.** A classical baseline (MFCC + logistic regression) was compared against a from-scratch CNN in a tracked experiment. The baseline *matched or beat* the CNN on every metric at a fraction of the size and cost — so the lighter model shipped. Deep learning isn't free, and on this data it didn't earn its place.
- **Deployment-honest evaluation.** Beyond accuracy: a threshold sweep with named operating points, Equal Error Rate, and **false-accepts-per-hour** measured over real keyword-free audio — the metric that reflects how a detector actually behaves on a live stream.
- **A closed improvement loop.** The real-world failure was diagnosed (too-narrow negative training class) and fixed by mining the detector's own false triggers as new negatives, then re-measuring on held-out audio to prove the fix honestly.
- **Production-shaped engineering.** A single source of truth for feature extraction (so training and serving can never drift), a lean serving image, service-name networking, a persistent database, and a clean feature-branch Git history.

## Architecture

```
┌────────────┐    audio clip     ┌─────────────────────┐      ┌────────────┐
│ mic client │ ────────────────▶ │  FastAPI  /predict  │ ───▶ │  Postgres  │
│ (host)     │ ◀──────────────── │  (container)        │      │ (container)│
└────────────┘   prob + decision └─────────────────────┘      └────────────┘
   holds the mic,                  featurizes + scores,          logs every
   3s window + debounce            logs each prediction          prediction
```

Audio *capture* stays on the host (a container can't portably reach a microphone); *inference* runs in the container. The two talk over HTTP.

## Quick start

**Requirements:** Docker Desktop. (For training/evaluation scripts: Python 3.12 and the packages in `requirements.txt`.)

```bash
# 1. produce the model artifact on the host (needs the training data)
python src/export_model.py

# 2. bring up the multi-container app
docker compose up --build

# 3. try it — open the interactive API docs and upload a clip
#    http://localhost:8000/docs
```

Query the logged predictions:

```bash
docker compose exec db psql -U iris -d predictions -c "SELECT * FROM predictions;"
docker compose exec db psql -U iris -d predictions \
  -c "SELECT is_wake_word, COUNT(*) FROM predictions GROUP BY is_wake_word;"
```

## How it was built

The project was built in stages, each on its own feature branch:

| Stage | What |
|---|---|
| 1–2 | Data manifest, stratified split, shared feature extraction (log-Mel / MFCC) with a skew-guard test |
| 3 | MFCC classical baseline (logreg / random forest) with MLflow tracking |
| 4 | Small log-Mel CNN — the measured baseline-vs-deep-learning comparison |
| 5 | Evaluation: test-set metrics, threshold sweep + EER, false-accepts-per-hour, hard negative mining |
| 6 | Containerized inference service (FastAPI, model baked in) |
| 7 | Multi-container app: Postgres prediction logging via Docker Compose |
| 8 | Live microphone client |

Full reasoning — every decision, result, and stumble — is in [`Audio_Classification_Model_Development_Journey.md`](Audio_Classification_Model_Development_Journey.md).

## Results

| Metric (test set, logreg) | Value |
|---|---|
| Accuracy | 0.906 |
| AUC | 0.969 |
| Equal Error Rate | 0.104 (at threshold 0.639) |

**False accepts per hour** on real keyword-free audio, before vs. after hard negative mining (held-out recording):

| Threshold | Before | After |
|---|---|---|
| 0.50 | 103 / hr | ~4 / hr |
| 0.79 | 38 / hr | 0 / hr |

## Tech

Python · scikit-learn · librosa · FastAPI · Docker & Docker Compose · PostgreSQL · MLflow

## Project layout

```
src/
  collect_audio.py        # recorder used to build the dataset
  make_manifest.py        # dataset manifest
  data.py                 # stratified train/val/test split
  features.py             # SINGLE source of truth for audio → features
  train_baseline.py       # MFCC baseline (+ MLflow)
  train_cnn.py            # log-Mel CNN (+ MLflow)
  evaluate.py             # test metrics, threshold sweep, EER
  fa_per_hour.py          # false-accepts-per-hour on a cold stream
  hard_negative_mining.py # mine false triggers, retrain, re-measure
  export_model.py         # produce the deployable model artifact
  serve.py                # FastAPI inference service
  db.py                   # Postgres prediction logging
tests/
  test_features.py        # locks the feature contract (skew guard)
Dockerfile · compose.yaml · requirements*.txt
```

## Notes & future work

Deliberately deferred, and why: persisting mined hard negatives as tracked dataset files (with provenance); mining across more diverse audio; an audio-pretrained comparison (YAMNet/PANNs); a Compose health check to replace the `depends_on` startup gap; and ONNX export for edge deployment. Each is a considered "later," not an oversight.

*The dataset (voice recordings) is not included for privacy reasons; `src/collect_audio.py` builds a new one.*
