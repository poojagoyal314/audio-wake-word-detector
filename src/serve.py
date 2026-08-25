"""
serve.py — Stage 6: the inference service.

A lean FastAPI app that loads the exported model once at startup and scores
uploaded audio clips. Deliberately imports only features.py (numpy + librosa) and
joblib — NOT train_baseline/mlflow/matplotlib — so the serving container stays
small and carries only what inference needs.

Endpoints:
  GET  /health   -> liveness + whether the model loaded (used by the container
                    health check in Stage 7).
  POST /predict  -> multipart file upload ('file'); returns probability, the
                    decision at the configured operating threshold, and the
                    threshold used.

Design (ties back to the whole project):
  * The client sends RAW AUDIO, not features. Featurization happens here, via the
    same features.mfcc_features the model was trained on -> no train/serve skew.
  * The windowing/debounce loop that turns a live mic stream into 3 s clips lives
    in the CLIENT (Stage 8), not here. This service scores one clip per call.

Run locally (host, before Docker):
    uvicorn serve:app --host 0.0.0.0 --port 8000
    # from repo root:  uvicorn --app-dir src serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException

# features.py is the ONLY project module the server needs for inference.
from features import mfcc_features, TARGET_SR, CLIP_SAMPLES, fix_length
import db  # prediction logging (Postgres) — optional, never breaks the request

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("serve")

# Model artifacts. MODEL_DIR env var lets the container point at wherever it baked
# them; the default resolves relative to the repo root (parent of src/), so it
# works no matter which directory uvicorn is launched from.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", _DEFAULT_MODEL_DIR))
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

# Loaded once at startup, not per request.
_model = None
_meta = {}
_threshold = 0.5
_model_version = "unknown"
_db_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown logic (modern replacement for @app.on_event('startup'),
    which is deprecated). Loads the model and brings up prediction logging."""
    global _model, _meta, _threshold, _model_version, _db_ready
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        log.info(f"model loaded from {MODEL_PATH}")
    else:
        log.warning(f"model file not found at {MODEL_PATH}")
    if META_PATH.exists():
        _meta = json.loads(META_PATH.read_text())
        _threshold = float(_meta.get("operating_threshold", 0.5))
        _model_version = _meta.get("trained_on", "v1")
    # Bring up prediction logging. Retries internally to bridge the depends_on
    # gap; returns False (logging disabled) rather than crashing if the DB is down.
    _db_ready = db.init_db()
    log.info(f"startup complete — db_logging={_db_ready}")
    yield
    # (nothing to clean up on shutdown)


app = FastAPI(title="Like-a-Bosch detector", version="1.0", lifespan=lifespan)


def _decode_upload(raw: bytes) -> np.ndarray:
    """Bytes of an audio file -> mono float32 waveform at TARGET_SR.

    Uses soundfile (libsndfile) for wav/flac/ogg. The live client sends wav, so
    this path covers it. (m4a would need the ffmpeg decode path from fa_per_hour;
    not wired here because the client records wav.)
    """
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim > 1:              # stereo -> mono
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    return fix_length(data.astype(np.float32))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "threshold": _threshold,
        "db_logging": _db_ready,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run export_model.py.")
    raw = await file.read()
    try:
        y = _decode_upload(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    # SAME features the model was trained on (single source of truth).
    x = mfcc_features(y).reshape(1, -1)
    proba = float(_model.predict_proba(x)[0, 1])
    is_wake = proba >= _threshold

    # Log to Postgres AFTER computing the prediction. Wrapped so a DB failure
    # never breaks the response — resilience over consistency (the prediction
    # still returns even if the log row doesn't land).
    db.log_prediction(
        probability=proba,
        is_wake_word=bool(is_wake),
        threshold=_threshold,
        model_version=_model_version,
    )

    log.info(f"prediction: probability={proba:.4f} is_wake_word={is_wake} db_logged={logged}")

    return {
        "probability": round(proba, 4),
        "is_wake_word": bool(is_wake),
        "threshold": _threshold,
        "phrase": "like a Bosch",
    }
