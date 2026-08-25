# syntax=docker/dockerfile:1

# --- Base image -------------------------------------------------------------
# python:3.12-slim, not full (too big) and not alpine: alpine uses musl instead
# of glibc, and the scientific wheels we need (numpy, scipy, scikit-learn,
# librosa) ship as glibc binaries. On alpine pip would compile them from source
# — slow, fragile, and ultimately larger. slim is the sane default for ML.
FROM python:3.12-slim


# Flush Python stdout/stderr immediately so print() shows up in `docker compose
# logs` in real time (default buffering can hide startup messages entirely).
ENV PYTHONUNBUFFERED=1


# --- System (OS-level) dependencies ----------------------------------------
# libsndfile1 and ffmpeg are NOT Python packages, so pip cannot install them.
# They are OS-level libraries that librosa/soundfile need to read audio:
#   libsndfile1 -> wav/flac/ogg decoding
#   ffmpeg      -> m4a/mp3/aac decoding (the format the phone recorder produces)
# Install them with apt, clean the apt cache in the SAME layer to keep the image
# small (a separate cleanup layer wouldn't actually shrink the image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer) ------------------------------------
# Copy requirements FIRST and install, before copying the code. Editing code
# then does not invalidate this expensive layer, so rebuilds are fast. Only a
# change to the requirements file re-runs the install.
#
# We install requirements-SERVE.txt, not the full requirements.txt: the serving
# image needs only inference deps, not tensorflow/mlflow/matplotlib/pytest. That
# keeps the image small — the same "serve carries only what it needs" principle
# behind serve.py importing only features.py.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# --- Application code + model artifact --------------------------------------
# The code (serve.py imports only features.py at runtime, but we copy the whole
# src/ for simplicity) and the pre-trained model produced on the host by
# export_model.py. The model is baked in: it changes rarely and we want the
# image to be self-contained. .dockerignore keeps data/, mlruns/, mlflow.db,
# and the venv OUT of the build — but deliberately lets models/model.joblib in.
COPY src/ ./src/
COPY models/ ./models/

# Tell the app where the baked-in model lives (serve.py reads MODEL_DIR).
ENV MODEL_DIR=/app/models

# --- Networking -------------------------------------------------------------
# EXPOSE is documentation only; it does not publish the port. Publishing happens
# at run time with -p (or Compose ports:). The app must bind 0.0.0.0 (below), not
# 127.0.0.1, or the published port can't reach it from the host.
EXPOSE 8000

# --- Start the service ------------------------------------------------------
# JSON-array (exec) form so uvicorn receives shutdown signals directly and
# `docker stop` exits cleanly instead of being force-killed after a timeout.
# --app-dir src makes 'serve:app' importable; --host 0.0.0.0 is the binding that
# makes the container reachable from the host.
CMD ["uvicorn", "--app-dir", "src", "serve:app", "--host", "0.0.0.0", "--port", "8000"]


