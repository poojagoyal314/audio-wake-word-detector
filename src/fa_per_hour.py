"""
fa_per_hour.py — Stage 5, part 2: false accepts per hour on a cold stream.

The test set gave a false-positive RATE on balanced discrete clips. That is not
how the detector behaves on a live microphone, where the audio is essentially
all non-keyword and the base rate is nothing like 50/50. This script measures
the number that actually matters for deployment: how often does the detector
fire on a continuous, keyword-free recording, expressed per hour.

How it works
------------
* Loads every recording in data/cold_stream/ (any format ffmpeg can read: m4a,
  mp3, wav...), resampled to the same 16 kHz mono as training.
* Slides a 3 s window across each stream, hopping HOP_SECONDS each step, and
  scores every window with the chosen model using the SAME features.py pipeline
  used in training — so the stream is processed identically to the clips.
* A window "triggers" if its probability >= threshold. Because every window is
  keyword-free, every trigger is a FALSE accept.
* Debounce: after a trigger, ignore further triggers for DEBOUNCE_SECONDS. This
  mirrors the deployment design ("wait 3 s to avoid re-detecting the same
  audio") and means we count detection EVENTS, not overlapping windows — one
  sustained false trigger is one false accept, not ten.
* Repeats the count at each candidate threshold and divides by the total stream
  hours to get false accepts / hour.

Run it (after training is available and clips are in data/cold_stream/):
    python src/fa_per_hour.py
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    signal_to_logmel, TARGET_SR, CLIP_SAMPLES, CLIP_SECONDS,
)
from data import load_manifest, make_splits  # noqa: E402
from train_baseline import build_matrix, make_model  # noqa: E402

HOP_SECONDS = 0.5        # how far the window jumps each step
DEBOUNCE_SECONDS = 3.0   # ignore new triggers for this long after one fires
HOP_SAMPLES = int(TARGET_SR * HOP_SECONDS)

# Score stream windows with the SAME MFCC features the model was trained on —
# the canonical extractor from features.py (single source of truth). This alias
# keeps the in-memory name used elsewhere (e.g. hard_negative_mining).
from features import mfcc_features as mfcc_vector_from_signal  # noqa: E402,F401


def load_stream(path: Path) -> np.ndarray:
    """Load any audio file as mono float32 at 16 kHz.

    Newer librosa loads via libsndfile, which does NOT support m4a/AAC. So for
    formats libsndfile can't read we decode with ffmpeg directly (ffmpeg reads
    m4a/mp3/aac fine) and hand the raw PCM to numpy. WAV/FLAC/OGG still go the
    fast libsndfile path.
    """
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
        return y
    return _load_via_ffmpeg(path)


def _ffmpeg_exe() -> str:
    """Locate an ffmpeg executable.

    Prefer the binary bundled in the imageio-ffmpeg pip package (no system
    install needed, keeps everything inside the venv). Fall back to a system
    'ffmpeg' on PATH — which is what the Docker image will have. This is why the
    same code works both on a bare Windows box (via pip) and in the container.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # rely on system PATH (e.g. inside the container)


def _load_via_ffmpeg(path: Path) -> np.ndarray:
    """Decode arbitrary audio (m4a, mp3, aac) to mono 16 kHz float32 via ffmpeg."""
    import subprocess
    cmd = [
        _ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(TARGET_SR),
        "-f", "f32le", "-",  # raw 32-bit float little-endian to stdout
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode {path.name}:\n{proc.stderr.decode(errors='ignore')}\n"
            "Install ffmpeg — either 'pip install imageio-ffmpeg' or a system ffmpeg on PATH."
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def window_probabilities(y: np.ndarray, model) -> np.ndarray:
    """Slide a 3 s window with HOP_SAMPLES step; return one probability per window."""
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    starts = range(0, len(y) - CLIP_SAMPLES + 1, HOP_SAMPLES)
    feats = [mfcc_vector_from_signal(y[s:s + CLIP_SAMPLES]) for s in starts]
    if not feats:
        return np.array([])
    X = np.stack(feats)
    return model.predict_proba(X)[:, 1]


def count_false_accepts(probs: np.ndarray, threshold: float) -> int:
    """Count debounced trigger EVENTS among windows exceeding the threshold."""
    debounce_windows = int(round(DEBOUNCE_SECONDS / HOP_SECONDS))
    events = 0
    cooldown = 0
    for p in probs:
        if cooldown > 0:
            cooldown -= 1
            continue
        if p >= threshold:
            events += 1
            cooldown = debounce_windows  # ignore the next few windows
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="False accepts per hour on a cold stream")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--stream-dir", default="data/cold_stream")
    parser.add_argument("--model", default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", default="0.5,0.639,0.79",
                        help="comma-separated thresholds to report (default: the "
                             "operating points from evaluate.py)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    stream_dir = Path(args.stream_dir)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    # --- train the chosen model on train+val (same as final eval) ---------
    df = load_manifest(args.manifest)
    splits = make_splits(df, seed=args.seed)
    X_train, y_train = build_matrix(splits.train, data_root)
    X_val, y_val = build_matrix(splits.val, data_root)
    X_fit = np.concatenate([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    model = make_model(args.model, args.seed)
    model.fit(X_fit, y_fit)
    print(f"Trained {args.model} on {len(X_fit)} clips.\n")

    # --- gather the cold stream -------------------------------------------
    audio_files = sorted(
        p for p in stream_dir.iterdir()
        if p.suffix.lower() in {".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac"}
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {stream_dir}")

    print(f"Cold stream: {len(audio_files)} file(s)")
    all_probs = []
    total_seconds = 0.0
    for f in audio_files:
        y = load_stream(f)
        dur = len(y) / TARGET_SR
        total_seconds += dur
        probs = window_probabilities(y, model)
        all_probs.append(probs)
        print(f"  {f.name:<28} {dur/60:6.2f} min   {len(probs):>5} windows")
    probs = np.concatenate(all_probs)
    total_hours = total_seconds / 3600.0
    print(f"\nTotal: {total_seconds/60:.1f} min ({total_hours:.3f} h), "
          f"{len(probs)} windows scored.")
    print(f"Window = {CLIP_SECONDS}s, hop = {HOP_SECONDS}s, "
          f"debounce = {DEBOUNCE_SECONDS}s\n")

    # --- false accepts per hour at each threshold -------------------------
    print("=== FALSE ACCEPTS PER HOUR (keyword-free stream) ===")
    print(f"  {'threshold':>9}   {'false accepts':>13}   {'per hour':>9}")
    for t in thresholds:
        fa = count_false_accepts(probs, t)
        fa_hr = fa / total_hours if total_hours else 0.0
        print(f"  {t:>9.3f}   {fa:>13d}   {fa_hr:>9.2f}")

    print("\nInterpretation: at a given threshold, this is how many times the")
    print("detector would wrongly wake per hour of ordinary, keyword-free audio.")
    print("Higher threshold -> fewer false wakes, but (from evaluate.py) more")
    print("missed real utterances. Pick the operating point from BOTH tables.")


if __name__ == "__main__":
    main()
