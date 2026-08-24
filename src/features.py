"""
features.py — audio -> log-Mel spectrogram.

This is the SINGLE source of truth for feature extraction. Both the training
code and the serving code import `wav_to_logmel` from here, so the two can
never drift apart. That train/serve consistency is the whole reason this lives
in one module instead of being reimplemented in two places.

If you change any constant below, `tests/test_features.py` will fail on purpose.
That failure is the guardrail: it forces you to notice that every model trained
before the change now expects a different input than the server produces.
"""

from __future__ import annotations

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Frozen feature parameters. These define the contract between train and serve.
# ---------------------------------------------------------------------------
TARGET_SR = 16_000      # resample everything to 16 kHz (speech lives < 8 kHz)
CLIP_SECONDS = 3.0      # every clip is exactly 3 s by collection design
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)  # 48_000

N_FFT = 512             # ~32 ms analysis window at 16 kHz
WIN_LENGTH = 400        # 25 ms window
HOP_LENGTH = 160        # 10 ms hop -> ~301 frames over 3 s
N_MELS = 64             # frequency resolution of the mel filterbank
FMIN = 20               # ignore sub-audible rumble
FMAX = 8000             # Nyquist at 16 kHz

TOP_DB = 80.0           # matches the -80..0 dB floor used in the original project

# Derived output shape, computed once so callers and tests agree on it.
# librosa uses center-padding, so n_frames = 1 + floor(CLIP_SAMPLES / HOP_LENGTH).
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH  # 301
FEATURE_SHAPE = (N_MELS, N_FRAMES)         # (64, 301)


def load_audio(path: str) -> np.ndarray:
    """Load a wav from disk as mono float32 at TARGET_SR, fixed to CLIP_SAMPLES."""
    # librosa.load resamples to TARGET_SR and downmixes to mono for us.
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return fix_length(y)


def fix_length(y: np.ndarray) -> np.ndarray:
    """Pad with zeros or truncate so the signal is exactly CLIP_SAMPLES long.

    Clips are already 3 s, but this makes the pipeline robust to a clip that is
    off by a few samples, and — importantly — it is the SAME operation the live
    mic client will need when it hands us a 3 s window that is a sample or two
    short. Serving and training must fix length identically.
    """
    if len(y) > CLIP_SAMPLES:
        return y[:CLIP_SAMPLES]
    if len(y) < CLIP_SAMPLES:
        return np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y


def signal_to_logmel(y: np.ndarray) -> np.ndarray:
    """Core transform: a fixed-length waveform -> normalized log-Mel in [0, 1].

    Steps:
      1. mel power spectrogram
      2. power -> dB with a fixed 80 dB floor (values land in [-80, 0])
      3. shift/scale that fixed range to [0, 1]

    Note we normalize against a *fixed* range, not against dataset statistics.
    That means no stored mean/std to keep in sync between train and serve — the
    transform is completely self-contained and deterministic per clip.
    """
    y = fix_length(np.asarray(y, dtype=np.float32))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    # ref=np.max normalizes each clip to its own peak, so top_db clips at -80 dB.
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)  # [-80, 0]

    # Map [-80, 0] -> [0, 1]. Fully determined by TOP_DB; no dataset stats.
    normed = (log_mel + TOP_DB) / TOP_DB
    return normed.astype(np.float32)


def wav_to_logmel(path: str) -> np.ndarray:
    """Convenience wrapper: file path -> normalized log-Mel of shape FEATURE_SHAPE."""
    return signal_to_logmel(load_audio(path))


# ---------------------------------------------------------------------------
# MFCC features — the representation the SHIPPED model (logreg) uses.
#
# This is the single source of truth for MFCC extraction. Training, hard-negative
# mining, and the serving container all import mfcc_features / mfcc_features_from_file
# from here, so the production model can never be fed features computed a different
# way than it was trained on (the train/serve-skew guarantee). Keeping it in
# features.py — which only needs numpy + librosa — also means the serving container
# does not have to pull in matplotlib/mlflow just to featurize a clip.
# ---------------------------------------------------------------------------
N_MFCC = 40           # number of MFCC coefficients
POOLING = "mean+std"  # temporal pooling -> one fixed-length vector per clip
MFCC_FEATURE_LEN = 2 * N_MFCC  # 80: mean and std of each coefficient


def mfcc_features(y: np.ndarray) -> np.ndarray:
    """Fixed-length MFCC feature vector (length MFCC_FEATURE_LEN) from a waveform.

    MFCCs give an (N_MFCC, n_frames) matrix — still time-varying. A classical
    model needs a fixed-length vector, so we pool over time with mean and std of
    each coefficient. Mean = average spectral shape; std = how much it moves.
    """
    y = fix_length(np.asarray(y, dtype=np.float32))
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def mfcc_features_from_file(path: str) -> np.ndarray:
    """File path -> MFCC feature vector (loads + fixes length via load_audio)."""
    return mfcc_features(load_audio(path))


if __name__ == "__main__":
    # Tiny self-check you can run by hand: python src/features.py <some.wav>
    import sys
    if len(sys.argv) > 1:
        feats = wav_to_logmel(sys.argv[1])
        print(f"{sys.argv[1]} -> shape {feats.shape}, "
              f"min {feats.min():.3f}, max {feats.max():.3f}, dtype {feats.dtype}")
    else:
        print(f"Feature contract: shape {FEATURE_SHAPE}, "
              f"{TARGET_SR} Hz, {N_MELS} mels, {N_FRAMES} frames")
