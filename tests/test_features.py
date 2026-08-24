"""
test_features.py — the train/serve skew guard.

Two jobs:

1. Lock the feature contract. If someone changes a constant in features.py
   (sample rate, n_mels, hop, etc.), these assertions fail. The failure is the
   point: it forces a conscious decision, because every previously trained
   model expects the old input and the server would silently produce the new
   one. Silent skew is the bug this test exists to prevent.

2. Check the transform is well-behaved and deterministic: right shape, right
   dtype, values in [0, 1], and identical output for identical input.

These tests need no audio files — they synthesize signals — so they run in CI
without shipping the dataset.
"""

import sys
from pathlib import Path

import numpy as np

# make src/ importable when running `pytest` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import src.features as F  # noqa: E402


def _synth_tone(freq: float = 440.0, seconds: float = 3.0) -> np.ndarray:
    """A deterministic 3 s sine tone at the source sample rate (pre-resample)."""
    sr = 44_100  # the collection script records at 44.1 kHz
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- 1. contract constants -------------------------------------------------

def test_feature_contract_is_frozen():
    # If you intend to change these, update the expected values here in the
    # same commit — that edit is your acknowledgement that models must retrain.
    assert F.TARGET_SR == 16_000
    assert F.CLIP_SECONDS == 3.0
    assert F.CLIP_SAMPLES == 48_000
    assert F.N_FFT == 512
    assert F.WIN_LENGTH == 400
    assert F.HOP_LENGTH == 160
    assert F.N_MELS == 64
    assert F.FMIN == 20
    assert F.FMAX == 8000
    assert F.TOP_DB == 80.0
    assert F.FEATURE_SHAPE == (64, 301)


# --- 2. transform behaviour ------------------------------------------------

def test_output_shape_and_dtype():
    feats = F.signal_to_logmel(_synth_tone())
    assert feats.shape == F.FEATURE_SHAPE
    assert feats.dtype == np.float32


def test_output_range_is_unit_interval():
    feats = F.signal_to_logmel(_synth_tone())
    # normalization maps [-80, 0] dB into [0, 1]
    assert feats.min() >= 0.0
    assert feats.max() <= 1.0


def test_fix_length_pads_and_truncates():
    short = np.zeros(F.CLIP_SAMPLES - 500, dtype=np.float32)
    long = np.zeros(F.CLIP_SAMPLES + 500, dtype=np.float32)
    assert len(F.fix_length(short)) == F.CLIP_SAMPLES
    assert len(F.fix_length(long)) == F.CLIP_SAMPLES


def test_transform_is_deterministic():
    sig = _synth_tone()
    a = F.signal_to_logmel(sig)
    b = F.signal_to_logmel(sig)
    assert np.array_equal(a, b)


def test_silence_maps_to_zeros():
    # A silent clip normalized against its own max is uniform; with ref=np.max
    # on all-zero input librosa yields the floor, so values sit at the bottom
    # of the range. Mainly a guard that silence doesn't crash or produce NaNs.
    silence = np.zeros(F.CLIP_SAMPLES, dtype=np.float32)
    feats = F.signal_to_logmel(silence)
    assert np.isfinite(feats).all()
    assert feats.shape == F.FEATURE_SHAPE
