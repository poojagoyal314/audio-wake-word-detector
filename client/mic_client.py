"""
mic_client.py — Stage 8: the live microphone client.

Turns the containerized /predict service into a real-time wake-word detector.
Holds the microphone (which is why it runs on the HOST, not in the container —
a container can't portably reach the mic), slides a 3-second window across the
live audio, and POSTs each window to the service. When the phrase is detected it
prints a clear signal, then debounces so one utterance isn't announced repeatedly.

The client is deliberately "dumb": capture -> POST raw audio -> react. All the ML
(featurization + model) lives behind /predict, so there's no feature code here to
drift from training. The windowing/hop/debounce mirrors fa_per_hour.py exactly —
the only difference is the audio comes from a live mic instead of a file.

Requirements (host-side):
    pip install sounddevice numpy soundfile requests
Run (with `docker compose up` already running):
    python client/mic_client.py
Stop with Ctrl+C.
"""

from __future__ import annotations

import io
import time
import argparse
import collections

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests

# --- must match the server / features.py contract --------------------------
TARGET_SR = 16_000        # record straight at 16 kHz (what the model expects)
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)   # 48_000
HOP_SECONDS = 0.5         # score a window every 0.5 s
DEBOUNCE_SECONDS = 3.0    # after a detection, stay quiet this long

DEFAULT_URL = "http://localhost:8000/predict"


def score_window(window: np.ndarray, url: str, timeout: float = 5.0):
    """POST a 3 s window as an in-memory WAV to /predict; return the JSON dict."""
    buf = io.BytesIO()
    sf.write(buf, window, TARGET_SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    resp = requests.post(url, files={"file": ("window.wav", buf, "audio/wav")},
                         timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 'like a Bosch' mic detector")
    parser.add_argument("--url", default=DEFAULT_URL, help="the /predict endpoint")
    parser.add_argument("--device", type=int, default=None,
                        help="input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true",
                        help="list audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # rolling buffer holding the most recent CLIP_SAMPLES samples
    buffer = collections.deque(maxlen=CLIP_SAMPLES)
    hop_samples = int(TARGET_SR * HOP_SECONDS)

    # sanity: is the service reachable before we start listening?
    health_url = args.url.rsplit("/", 1)[0] + "/health"
    try:
        h = requests.get(health_url, timeout=5).json()
        if not h.get("model_loaded"):
            print("WARNING: service is up but no model is loaded. "
                  "Run export_model.py and rebuild the container.")
        print(f"Service healthy. Operating threshold = {h.get('threshold')}.")
    except Exception as e:
        print(f"ERROR: cannot reach the service at {health_url}: {e}")
        print("Is `docker compose up` running?")
        return

    print("\nListening for 'like a Bosch'...  (Ctrl+C to stop)\n")

    cooldown_until = 0.0  # wall-clock time until which we suppress detections

    # audio callback fills the rolling buffer from the live mic
    def audio_cb(indata, frames, time_info, status):
        if status:
            # e.g. input overflow — non-fatal, just note it
            print(f"(audio status: {status})")
        buffer.extend(indata[:, 0])  # mono channel

    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                            device=args.device, callback=audio_cb):
            while True:
                time.sleep(HOP_SECONDS)

                # need a full window before scoring
                if len(buffer) < CLIP_SAMPLES:
                    continue

                window = np.array(buffer, dtype=np.float32)
                try:
                    result = score_window(window, args.url)
                except Exception as e:
                    print(f"(request failed, skipping window: {e})")
                    continue

                prob = result.get("probability", 0.0)
                is_wake = result.get("is_wake_word", False)

                now = time.time()
                if is_wake and now >= cooldown_until:
                    print(f"  ✓ DETECTED  'like a Bosch'   (p={prob:.3f})")
                    cooldown_until = now + DEBOUNCE_SECONDS   # debounce
                else:
                    # live readout so you can see it working; overwrite in place
                    state = "cooldown" if now < cooldown_until else "listening"
                    print(f"    {state:<9}  p={prob:.3f}", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
