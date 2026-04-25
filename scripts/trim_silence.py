"""Trim leading + trailing silence from audio clips before voicepack extraction.

Voicepack style encoders average per-clip acoustic signatures into a single
ref_s vector that the decoder treats as voice identity. If clips have any
consistent silence padding at either end, that silence becomes part of the
voice — the decoder dutifully reproduces it on every synthesis.

We've hit this twice in bol-tts-marathi:
  * **Leading silence** (IndicTTS, fixed pre-Stage-1) — SPRINGLab clips have
    0.3-0.4 s of pre-speech silence. Result: voicepack ate the first word /
    rendered an audible hiss before speech started.
  * **Trailing silence** (Rasa, observed v0.1-preview) — Rasa Marathi clips
    don't tail-trim consistently. Result: word-final semi-vowels and
    aspirated stops get clipped on Asha + Vivek synthesis (e.g. final /j/
    in सांगतोय fades early). Mukta + Dnyanesh, derived from re-trimmed
    IndicTTS, don't show this.

Run on any audio dir before passing to `scripts/upstream/extract_voicepack.py`.
Defaults keep 50 ms of pad on each side — enough for natural attack/release,
small enough to not contaminate the style vector.

Usage:

    python scripts/trim_silence.py \
        --src-dir dataset/audio/rasa_marathi_female \
        --dst-dir dataset/audio/rasa_marathi_female_trimmed

    # then extract voicepack from the _trimmed dir as usual
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _find_speech_range(audio: np.ndarray, sr: int, threshold: float, win_sec: float) -> tuple[int, int]:
    """Return (start_sample, end_sample) for the speech region."""
    win = max(1, int(sr * win_sec))
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(win) / win, mode="same"))
    above = np.where(rms > threshold)[0]
    if len(above) == 0:
        return 0, len(audio)
    return int(above[0]), int(above[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True, type=Path)
    ap.add_argument("--dst-dir", required=True, type=Path)
    ap.add_argument("--lead-pad", type=float, default=0.05, help="keep N sec of pre-speech pad (default 50 ms)")
    ap.add_argument("--trail-pad", type=float, default=0.05, help="keep N sec of post-speech pad")
    ap.add_argument("--threshold", type=float, default=0.005, help="RMS threshold (0-1) for speech detection")
    ap.add_argument("--win-sec", type=float, default=0.01, help="RMS window (default 10 ms)")
    ap.add_argument("--pattern", default="*.wav")
    args = ap.parse_args()

    args.dst_dir.mkdir(parents=True, exist_ok=True)

    trimmed = skipped = 0
    for src in sorted(args.src_dir.glob(args.pattern)):
        try:
            audio, sr = sf.read(str(src), dtype="float32")
        except Exception:
            skipped += 1
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        start, end = _find_speech_range(audio, sr, args.threshold, args.win_sec)
        if end <= start:
            skipped += 1
            continue
        start = max(0, start - int(args.lead_pad * sr))
        end   = min(len(audio), end + int(args.trail_pad * sr))
        sf.write(str(args.dst_dir / src.name), audio[start:end], sr)
        trimmed += 1

    print(f"trimmed {trimmed} clips → {args.dst_dir}  (skipped {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
