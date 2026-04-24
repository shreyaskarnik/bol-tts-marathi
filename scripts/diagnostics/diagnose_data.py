"""Diagnose whether training data is producing NaN/Inf in audio or mel.

If the symbol table is correct but training still NaNs from step 1, the next
suspect is the data pipeline — a corrupt WAV, silent audio leading to log(0),
or some other numerical edge case in the dataloader.

This script reproduces what StyleTTS2's meldataset.py does for a handful of
training samples, checks for NaN/Inf at every stage, and reports.

Run on pod:
    python3 /workspace/bol_run/scripts/diagnose_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace/bol_run/StyleTTS2")

import numpy as np
import soundfile as sf
import torch
import librosa


TRAINING_DIR = Path("/workspace/bol_run/training")
AUDIO_DIR = Path("/workspace/bol_run/dataset/audio")


def check_audio(wav_path: Path) -> dict:
    """Load audio, check for NaN/Inf, compute mel, check for NaN/Inf."""
    out = {"path": str(wav_path)}
    try:
        arr, sr = sf.read(str(wav_path))
    except Exception as e:
        out["read_error"] = str(e)
        return out
    arr = arr.astype(np.float32)
    out["sr"] = sr
    out["duration_s"] = round(len(arr) / sr, 2)
    out["shape"] = arr.shape
    out["has_nan"] = bool(np.isnan(arr).any())
    out["has_inf"] = bool(np.isinf(arr).any())
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["rms"] = float(np.sqrt(np.mean(arr ** 2)))

    # Check for silent frames (RMS too low → log(0))
    FRAME = 300
    if len(arr) > FRAME * 10:
        # RMS per frame
        n = len(arr) // FRAME
        frames = arr[: n * FRAME].reshape(n, FRAME)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        out["silent_frames_ratio"] = float((frame_rms < 1e-5).mean())
        out["min_frame_rms"] = float(frame_rms.min())

    # Compute mel (mimic meldataset.py's MelSpectrogram call)
    try:
        # StyleTTS2's preprocess: to_mel uses torchaudio's MelSpectrogram
        # with n_fft=2048, win=1200, hop=300, n_mels=80
        mel = librosa.feature.melspectrogram(
            y=arr, sr=sr, n_fft=2048, hop_length=300, win_length=1200,
            n_mels=80, fmin=0, fmax=8000,
        )
        # StyleTTS2 does: log(1e-5 + mel)  per meldataset.py line 65
        log_mel = np.log(1e-5 + mel)
        out["mel_shape"] = log_mel.shape
        out["mel_nan"] = bool(np.isnan(log_mel).any())
        out["mel_inf"] = bool(np.isinf(log_mel).any())
        out["mel_min"] = float(log_mel.min())
        out["mel_max"] = float(log_mel.max())
        out["mel_mean"] = float(log_mel.mean())
    except Exception as e:
        out["mel_error"] = str(e)

    return out


def main() -> None:
    # Load first N manifest entries
    train_list = TRAINING_DIR / "train_list.txt"
    with open(train_list) as f:
        lines = f.read().strip().split("\n")

    print(f"total train lines: {len(lines)}")
    print(f"audio dir: {AUDIO_DIR}")
    print()

    # Check first 5 and a random sampling
    import random
    random.seed(42)
    indices = list(range(5)) + random.sample(range(5, len(lines)), 5)

    any_bad = False
    for i in indices:
        line = lines[i]
        parts = line.split("|")
        if len(parts) < 3:
            print(f"[{i}] malformed line")
            continue
        wav_rel = parts[0]
        wav_path = AUDIO_DIR / wav_rel
        result = check_audio(wav_path)
        flag = ""
        if result.get("has_nan") or result.get("has_inf") or result.get("mel_nan") or result.get("mel_inf"):
            flag = "  ❌ NaN/Inf"
            any_bad = True
        elif result.get("silent_frames_ratio", 0) > 0.5:
            flag = "  ⚠️  mostly silent"
        print(f"[{i}] {wav_rel}{flag}")
        for k, v in result.items():
            if k == "path":
                continue
            print(f"     {k}: {v}")
        print()

    if any_bad:
        print("❌ FOUND NaN/Inf in data — that's our NaN culprit.")
    else:
        print("✓ Sample audio + mel are clean. NaN is coming from somewhere in model/loss forward.")
        print("  Next: need to inspect model forward pass directly.")


if __name__ == "__main__":
    main()
