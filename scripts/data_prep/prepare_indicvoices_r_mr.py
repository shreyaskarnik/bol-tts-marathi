"""prepare_indicvoices_r_mr.py — stream ai4bharat/indicvoices_r Marathi subset,
filter by duration/SNR/CER/scenario, downsample 48kHz→24kHz mono, phonemize via
misaki espeak, write WAVs + manifest fragment.

Output layout (relative to this repo root, bol-tts-marathi/):
  dataset/audio/indicvoices_r/<speaker_short>_<idx:06d>.wav   24 kHz mono 16-bit
  training/indicvoices_r_mr.txt                               path|ipa|speaker_name
  training/indicvoices_r_mr_stats.json                        run summary

Usage:
  python3 scripts/prepare_indicvoices_r_mr.py                # full run
  python3 scripts/prepare_indicvoices_r_mr.py --max 100      # smoke test with 100 rows
  python3 scripts/prepare_indicvoices_r_mr.py --dry-run      # stats-only, no writes

Requires: HF_TOKEN env var OR `hf auth login` already run. Dataset is CC-BY-4.0
but gated (requires one-click web acceptance first).

Resume: if output manifest exists, skips entries already written by (speaker,idx).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from misaki import espeak
from tqdm import tqdm

# ── Filter thresholds (adjust here; keep defaults in sync with README) ───────
DURATION_MIN = 2.0       # seconds
DURATION_MAX = 15.0
SNR_MIN_READ = 45.0
SNR_MIN_EXTEMPORE = 55.0
CER_MAX_EXTEMPORE = 0.05

TARGET_SR = 24_000

# ── Paths ────────────────────────────────────────────────────────────────────
# Script lives at scripts/data_prep/, so parents[2] = repo root.
import os
REPO_ROOT = Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[2]))
WAVS_DIR = REPO_ROOT / "dataset" / "audio" / "indicvoices_r"
MANIFEST_PATH = REPO_ROOT / "training" / "indicvoices_r_mr.txt"
STATS_PATH = REPO_ROOT / "training" / "indicvoices_r_mr_stats.json"


def short_speaker(sid: str) -> str:
    """S4259983900382350 → mr_s383 (stable 3-char suffix for readable filenames)."""
    return "mr_s" + hashlib.blake2s(sid.encode(), digest_size=2).hexdigest()


def passes_content_filter(row: dict) -> tuple[bool, str]:
    """Return (keep, reason). reason is 'ok' if keep=True, else the fail cause."""
    dur = row.get("duration")
    if dur is None or dur < DURATION_MIN:
        return False, "too_short"
    if dur > DURATION_MAX:
        return False, "too_long"

    scenario = row.get("scenario")
    if scenario not in ("Read", "Extempore"):
        return False, "bad_scenario"

    snr = row.get("snr") or 0.0
    if scenario == "Read" and snr < SNR_MIN_READ:
        return False, "low_snr_read"
    if scenario == "Extempore" and snr < SNR_MIN_EXTEMPORE:
        return False, "low_snr_extempore"

    # cer field is stringified tensor, e.g. "tensor(0.0402)"; parse it
    if scenario == "Extempore":
        cer_s = str(row.get("cer") or "")
        try:
            cer = float(cer_s.replace("tensor(", "").replace(")", ""))
        except ValueError:
            return False, "bad_cer"
        if cer > CER_MAX_EXTEMPORE:
            return False, "high_cer"

    text = (row.get("normalized") or row.get("text") or "").strip()
    if not text:
        return False, "empty_text"

    return True, "ok"


def resample_to_target(arr: np.ndarray, src_sr: int) -> np.ndarray:
    """48 kHz stereo-or-mono → 24 kHz mono float32 in [-1, 1]."""
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if src_sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=src_sr, target_sr=TARGET_SR)
    # Normalize to 0.95 peak to prevent clipping on int16 encode
    peak = np.abs(arr).max()
    if peak > 0:
        arr = arr / peak * 0.95
    return arr


def decode_audio(audio_obj) -> tuple[np.ndarray | None, int | None]:
    """Decode whatever HF datasets hands us into (np.float32, sr). None,None on failure."""
    if isinstance(audio_obj, dict):
        if audio_obj.get("array") is not None:
            return np.asarray(audio_obj["array"], dtype=np.float32), audio_obj["sampling_rate"]
        if audio_obj.get("bytes"):
            arr, sr = sf.read(io.BytesIO(audio_obj["bytes"]))
            return arr.astype(np.float32), sr
    return None, None


def load_existing_manifest(path: Path) -> set[tuple[str, int]]:
    """Return set of (speaker_short, idx) already written, for resume support."""
    seen: set[tuple[str, int]] = set()
    if not path.exists():
        return seen
    for line in path.read_text().splitlines():
        if not line or "|" not in line:
            continue
        wav_path = line.split("|", 1)[0]
        # indicvoices_r/mr_sXXX_000123.wav → (mr_sXXX, 123)
        stem = Path(wav_path).stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            seen.add((parts[0], int(parts[1])))
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="process at most N rows (for smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="filter-only stats, no writes or downloads")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        # fall back to file token
        token_path = Path.home() / ".cache/huggingface/token"
        if token_path.exists():
            os.environ["HF_TOKEN"] = token_path.read_text().strip()
        else:
            print("[FAIL] no HF_TOKEN env and no ~/.cache/huggingface/token", file=sys.stderr)
            return 2

    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prep] streaming ai4bharat/indicvoices_r:Marathi split={args.split}")
    ds = load_dataset(
        "ai4bharat/indicvoices_r",
        "Marathi",
        split=args.split,
        streaming=True,
    )
    # Decode audio manually (we need explicit control over resample)
    ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))

    print("[prep] initializing misaki Marathi G2P (may take ~10s)...")
    g2p = espeak.EspeakG2P(language="mr")

    seen = load_existing_manifest(MANIFEST_PATH)
    if seen:
        print(f"[prep] resuming, found {len(seen)} entries already written")

    counts: dict[str, int] = defaultdict(int)
    per_speaker_idx: dict[str, int] = defaultdict(int)
    total_seconds = 0.0

    manifest_out = MANIFEST_PATH.open("a" if seen else "w", encoding="utf-8")
    pbar = tqdm(desc="IV-R MR", unit="utt")

    try:
        for i, row in enumerate(ds):
            if args.max and i >= args.max:
                break
            counts["seen"] += 1

            keep, reason = passes_content_filter(row)
            counts[reason] += 1
            if not keep:
                pbar.update(1)
                continue

            sid = row["speaker_id"]
            short = short_speaker(sid)
            idx = per_speaker_idx[short]
            per_speaker_idx[short] += 1

            if (short, idx) in seen:
                counts["skipped_already_written"] += 1
                pbar.update(1)
                continue

            if args.dry_run:
                counts["would_write"] += 1
                total_seconds += row["duration"]
                pbar.update(1)
                continue

            # decode + resample
            arr, src_sr = decode_audio(row["audio"])
            if arr is None:
                counts["audio_decode_fail"] += 1
                pbar.update(1)
                continue
            arr = resample_to_target(arr, src_sr)

            # phonemize
            text = (row.get("normalized") or row["text"]).strip()
            try:
                ipa, _ = g2p(text)
            except Exception as e:
                counts["g2p_fail"] += 1
                pbar.update(1)
                continue

            # write wav
            wav_name = f"{short}_{idx:06d}.wav"
            wav_path = WAVS_DIR / wav_name
            sf.write(str(wav_path), arr, TARGET_SR, subtype="PCM_16")

            # write manifest entry
            rel_path = f"indicvoices_r/{wav_name}"
            manifest_out.write(f"{rel_path}|{ipa}|{short}\n")
            manifest_out.flush()

            counts["written"] += 1
            total_seconds += row["duration"]
            pbar.update(1)
            pbar.set_postfix(kept=counts.get("written", 0), hrs=f"{total_seconds/3600:.1f}")
    finally:
        manifest_out.close()
        pbar.close()

    # stats summary
    stats = {
        "filter_thresholds": {
            "DURATION_MIN": DURATION_MIN, "DURATION_MAX": DURATION_MAX,
            "SNR_MIN_READ": SNR_MIN_READ, "SNR_MIN_EXTEMPORE": SNR_MIN_EXTEMPORE,
            "CER_MAX_EXTEMPORE": CER_MAX_EXTEMPORE,
        },
        "target_sr": TARGET_SR,
        "counts": dict(counts),
        "kept_speakers": len(per_speaker_idx),
        "kept_hours": round(total_seconds / 3600, 2),
        "dry_run": args.dry_run,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n[prep] stats → {STATS_PATH}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
