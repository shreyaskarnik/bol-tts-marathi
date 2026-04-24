"""Stream `SPRINGLab/IndicTTS_Marathi` from HuggingFace, split by gender,
resample to 24 kHz, and write to per-gender audio dirs for voicepack
extraction.

Only needs ~300 clips per gender for a clean mean-style voicepack; we default
to 500 each (under 5 min of download on a decent connection, ~500 MB total).

Output layout (relative to --repo-root):

    dataset/audio/indictts_mr_female/indictts_f_00000.wav ...
    dataset/audio/indictts_mr_male/indictts_m_00000.wav ...
    training/indictts_mr_stats.json

Usage:

    # default: 500 per gender, 24 kHz
    python scripts/data_prep/prepare_indictts_marathi.py

    # smaller smoke run
    python scripts/data_prep/prepare_indictts_marathi.py --per-gender 50

    # verify gender labels by writing `_sample_female.wav` + `_sample_male.wav`
    # at the top of each dir — spot-check via `afplay` before committing
    python scripts/data_prep/prepare_indictts_marathi.py --probe

Next step after this script: see `docs/VOICEPACKS.md` for `extract_voicepacks_mr.py`.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SR = 24_000
DATASET_ID = "SPRINGLab/IndicTTS_Marathi"


def _resample_to_24k(arr: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample a 1-D float32 audio array from src_sr to 24 kHz."""
    if src_sr == TARGET_SR:
        return arr.astype(np.float32, copy=False)
    # librosa handles arbitrary source rates cleanly
    import librosa
    return librosa.resample(arr.astype(np.float32), orig_sr=src_sr, target_sr=TARGET_SR).astype(
        np.float32
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[2])),
    )
    ap.add_argument(
        "--per-gender",
        type=int,
        default=500,
        help="max clips per gender (voicepack extraction only needs ~200)",
    )
    ap.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="skip clips shorter than this (s) — style encoder needs enough frames",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="also save _sample_female.wav + _sample_male.wav for manual gender verification",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    female_dir = repo_root / "dataset" / "audio" / "indictts_mr_female"
    male_dir = repo_root / "dataset" / "audio" / "indictts_mr_male"
    stats_path = repo_root / "training" / "indictts_mr_stats.json"
    female_dir.mkdir(parents=True, exist_ok=True)
    male_dir.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # HF gender convention: 0 = female, 1 = male (we verify via --probe)
    GENDER_FEMALE = 0
    GENDER_MALE = 1

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "need `datasets`: uv pip install datasets librosa"
        ) from e

    print(f"streaming {DATASET_ID} (train split)…")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    female_count = 0
    male_count = 0
    skipped_short = 0
    per_gender_target = args.per_gender
    manifest_rows: list[dict] = []

    for i, row in enumerate(ds):
        if female_count >= per_gender_target and male_count >= per_gender_target:
            break

        gender = int(row["gender"])
        if gender == GENDER_FEMALE and female_count < per_gender_target:
            out_dir = female_dir
            idx = female_count
            prefix = "indictts_f"
        elif gender == GENDER_MALE and male_count < per_gender_target:
            out_dir = male_dir
            idx = male_count
            prefix = "indictts_m"
        else:
            continue

        audio = row["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        src_sr = int(audio["sampling_rate"])
        dur = len(arr) / src_sr
        if dur < args.min_duration:
            skipped_short += 1
            continue

        arr24 = _resample_to_24k(arr, src_sr)
        wav_name = f"{prefix}_{idx:05d}.wav"
        sf.write(out_dir / wav_name, arr24, TARGET_SR)

        manifest_rows.append(
            {
                "path": str((out_dir / wav_name).relative_to(repo_root)),
                "text": row.get("text", ""),
                "gender": "female" if gender == GENDER_FEMALE else "male",
                "duration": round(len(arr24) / TARGET_SR, 2),
            }
        )

        if gender == GENDER_FEMALE:
            female_count += 1
        else:
            male_count += 1

        if (female_count + male_count) % 50 == 0:
            print(f"  female={female_count}  male={male_count}  skipped_short={skipped_short}")

    print(
        f"done. female={female_count}  male={male_count}  skipped_short={skipped_short}  "
        f"total written={female_count + male_count}"
    )

    # Save stats/manifest
    stats = {
        "dataset": DATASET_ID,
        "target_sample_rate": TARGET_SR,
        "female_clips": female_count,
        "male_clips": male_count,
        "skipped_short": skipped_short,
        "min_duration_sec": args.min_duration,
        "female_dir": str(female_dir.relative_to(repo_root)),
        "male_dir": str(male_dir.relative_to(repo_root)),
        "clips": manifest_rows,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote stats: {stats_path}")

    if args.probe:
        # Copy first clip of each gender to a clearly-named sample for manual listening
        for gender_dir, out_name in [
            (female_dir, "_sample_female.wav"),
            (male_dir, "_sample_male.wav"),
        ]:
            clips = sorted(gender_dir.glob("*.wav"))
            if clips:
                sample = gender_dir / out_name
                sample.write_bytes(clips[0].read_bytes())
                print(f"probe sample: afplay {sample}")

    print()
    print("next steps:")
    print(f"  # extract mf_mukta voicepack (female)")
    print(
        f"  python scripts/upstream/extract_voicepack.py \\\n"
        f"    --model       <stage2-ckpt.pth> \\\n"
        f"    --style-encoder-model <stage1-ckpt.pth> \\\n"
        f"    --audio-dir   {female_dir} \\\n"
        f"    --output      checkpoints/voices/mf_mukta.pt"
    )
    print()
    print(f"  # extract mm_dnyanesh voicepack (male)")
    print(
        f"  python scripts/upstream/extract_voicepack.py \\\n"
        f"    --model       <stage2-ckpt.pth> \\\n"
        f"    --style-encoder-model <stage1-ckpt.pth> \\\n"
        f"    --audio-dir   {male_dir} \\\n"
        f"    --output      checkpoints/voices/mm_dnyanesh.pt"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
