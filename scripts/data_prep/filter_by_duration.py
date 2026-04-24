"""Filter train_list.txt to utterances <= max-seconds, regenerating from the
`.full.txt` backup. Keeps the full manifest preserved for future re-filters.

Usage:
    python3 /workspace/bol_run/scripts/filter_by_duration.py --max-seconds 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument(
        "--training-dir",
        default="/workspace/bol_run/training",
        help="dir with train_list.txt and train_list.full.txt",
    )
    ap.add_argument(
        "--audio-dir",
        default="/workspace/bol_run/dataset/audio",
    )
    args = ap.parse_args()

    training = Path(args.training_dir)
    audio = Path(args.audio_dir)

    src = training / "train_list.txt"
    backup = training / "train_list.full.txt"

    if not backup.exists():
        # first run: snapshot current train_list.txt as the full backup
        if not src.exists():
            print(f"[fail] neither {src} nor {backup} exists", file=sys.stderr)
            return 2
        backup.write_text(src.read_text())
        print(f"[init] snapshotted {src.name} → {backup.name}")

    target = int(args.max_seconds * 24000)
    out: list[str] = []
    dropped = 0
    missing = 0
    for line in backup.read_text().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        wav = audio / parts[0]
        try:
            info = sf.info(str(wav))
        except Exception:
            missing += 1
            continue
        if info.frames <= target:
            out.append(line)
        else:
            dropped += 1

    src.write_text("\n".join(out) + "\n")
    print(f"[ok] max_seconds={args.max_seconds}  kept={len(out)}  dropped={dropped}  missing_wav={missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
