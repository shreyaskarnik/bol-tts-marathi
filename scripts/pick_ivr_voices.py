#!/usr/bin/env python3
"""pick_ivr_voices.py — help pick IV-R speakers for mf_mukta and mm_dnyanesh.

We didn't store per-speaker gender when we built indicvoices_r_mr.txt (we
hashed speaker IDs to `mr_sXXXX` via short_speaker()). So the workflow is:

    1. Run this script (pod-side) to list the top-N IV-R speakers by utt count,
       along with 3 sample WAV paths per speaker.
    2. Listen to the samples (scp a few back to your Mac, or just `afplay` / `ffplay`).
    3. Pick one female-sounding speaker (→ mf_mukta) and one male-sounding (→ mm_dnyanesh).
    4. Pass those IDs to extract_voicepacks_mr.py:
         --mukta-speaker mr_sXXXX --dnyanesh-speaker mr_sYYYY

Usage:

    python scripts/pick_ivr_voices.py \
        --manifest /workspace/bol_run/training/indicvoices_r_mr.txt \
        --audio-dir /workspace/bol_run/dataset/audio/indicvoices_r \
        --top 20

Output: prints a ranked table (speaker_id, n_utts, sample_wav_1, sample_wav_2, sample_wav_3).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/workspace/bol_run/training/indicvoices_r_mr.txt"),
        help="Path to indicvoices_r_mr.txt manifest (format: relpath|ipa|speaker).",
    )
    ap.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("/workspace/bol_run/dataset/audio/indicvoices_r"),
        help="Directory holding IV-R wavs (relpaths in the manifest resolve here's parent).",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/bol_run/dataset/audio"),
        help="Root the manifest's first column is relative to (default: dataset/audio).",
    )
    ap.add_argument("--top", type=int, default=20,
                    help="Show top-N speakers by utterance count (default: 20).")
    ap.add_argument("--samples-per-speaker", type=int, default=3,
                    help="Number of sample wav paths to print per speaker (default: 3).")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"ERROR: manifest not found: {args.manifest}")

    # speaker -> list of relpaths (preserve manifest order)
    clips_by_speaker: dict[str, list[str]] = defaultdict(list)
    with args.manifest.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            relpath, _ipa, speaker = parts[0], parts[1], parts[2]
            # Skip Rasa rows just in case the manifest is the merged one.
            if not speaker.startswith("mr_s"):
                continue
            clips_by_speaker[speaker].append(relpath)

    counts = Counter({sp: len(clips) for sp, clips in clips_by_speaker.items()})
    total_speakers = len(counts)
    total_utts = sum(counts.values())
    print(f"# IV-R manifest: {args.manifest}")
    print(f"# Total IV-R speakers: {total_speakers}   total IV-R utterances: {total_utts}")
    print(f"# Showing top {args.top} by utt count. "
          f"Listen to sample clips, then pick one female + one male.\n")

    header = f"{'rank':>4}  {'speaker':<10}  {'n_utts':>7}  samples"
    print(header)
    print("-" * len(header))

    for rank, (speaker, n) in enumerate(counts.most_common(args.top), start=1):
        clips = clips_by_speaker[speaker][: args.samples_per_speaker]
        resolved = [str(args.dataset_root / c) for c in clips]
        print(f"{rank:>4}  {speaker:<10}  {n:>7}  " + "  ".join(resolved))

    print()
    print("# Quick listen one-liner (pod):")
    print("#   aplay <sample_wav>   # or ffplay -nodisp -autoexit")
    print("# Or scp to your Mac and use afplay / QuickTime.")
    print()
    print("# Once chosen, run:")
    print("#   python scripts/extract_voicepacks_mr.py \\")
    print("#       --checkpoint <path-to-stage2.pth> \\")
    print("#       --style-encoder-checkpoint <path-to-stage1.pth> \\")
    print("#       --mukta-speaker mr_sXXXX \\")
    print("#       --dnyanesh-speaker mr_sYYYY")


if __name__ == "__main__":
    main()
