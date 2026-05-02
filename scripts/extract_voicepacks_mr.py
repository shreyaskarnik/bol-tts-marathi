#!/usr/bin/env python3
"""extract_voicepacks_mr.py — extract all 4 Marathi voicepacks after Stage 2.

Four named Marathi voicepacks (see docs/VOICEPACKS.md):
    mf_asha      → Rasa marathi_female     (auto-picked clips from rasa/)
    mm_vivek     → Rasa marathi_male       (auto-picked clips from rasa/)
    mf_mukta     → IV-R mr_sXXXX (female)  (user picks via pick_ivr_voices.py)
    mm_dnyanesh  → IV-R mr_sYYYY (male)    (user picks via pick_ivr_voices.py)

Extraction wraps semidark/kokoro-deutsch's `extract_voicepack.py`,
vendored verbatim at `scripts/upstream/extract_voicepack.py` (see NOTICE).
Override via --upstream-script.

Output: four .pt files of shape [510, 1, 256] (float32), one per voice, into
the --output-dir (default /workspace/bol_run/voices/).

Typical pod invocation (after Stage 2 finishes):

    python scripts/extract_voicepacks_mr.py \
        --checkpoint /workspace/bol_run/StyleTTS2/logs/kokoro-marathi/epoch_2nd_00009.pth \
        --style-encoder-checkpoint /workspace/bol_run/StyleTTS2/logs/kokoro-marathi/epoch_1st_00002.pth \
        --mukta-speaker mr_s1418 \
        --dnyanesh-speaker mr_s3df3 \
        --output-dir /workspace/bol_run/voices/

The script:
  1. Validates the checkpoint exists.
  2. For each voice, collects a list of reference WAVs:
        - Rasa voices: auto-filters rasa/marathi_{female,male}_*.wav by duration
          (4-10s) and picks the first N clean clips (seeded shuffle for diversity).
        - IV-R voices: globs indicvoices_r/{speaker}_*.wav.
     All picked wavs are staged (symlinked) into a per-voice temp directory so
     upstream extract_voicepack() gets one clean --audio-dir per voice.
  3. Calls upstream extract_voicepack() to produce voices/<id>.pt.

Safe to re-run: temp staging dirs are cleaned up on exit.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import shutil
import sys
import tempfile
from pathlib import Path

# Default reference-audio sources (pod-side). Override via CLI if paths differ.
DEFAULT_RASA_DIR = Path("/workspace/bol_run/dataset/audio/rasa")
DEFAULT_IVR_DIR = Path("/workspace/bol_run/dataset/audio/indicvoices_r")
DEFAULT_SPRINGLAB_DIR = Path("/workspace/bol_run/dataset/audio/springlab_mr")

# Where the upstream semidark script lives on the pod (same layout as our
# The upstream extract_voicepack.py is vendored in ./scripts/upstream/.
DEFAULT_UPSTREAM_SCRIPT = Path(
    str(Path(__file__).resolve().parent / "upstream" / "extract_voicepack.py")
)

# Reference-clip selection knobs.
REF_CLIP_MIN_SEC = 4.0
REF_CLIP_MAX_SEC = 10.0
REF_CLIPS_PER_VOICE_RASA = 40  # upstream samples 200 by default, but Rasa has
#                                 thousands of clips per speaker, so 40 diverse
#                                 in-range clips is plenty and much faster.
REF_CLIPS_PER_VOICE_IVR = 60   # IV-R has <120 clips/speaker; take up to 60.
REF_CLIPS_PER_VOICE_SPRINGLAB = 40  # SPRINGLab has ~5K clips/speaker, glob+filter same as Rasa.

VOICES = [
    # (voice_id, source, selector_info_for_voice)
    # source is "rasa" / "ivr" / "springlab"; for rasa+springlab the selector is
    # the speaker-name prefix; for ivr it's filled from CLI.
    {"id": "mf_asha",      "source": "rasa",      "prefix": "marathi_female"},
    {"id": "mm_vivek",     "source": "rasa",      "prefix": "marathi_male"},
    {"id": "mf_mukta",     "source": "ivr",       "prefix": None},   # filled from --mukta-speaker
    {"id": "mm_dnyanesh",  "source": "ivr",       "prefix": None},   # filled from --dnyanesh-speaker
    {"id": "mf_priya",     "source": "springlab", "prefix": "springlab_female"},
    {"id": "mm_arjun",     "source": "springlab", "prefix": "springlab_male"},
]


def load_upstream(script_path: Path):
    """Import the upstream extract_voicepack module from a file path."""
    if not script_path.exists():
        sys.exit(
            f"ERROR: upstream extract_voicepack.py not found at {script_path}.\n"
            f"Pass --upstream-script to point at a different copy of extract_voicepack.py."
        )
    spec = importlib.util.spec_from_file_location("upstream_extract_voicepack", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "extract_voicepack"):
        sys.exit(
            f"ERROR: {script_path} does not define extract_voicepack(); "
            f"upstream layout may have changed."
        )
    return module


def duration_sec(wav_path: Path) -> float:
    """Return duration in seconds via soundfile.info (cheap, no decode)."""
    import soundfile as sf
    info = sf.info(str(wav_path))
    return info.frames / float(info.samplerate)


def collect_rasa_refs(rasa_dir: Path, prefix: str, n: int, rng: random.Random) -> list[Path]:
    """Pick up to `n` Rasa clips whose duration is in [REF_CLIP_MIN_SEC, REF_CLIP_MAX_SEC]."""
    all_clips = sorted(rasa_dir.glob(f"{prefix}_*.wav"))
    if not all_clips:
        sys.exit(f"ERROR: no clips found matching {rasa_dir}/{prefix}_*.wav")
    rng.shuffle(all_clips)
    picked: list[Path] = []
    for p in all_clips:
        try:
            d = duration_sec(p)
        except Exception:
            continue
        if REF_CLIP_MIN_SEC <= d <= REF_CLIP_MAX_SEC:
            picked.append(p)
            if len(picked) >= n:
                break
    if len(picked) < max(10, n // 4):
        print(
            f"  WARNING: only {len(picked)} in-range clips for prefix={prefix} "
            f"(duration filter {REF_CLIP_MIN_SEC}-{REF_CLIP_MAX_SEC}s). "
            f"Extraction will still proceed."
        )
    return picked


def collect_ivr_refs(ivr_dir: Path, speaker: str, n: int) -> list[Path]:
    """Grab all clips for an IV-R speaker. No duration filter — these are
    already filtered to 2-15s by prepare_indicvoices_r_mr.py."""
    all_clips = sorted(ivr_dir.glob(f"{speaker}_*.wav"))
    if not all_clips:
        sys.exit(
            f"ERROR: no clips found for IV-R speaker {speaker} in {ivr_dir}. "
            f"Did you pass the right --{'mukta' if 'mukta' in speaker else 'dnyanesh'}-speaker id?"
        )
    return all_clips[:n]


def stage_refs(wavs: list[Path], stage_dir: Path) -> None:
    """Symlink wavs into stage_dir so upstream script sees a clean --audio-dir."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for src in wavs:
        dst = stage_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # Fallback for filesystems without symlink support.
            shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract all 4 Marathi voicepacks from a Stage 2 checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Path to Stage 2 .pth checkpoint (e.g. epoch_2nd_00009.pth).")
    ap.add_argument("--style-encoder-checkpoint", type=Path, default=None,
                    help="Optional Stage 1 checkpoint to pull style_encoder weights from. "
                         "Recommended (upstream notes Stage 2 can degrade style_encoder).")
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/bol_run/voices"),
                    help="Directory to write voicepack .pt files (default: /workspace/bol_run/voices/).")
    ap.add_argument("--rasa-dir", type=Path, default=DEFAULT_RASA_DIR,
                    help=f"Rasa audio dir (default: {DEFAULT_RASA_DIR}).")
    ap.add_argument("--ivr-dir", type=Path, default=DEFAULT_IVR_DIR,
                    help=f"IndicVoices-R audio dir (default: {DEFAULT_IVR_DIR}).")
    ap.add_argument("--springlab-dir", type=Path, default=DEFAULT_SPRINGLAB_DIR,
                    help=f"SPRINGLab audio dir (default: {DEFAULT_SPRINGLAB_DIR}).")
    ap.add_argument("--mukta-speaker", required=True,
                    help="IV-R speaker id for mf_mukta (e.g. mr_s1418). "
                         "Use pick_ivr_voices.py to pick.")
    ap.add_argument("--dnyanesh-speaker", required=True,
                    help="IV-R speaker id for mm_dnyanesh (e.g. mr_s3df3). "
                         "Use pick_ivr_voices.py to pick.")
    ap.add_argument("--upstream-script", type=Path, default=DEFAULT_UPSTREAM_SCRIPT,
                    help=f"Path to semidark's scripts/extract_voicepack.py "
                         f"(default: {DEFAULT_UPSTREAM_SCRIPT}).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Device for mel + encoders (default: auto).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for Rasa clip selection (default: 42).")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Voice IDs to skip (e.g. --skip mf_mukta mm_dnyanesh to do Rasa only).")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    if args.style_encoder_checkpoint and not args.style_encoder_checkpoint.exists():
        sys.exit(f"ERROR: style-encoder-checkpoint not found: {args.style_encoder_checkpoint}")

    upstream = load_upstream(args.upstream_script)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fill in IV-R speaker IDs on the voice specs.
    voices = [dict(v) for v in VOICES]
    for v in voices:
        if v["id"] == "mf_mukta":
            v["prefix"] = args.mukta_speaker
        elif v["id"] == "mm_dnyanesh":
            v["prefix"] = args.dnyanesh_speaker

    rng = random.Random(args.seed)

    with tempfile.TemporaryDirectory(prefix="voicepack_stage_") as tmpdir:
        tmp_root = Path(tmpdir)

        for v in voices:
            vid = v["id"]
            if vid in args.skip:
                print(f"\n[skip] {vid}")
                continue

            print(f"\n=== Voice: {vid}  (source={v['source']}, prefix={v['prefix']}) ===")
            stage_dir = tmp_root / vid
            if v["source"] == "rasa":
                refs = collect_rasa_refs(args.rasa_dir, v["prefix"], REF_CLIPS_PER_VOICE_RASA, rng)
            elif v["source"] == "springlab":
                # SPRINGLab uses same prefix-glob+duration-filter as Rasa.
                refs = collect_rasa_refs(args.springlab_dir, v["prefix"], REF_CLIPS_PER_VOICE_SPRINGLAB, rng)
            else:
                refs = collect_ivr_refs(args.ivr_dir, v["prefix"], REF_CLIPS_PER_VOICE_IVR)
            print(f"  picked {len(refs)} reference clips -> staged at {stage_dir}")
            stage_refs(refs, stage_dir)

            out_path = args.output_dir / f"{vid}.pt"
            print(f"  extracting -> {out_path}")
            upstream.extract_voicepack(
                model_path=str(args.checkpoint),
                audio_dir=str(stage_dir),
                output_path=str(out_path),
                num_samples=len(refs),   # use all staged clips
                device=args.device,
                style_encoder_model=(
                    str(args.style_encoder_checkpoint)
                    if args.style_encoder_checkpoint else None
                ),
            )

    print("\nAll voicepacks extracted:")
    for v in voices:
        if v["id"] in args.skip:
            continue
        p = args.output_dir / f"{v['id']}.pt"
        status = f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "MISSING"
        print(f"  {p}  {status}")


if __name__ == "__main__":
    main()
