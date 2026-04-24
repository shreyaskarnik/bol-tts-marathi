"""prepare_rasa_mr.py — adapt our existing Rasa Marathi data to the
bol-tts-marathi training layout.

Reads:
  ${BOL_ROOT}/data/train_data.txt
  ${BOL_ROOT}/data/val_data.txt
  ${BOL_ROOT}/data/wavs/marathi_*.wav

Writes:
  bol-tts-marathi/dataset/audio/rasa/marathi_*.wav      (shutil.copy2)
  bol-tts-marathi/training/rasa_mr.txt                   (rasa/...|ipa|name)

Speaker-id remap: 2 → marathi_female, 3 → marathi_male (1/0 dropped).
IPA cleanup: U+00A0 (NBSP) → U+0020. No other changes.
Idempotent: wav copies skipped if destination exists; manifest overwritten.
"""
from __future__ import annotations

import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
# Script lives at scripts/data_prep/prepare_rasa_mr.py, so parents[2] = repo root.
# Override via BOL_REPO env var if you run from a different layout.
import os
REPO_ROOT = Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[2]))
# Source audio root: where the unprocessed Rasa WAVs live before prep.
# Override via BOL_RASA_SRC env var, else falls back to repo-local data/.
SRC_DATA = Path(os.environ.get("BOL_RASA_SRC", REPO_ROOT / "data"))
SRC_WAVS = SRC_DATA / "wavs"
SRC_TRAIN = SRC_DATA / "train_data.txt"
SRC_VAL = SRC_DATA / "val_data.txt"

DST_WAVS = REPO_ROOT / "dataset" / "audio" / "rasa"
DST_MANIFEST = REPO_ROOT / "training" / "rasa_mr.txt"
TRAINING_DIR = REPO_ROOT / "training"

SPEAKER_MAP = {"2": "marathi_female", "3": "marathi_male"}


def load_marathi_lines(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (wav_rel, ipa, speaker_id) for Marathi rows (sid 2 or 3)."""
    rows: list[tuple[str, str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or "|" not in raw:
            continue
        parts = raw.split("|")
        if len(parts) != 3:
            continue
        wav_rel, ipa, sid = parts
        if sid not in SPEAKER_MAP:
            continue
        rows.append((wav_rel, ipa, sid))
    return rows


def copy_wavs(rows: list[tuple[str, str, str]]) -> int:
    """Copy unique wav files with shutil.copy2. Skip if dest already exists.
    Returns number of files newly copied."""
    DST_WAVS.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    unique_wav_names: list[str] = []
    for wav_rel, _, _ in rows:
        name = Path(wav_rel).name
        if name in seen:
            continue
        seen.add(name)
        unique_wav_names.append(name)

    copied = 0
    skipped = 0
    progress_every = 500
    for i, name in enumerate(tqdm(unique_wav_names, desc="copy wavs", unit="wav"), 1):
        src = SRC_WAVS / name
        dst = DST_WAVS / name
        if dst.exists():
            skipped += 1
        else:
            if not src.exists():
                print(f"[WARN] missing source wav: {src}", file=sys.stderr)
                continue
            shutil.copy2(src, dst)
            copied += 1
        if i % progress_every == 0:
            print(f"  [{i}/{len(unique_wav_names)}] copied_new={copied} skipped_existing={skipped}")
    print(f"[copy] total_unique={len(unique_wav_names)} copied_new={copied} skipped_existing={skipped}")
    return copied


def rewrite_manifest(rows: list[tuple[str, str, str]]) -> dict:
    """Write rasa_mr.txt and return stats dict."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    nbsp_lines = 0
    per_speaker: Counter[str] = Counter()
    total_ipa_chars = 0

    out_lines: list[str] = []
    for wav_rel, ipa, sid in rows:
        had_nbsp = "\xa0" in ipa
        if had_nbsp:
            ipa = ipa.replace("\xa0", " ")
            nbsp_lines += 1
        name = Path(wav_rel).name
        new_wav_path = f"rasa/{name}"
        speaker_name = SPEAKER_MAP[sid]
        assert ipa == ipa.strip(), f"leading/trailing whitespace after clean: {ipa!r}"
        assert ipa != "", f"empty ipa for {name}"
        out_lines.append(f"{new_wav_path}|{ipa}|{speaker_name}")
        per_speaker[speaker_name] += 1
        total_ipa_chars += len(ipa)

    DST_MANIFEST.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {
        "total": len(out_lines),
        "per_speaker": dict(per_speaker),
        "total_ipa_chars": total_ipa_chars,
        "nbsp_cleaned_lines": nbsp_lines,
    }


def verify_wav_paths_exist(rows: list[tuple[str, str, str]]) -> None:
    missing: list[str] = []
    for wav_rel, _, _ in rows:
        name = Path(wav_rel).name
        if not (DST_WAVS / name).exists():
            missing.append(name)
    assert not missing, f"{len(missing)} referenced wavs missing in {DST_WAVS}, e.g. {missing[:3]}"


def verify_tokenizer_coverage(manifest_path: Path) -> None:
    """Import training/kokoro_symbols.py, run TextCleaner on 10 random lines,
    assert every char maps to a vocab index."""
    sys.path.insert(0, str(TRAINING_DIR))
    from kokoro_symbols import TextCleaner, dicts  # type: ignore

    cleaner = TextCleaner()
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    rng = random.Random(42)
    sample = rng.sample(lines, k=min(10, len(lines)))

    print("\n[verify] tokenizer coverage on 10 random lines:")
    all_covered = True
    for line in sample:
        ipa = line.split("|", 2)[1]
        ids = cleaner(ipa)
        missing_chars = [c for c in ipa if c not in dicts]
        coverage = len(ids) / len(ipa) if ipa else 1.0
        status = "OK " if not missing_chars else "FAIL"
        print(f"  [{status}] len={len(ipa)} ids={len(ids)} cov={coverage*100:.1f}% "
              f"missing={sorted(set(missing_chars))}")
        if missing_chars:
            all_covered = False

    assert all_covered, "tokenizer coverage < 100% on sample"

    # Also scan ENTIRE manifest for any uncovered char (stronger guarantee).
    unknown: Counter[str] = Counter()
    total_chars = 0
    for line in lines:
        ipa = line.split("|", 2)[1]
        total_chars += len(ipa)
        for c in ipa:
            if c not in dicts:
                unknown[c] += 1
    if unknown:
        print(f"[verify] FULL-MANIFEST uncovered chars: {dict(unknown)}")
    else:
        print(f"[verify] FULL-MANIFEST coverage 100% ({total_chars:,} chars across {len(lines):,} lines)")
    assert not unknown, f"uncovered chars in full manifest: {dict(unknown)}"


def main() -> int:
    print(f"[prep] src_data = {SRC_DATA}")
    print(f"[prep] dst_wavs = {DST_WAVS}")
    print(f"[prep] dst_manifest = {DST_MANIFEST}")

    train_rows = load_marathi_lines(SRC_TRAIN)
    val_rows = load_marathi_lines(SRC_VAL)
    print(f"[prep] train marathi rows: {len(train_rows)}  "
          f"val marathi rows: {len(val_rows)}  total: {len(train_rows) + len(val_rows)}")

    all_rows = train_rows + val_rows

    copy_wavs(all_rows)
    stats = rewrite_manifest(all_rows)
    verify_wav_paths_exist(all_rows)
    verify_tokenizer_coverage(DST_MANIFEST)

    print("\n[prep] === stats ===")
    print(f"  total lines kept:       {stats['total']}")
    print(f"  per-speaker counts:     {stats['per_speaker']}")
    print(f"  total IPA char count:   {stats['total_ipa_chars']:,}")
    print(f"  lines w/ NBSP cleaned:  {stats['nbsp_cleaned_lines']}")
    print(f"  manifest path:          {DST_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
