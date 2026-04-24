#!/usr/bin/env python3
"""
Build the two RunPod upload tarballs for the Marathi Kokoro-82M fine-tune.

Bundle 1: bol_training_v5.tar.gz  — code, configs, patched StyleTTS2, kokoro_base.pth
Bundle 2: bol_data_v2.tar.gz      — manifests + ~25,810 WAVs (~6.7 GB)

Idempotent: deletes existing output tarballs before rebuild.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Script lives at scripts/, so parents[1] = repo root.
PROJ = Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[1]))
STYLETTS2_SRC = PROJ / "StyleTTS2"

UPLOAD = PROJ / "upload"
TRAINING_BUNDLE = UPLOAD / "bol_training_v5.tar.gz"
DATA_BUNDLE = UPLOAD / "bol_data_v2.tar.gz"
CHECKSUMS = UPLOAD / "CHECKSUMS.txt"

TRAINING_ROOT = "bol_training_v5"
DATA_ROOT = "bol_data_v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def git_head(path: Path) -> str:
    """Return rev-parse HEAD of path's git repo. StyleTTS2 is a submodule (`.git`
    is a gitfile), so we use `git -C` which resolves it correctly."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception as exc:  # pragma: no cover
        return f"<git rev-parse failed: {exc}>"


def styletts2_exclude(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Filter for tar.add(): drop .git, __pycache__, *.pyc."""
    name = tarinfo.name
    parts = name.split("/")
    if ".git" in parts:
        return None
    if "__pycache__" in parts:
        return None
    if name.endswith(".pyc"):
        return None
    return tarinfo


def scripts_exclude(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Drop logs/ subdir and __pycache__ from scripts/."""
    name = tarinfo.name
    parts = name.split("/")
    if "logs" in parts:
        return None
    if "__pycache__" in parts:
        return None
    if name.endswith(".pyc"):
        return None
    return tarinfo


# ---------------------------------------------------------------------------
# Bundle 1
# ---------------------------------------------------------------------------


def build_training_bundle() -> None:
    print("=" * 70)
    print(f"Building Bundle 1: {TRAINING_BUNDLE.name}")
    print("=" * 70)

    if TRAINING_BUNDLE.exists():
        print(f"  removing existing {TRAINING_BUNDLE}")
        TRAINING_BUNDLE.unlink()

    # Pre-compute things we need for VERSION.txt
    kokoro_base = PROJ / "training" / "kokoro_base.pth"
    assert kokoro_base.is_file(), f"missing: {kokoro_base}"
    kb_size = kokoro_base.stat().st_size
    print(f"  computing sha256 of kokoro_base.pth ({human_bytes(kb_size)}) ...")
    kb_sha = sha256_of(kokoro_base)
    print(f"    sha256 = {kb_sha}")

    styletts2_commit = git_head(STYLETTS2_SRC)
    print(f"  StyleTTS2 HEAD: {styletts2_commit}")

    version_txt = (
        f"Bol Training Bundle v5 — Marathi Kokoro-82M fine-tune\n"
        f"Generated: {utc_now_iso()}\n"
        f"\n"
        f"StyleTTS2:\n"
        f"  fork: semidark/StyleTTS2 (main branch)\n"
        f"  commit: {styletts2_commit}\n"
        f"  Marathi kokoro_symbols.py: installed at StyleTTS2/kokoro_symbols.py (ɭ at index 144)\n"
        f"\n"
        f"kokoro_base.pth:\n"
        f"  size: {kb_size} bytes ({human_bytes(kb_size)})\n"
        f"  sha256: {kb_sha}\n"
        f"  source: hexgrad/Kokoro-82M kokoro-v1_0.pth converted via scripts/convert_kokoro_weights.py\n"
        f"  modules: bert, bert_encoder, predictor, decoder, text_encoder\n"
        f"  total params: 81.76M\n"
        f"\n"
        f"Config: configs/config_marathi_ft.yml (forked from semidark config_german_ft.yml; only log_dir changed)\n"
        f"\n"
        f"Data (in companion bundle bol_data_v2):\n"
        f"  train: 24,676 utterances\n"
        f"  val:   1,134 utterances\n"
        f"  speakers: 331 (329 IV-R + marathi_female + marathi_male)\n"
        f"  total hours: ~40h\n"
    )

    # Check for optional launch_training.sh
    launcher_src = PROJ / "scripts" / "launch_training.sh"
    has_launcher = launcher_src.is_file()
    if not has_launcher:
        print(
            "  WARNING: scripts/launch_training.sh is MISSING — will omit from bundle."
        )

    print(f"  writing {TRAINING_BUNDLE} ...")
    with tarfile.open(TRAINING_BUNDLE, "w:gz") as tar:

        def add(arcname: str, src: Path, filt=None) -> None:
            tar.add(str(src), arcname=f"{TRAINING_ROOT}/{arcname}", filter=filt)

        # VERSION.txt — write in-memory file
        import io

        data = version_txt.encode("utf-8")
        info = tarfile.TarInfo(name=f"{TRAINING_ROOT}/VERSION.txt")
        info.size = len(data)
        info.mtime = int(_dt.datetime.now().timestamp())
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))

        # StyleTTS2/ (patched fork)
        print("    + StyleTTS2/ (excluding .git, __pycache__, *.pyc)")
        add("StyleTTS2", STYLETTS2_SRC, filt=styletts2_exclude)

        # training/ files
        print("    + training/kokoro_base.pth")
        add("training/kokoro_base.pth", PROJ / "training" / "kokoro_base.pth")
        print("    + training/kokoro_symbols.py")
        add("training/kokoro_symbols.py", PROJ / "training" / "kokoro_symbols.py")
        print("    + training/OOD_texts.txt")
        add("training/OOD_texts.txt", PROJ / "training" / "OOD_texts.txt")
        print("    + training/config.json")
        add("training/config.json", PROJ / "training" / "config.json")

        # configs/
        print("    + configs/config_marathi_ft.yml")
        add(
            "configs/config_marathi_ft.yml",
            PROJ / "configs" / "config_marathi_ft.yml",
        )

        # scripts/ — everything, excluding logs/, __pycache__
        print("    + scripts/ (excluding logs/, __pycache__)")
        add("scripts", PROJ / "scripts", filt=scripts_exclude)


    size = TRAINING_BUNDLE.stat().st_size
    print(f"  done. {TRAINING_BUNDLE.name} = {human_bytes(size)} ({size} bytes)")


# ---------------------------------------------------------------------------
# Bundle 2
# ---------------------------------------------------------------------------


def build_data_bundle() -> None:
    print("=" * 70)
    print(f"Building Bundle 2: {DATA_BUNDLE.name}")
    print("=" * 70)

    if DATA_BUNDLE.exists():
        print(f"  removing existing {DATA_BUNDLE}")
        DATA_BUNDLE.unlink()

    # Spot-check manifest line counts for MANIFEST.txt
    train_list = PROJ / "training" / "train_list.txt"
    val_list = PROJ / "training" / "val_list.txt"
    rasa_txt = PROJ / "training" / "rasa_mr.txt"
    ivr_txt = PROJ / "training" / "indicvoices_r_mr.txt"

    train_n = sum(1 for _ in train_list.open())
    val_n = sum(1 for _ in val_list.open())
    rasa_n = sum(1 for _ in rasa_txt.open())
    ivr_n = sum(1 for _ in ivr_txt.open())

    print(
        f"  manifest line counts: train={train_n}, val={val_n}, rasa={rasa_n}, iv_r={ivr_n}"
    )

    rasa_wavs = sorted((PROJ / "dataset" / "audio" / "rasa").glob("*.wav"))
    ivr_wavs = sorted((PROJ / "dataset" / "audio" / "indicvoices_r").glob("*.wav"))
    print(f"  wav counts: rasa={len(rasa_wavs)}, indicvoices_r={len(ivr_wavs)}")

    manifest_txt = (
        f"Bol Data Bundle v2 — Marathi (Rasa + IndicVoices-R filtered)\n"
        f"Generated: {utc_now_iso()}\n"
        f"\n"
        f"Sources:\n"
        f"  rasa/        {len(rasa_wavs)} WAVs from ai4bharat/Rasa Marathi subset (studio, 2 speakers)\n"
        f"  indicvoices_r/ {len(ivr_wavs)} WAVs from ai4bharat/indicvoices_r Marathi filtered (SNR>55 Extempore, SNR>45 Read, duration 2-15s, CER<0.05 Extempore)\n"
        f"\n"
        f"Manifests:\n"
        f"  train_list.txt: {train_n} entries\n"
        f"  val_list.txt:   {val_n} entries (4.4% val; stratified per speaker)\n"
        f"\n"
        f"Format: wav_path|ipa_string|speaker_name (pipe-delimited)\n"
        f'  wav_path is relative to dataset/audio/ (e.g., "rasa/marathi_female_00123.wav")\n'
        f"  ipa_string uses Kokoro's 178-token vocab (via misaki and espeak-ng)\n"
        f'  speaker_name is descriptive (e.g., "marathi_female", "mr_s1418")\n'
        f"\n"
        f"Audio spec: 24 kHz mono, 16-bit PCM WAV\n"
    )

    print(f"  writing {DATA_BUNDLE} (large, ~5-10 min) ...")
    with tarfile.open(DATA_BUNDLE, "w:gz") as tar:

        def add(arcname: str, src: Path) -> None:
            tar.add(str(src), arcname=f"{DATA_ROOT}/{arcname}")

        # MANIFEST.txt
        import io

        data = manifest_txt.encode("utf-8")
        info = tarfile.TarInfo(name=f"{DATA_ROOT}/MANIFEST.txt")
        info.size = len(data)
        info.mtime = int(_dt.datetime.now().timestamp())
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))

        # training/ manifests
        print("    + training/train_list.txt")
        add("training/train_list.txt", train_list)
        print("    + training/val_list.txt")
        add("training/val_list.txt", val_list)
        print("    + training/rasa_mr.txt")
        add("training/rasa_mr.txt", rasa_txt)
        print("    + training/indicvoices_r_mr.txt")
        add("training/indicvoices_r_mr.txt", ivr_txt)

        # dataset/audio/
        print(f"    + dataset/audio/rasa/ ({len(rasa_wavs)} WAVs)")
        add("dataset/audio/rasa", PROJ / "dataset" / "audio" / "rasa")
        print(f"    + dataset/audio/indicvoices_r/ ({len(ivr_wavs)} WAVs)")
        add("dataset/audio/indicvoices_r", PROJ / "dataset" / "audio" / "indicvoices_r")

    size = DATA_BUNDLE.stat().st_size
    print(f"  done. {DATA_BUNDLE.name} = {human_bytes(size)} ({size} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    UPLOAD.mkdir(parents=True, exist_ok=True)

    build_training_bundle()
    build_data_bundle()

    # Spot checks
    print("=" * 70)
    print("Spot-check: top 15 entries per tarball")
    print("=" * 70)
    for p in (TRAINING_BUNDLE, DATA_BUNDLE):
        print(f"\n--- {p.name} ---")
        r = subprocess.run(
            ["tar", "-tzf", str(p)],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = r.stdout.splitlines()
        for line in lines[:15]:
            print(f"  {line}")
        print(f"  ... ({len(lines)} total entries)")

    # Checksums
    print("=" * 70)
    print("Computing SHA256 checksums")
    print("=" * 70)
    train_size = TRAINING_BUNDLE.stat().st_size
    data_size = DATA_BUNDLE.stat().st_size
    print(f"  {TRAINING_BUNDLE.name} = {human_bytes(train_size)} ...")
    train_sha = sha256_of(TRAINING_BUNDLE)
    print(f"    sha256: {train_sha}")
    print(f"  {DATA_BUNDLE.name} = {human_bytes(data_size)} ...")
    data_sha = sha256_of(DATA_BUNDLE)
    print(f"    sha256: {data_sha}")

    # Write CHECKSUMS.txt in standard `sha256sum`-compatible format so
    # `sha256sum -c CHECKSUMS.txt` Just Works on the pod.
    with CHECKSUMS.open("w") as fh:
        fh.write(f"{train_sha}  {TRAINING_BUNDLE.name}\n")
        fh.write(f"{data_sha}  {DATA_BUNDLE.name}\n")
    print(f"\nWrote {CHECKSUMS}")

    print("\n" + "=" * 70)
    print("BUILD SUMMARY")
    print("=" * 70)
    print(
        f"  {TRAINING_BUNDLE.name}  {train_size:>14d} bytes  ({human_bytes(train_size)})  sha256={train_sha}"
    )
    print(
        f"  {DATA_BUNDLE.name}     {data_size:>14d} bytes  ({human_bytes(data_size)})  sha256={data_sha}"
    )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
