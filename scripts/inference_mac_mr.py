"""Stage-1 Marathi Kokoro inference on Mac CPU.

Runs the converted Stage 1 checkpoint + mean-style voicepack through Kokoro's
KPipeline with a monkey-patched Marathi lang_code. Output is 24 kHz WAV.

Expect correct Marathi phonemes with near-English pacing — prosody predictor
is barely trained at Stage 1; Stage 2 is where duration/pitch get adversarial
shaping.

Usage:
    python scripts/inference_mac_mr.py
    python scripts/inference_mac_mr.py --text "माझे नाव अमित आहे."
    python scripts/inference_mac_mr.py --text-file my_tests.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Script lives at scripts/, so parents[1] = repo root.
# kokoro/ should be the submodule (or a symlinked clone) at repo root.
_REPO_ROOT = Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[1]))
_KOKORO_SRC = _REPO_ROOT / "kokoro" / "kokoro"
if _KOKORO_SRC.exists() and str(_KOKORO_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_KOKORO_SRC.parent))

import kokoro.pipeline as _kp  # noqa: E402
import numpy as np
import soundfile as sf
import torch

# Monkey-patch: add Marathi. Upstream KPipeline ships a,b,d,e,f,h,i,j,p,z only.
# 'h' is already Hindi; pick 'm' for Marathi → espeak-ng language 'mr'.
_kp.LANG_CODES["m"] = "mr"

from kokoro import KModel, KPipeline  # noqa: E402

# Loanword preprocessor: rewrite Latin-script English in input text to its
# conventional Devanagari spelling BEFORE phonemization. Mirrors the
# webgpu-demo's client-side preprocessor — without this, mixed Marathi+English
# (Minglish) input routes "Weekend" through espeak's English G2P → English IPA
# the Marathi-trained decoder doesn't know. With Devanagari transliteration,
# "Weekend" → "वीकेंड" → espeak-mr → Marathi IPA → clean output. See
# feedback_devanagari_transliteration_bypasses_decoder_ceiling.md.
_PREPROCESS_LOANWORDS_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocess_loanwords import preprocess as _preprocess_loanwords  # noqa: E402
    from preprocess_loanwords import load_loanword_map as _load_loanword_map  # noqa: E402
    _PREPROCESS_LOANWORDS_AVAILABLE = True
except Exception as _e:
    print(f"[warn] preprocess_loanwords not loadable ({_e}); skipping Minglish preprocessing")

# Default Marathi test set — retroflex (ळ=ɭ, ट, ड), aspirates, clusters,
# question prosody, numbers.
DEFAULT_TESTS = [
    "नमस्कार मी मराठी बोलतो.",
    "आज हवामान खूप छान आहे.",
    "तू कुठे चाललास?",
    "ती मुलगी झाडाखाली बसली आहे.",
    "मला केळी आणि आंबा आवडतो.",
    "हे पुस्तक खूप महत्त्वाचे आहे.",
    "सात वाजता भेटूया.",
]


def main() -> int:
    # Default checkpoint dir: $BOL_CHECKPOINTS if set, else <repo>/checkpoints
    default_ckpt_dir = Path(os.environ.get("BOL_CHECKPOINTS", _REPO_ROOT / "checkpoints"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=str(default_ckpt_dir / "kokoro_mr_final.pth"),
    )
    ap.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "config_mr.json"),
    )
    ap.add_argument(
        "--voicepack",
        default=str(default_ckpt_dir / "voices" / "mf_asha.pt"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(default_ckpt_dir / "test_output"),
    )
    ap.add_argument("--text", default=None, help="single Marathi sentence")
    ap.add_argument(
        "--text-file", default=None, help="newline-separated Marathi sentences"
    )
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument(
        "--no-loanword-preprocess",
        action="store_true",
        help="skip Latin→Devanagari loanword rewrite (e.g. for testing pure-Marathi corpora)",
    )
    args = ap.parse_args()

    if args.text:
        texts = [args.text]
    elif args.text_file:
        texts = [
            l.strip()
            for l in Path(args.text_file).read_text().splitlines()
            if l.strip()
        ]
    else:
        texts = DEFAULT_TESTS

    device = "cpu"
    print(f"device: {device}")
    print(f"loading KModel — config={args.config}")
    kmodel = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=args.config,
        model=args.model,
        disable_complex=True,
    ).to(device)
    kmodel.train(False)  # switch to inference mode (equivalent to .eval())

    print("creating KPipeline(lang_code='m' → espeak 'mr')")
    pipeline = KPipeline(lang_code="m", repo_id="hexgrad/Kokoro-82M", model=kmodel)

    print(f"loading voicepack: {args.voicepack}")
    voice = torch.load(args.voicepack, map_location="cpu", weights_only=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load loanword map once (~19K entries; load is cheap, lookup is dict).
    loanword_lookup = None
    if _PREPROCESS_LOANWORDS_AVAILABLE and not args.no_loanword_preprocess:
        try:
            loanword_lookup = _load_loanword_map()
            print(f"loaded loanword map: {len(loanword_lookup)} Latin→Devanagari entries")
        except Exception as e:
            print(f"[warn] loanword map load failed ({e}); skipping preprocess")
            loanword_lookup = None

    print(f"\nsynthesizing {len(texts)} utterance(s)...\n")
    for i, text in enumerate(texts, 1):
        original_text = text
        if loanword_lookup is not None:
            text = _preprocess_loanwords(text, loanword_lookup)
            if text != original_text:
                print(f"[{i}/{len(texts)}] {original_text}")
                print(f"           → {text}  (loanword preprocess)")
            else:
                print(f"[{i}/{len(texts)}] {text}")
        else:
            print(f"[{i}/{len(texts)}] {text}")
        chunks = []
        for _gs, ps, audio in pipeline(text, voice=voice, speed=args.speed):
            print(f"    phonemes: {ps}")
            chunks.append(audio)
        if not chunks:
            print("    (no audio)")
            continue
        wav = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        path = out / f"mr_{i:02d}.wav"
        sf.write(str(path), wav, 24000)
        print(f"    wrote {path} ({len(wav) / 24000:.1f}s)")

    print(f"\ndone. audio at: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
