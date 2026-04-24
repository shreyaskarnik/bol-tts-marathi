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

    print(f"\nsynthesizing {len(texts)} utterance(s)...\n")
    for i, text in enumerate(texts, 1):
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
