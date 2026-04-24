"""Synthesize Marathi text AND extract phoneme / word-level timestamps.

Kokoro's `KModel.forward_with_tokens` returns `(audio, pred_dur)` where
`pred_dur` is per-phoneme duration in frames (hop_size = 300). With the sample
rate (24000 Hz) that maps directly to seconds.

For WORD-level timestamps, we need to keep track of which phoneme indices
belong to which word. misaki's espeak backend returns the IPA for the whole
sentence as a single string, so we phonemize word-by-word ourselves and
accumulate the phoneme-index ranges per word.

Usage:

    python scripts/with_timestamps.py \
        --model     checkpoints/kokoro_mr_final.pth \
        --config    configs/config_mr.json \
        --voicepack voices/mf_asha.pt \
        --text      "नमस्कार, मी मराठी बोलतो. मला केळी आणि आंबा आवडतो." \
        --out-wav   out.wav \
        --out-json  out.json

Output JSON shape:

    {
      "sample_rate": 24000,
      "phonemes":   [{"phone": "n",  "start": 0.00,  "end": 0.04}, ...],
      "words":      [{"word":  "नमस्कार", "start": 0.00, "end": 0.52}, ...]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_submodule_to_path() -> None:
    here = Path(__file__).resolve().parent
    sub = here.parent / "kokoro" / "kokoro"
    if sub.exists() and str(sub.parent) not in sys.path:
        sys.path.insert(0, str(sub.parent))


_add_submodule_to_path()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import kokoro.pipeline as _kp  # noqa: E402

_kp.LANG_CODES["m"] = "mr"  # Marathi monkey-patch

from kokoro import KModel  # noqa: E402
from misaki import espeak  # noqa: E402

# pred_dur frames are at 600 samples each (prosody predictor runs at half the
# mel-frame rate; decoder upsamples 2x internally). Empirically verified:
# audio.numel() / pred_dur.sum() == 600.
HOP = 600
SR = 24000


def _tokenize(phonemes: str, vocab: dict[str, int]) -> list[int]:
    """Kokoro's KModel tokenization: drop phonemes not in vocab."""
    return [vocab[p] for p in phonemes if p in vocab]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--voicepack", required=True)
    ap.add_argument("--text", required=True, help="Marathi sentence(s) in Devanagari")
    ap.add_argument("--out-wav", default="out.wav")
    ap.add_argument("--out-json", default="out.json")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    vocab: dict[str, int] = cfg["vocab"]

    kmodel = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=args.config,
        model=args.model,
        disable_complex=True,  # harmless on CPU; required for ONNX export
    ).to(args.device)
    kmodel.train(False)

    voice = torch.load(args.voicepack, map_location="cpu", weights_only=True)

    g2p = espeak.EspeakG2P(language="mr")

    # Phonemize word-by-word so we know which phoneme indices belong to which word.
    # Treat whitespace as word separator. Punctuation stays attached to the preceding
    # word (this mirrors how misaki phonemizes the full string anyway).
    raw_words = args.text.split()
    words: list[str] = []
    word_ipas: list[str] = []
    for w in raw_words:
        ipa, _ = g2p(w)
        words.append(w)
        word_ipas.append(ipa)

    # Build the full phoneme sequence + per-word phoneme-index ranges
    all_phones: list[str] = []
    word_ranges: list[tuple[int, int]] = []  # [start_idx_in_all_phones, end_idx)
    for ipa in word_ipas:
        start = len(all_phones)
        for c in ipa:
            if c in vocab:
                all_phones.append(c)
            # Chars not in vocab (e.g. some stress markers) get dropped —
            # which matches KModel's tokenization, so timestamps stay aligned.
        end = len(all_phones)
        word_ranges.append((start, end))

    input_ids = torch.tensor([vocab[p] for p in all_phones], dtype=torch.long, device=args.device)

    # Kokoro expects a per-phoneme-position style slice — broadcast the voicepack's
    # [510, 1, 256] to the phoneme length. Standard KPipeline does this internally;
    # here we replicate it explicitly.
    voice = voice.to(args.device)
    ref_s = voice[len(input_ids) - 1]  # Kokoro's convention: use slice at index (n_phones - 1)

    # Forward with tokens to get pred_dur alongside audio
    audio, pred_dur = kmodel.forward_with_tokens(
        input_ids=input_ids, ref_s=ref_s, speed=float(args.speed)
    )
    pred_dur_np = pred_dur.squeeze().cpu().numpy()  # [n_phonemes]
    audio_np = audio.squeeze().cpu().numpy()

    # Per-phoneme durations in seconds
    dur_sec = pred_dur_np.astype("float64") * HOP / SR
    starts = dur_sec.cumsum() - dur_sec  # start time of each phoneme
    ends = starts + dur_sec

    phoneme_entries = [
        {"phone": p, "start": round(float(starts[i]), 4), "end": round(float(ends[i]), 4)}
        for i, p in enumerate(all_phones)
    ]

    word_entries = []
    for word, (s, e) in zip(words, word_ranges):
        if s == e:
            # word had no in-vocab phonemes (unlikely with Marathi)
            continue
        word_entries.append(
            {
                "word": word,
                "start": round(float(starts[s]), 4),
                "end": round(float(ends[e - 1]), 4),
            }
        )

    out_wav = Path(args.out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio_np, SR)

    out_json = Path(args.out_json)
    out_json.write_text(
        json.dumps(
            {"sample_rate": SR, "phonemes": phoneme_entries, "words": word_entries},
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"wrote {out_wav} ({len(audio_np) / SR:.2f}s) and {out_json} "
          f"({len(word_entries)} words, {len(phoneme_entries)} phonemes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
