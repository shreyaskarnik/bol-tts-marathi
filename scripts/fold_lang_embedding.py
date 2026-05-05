#!/usr/bin/env python3
"""Fold v0.4 lang_embedding[row] into bert.embeddings.word_embeddings.weight.

After scripts/upstream/test_inference.py:convert_checkpoint exports a v0.4
StyleTTS2 ckpt to Kokoro KModel format, the weights still carry
CustomAlbert.lang_embedding. Stock Kokoro KModel.forward_with_tokens does
not pass lang_ids, so the learned bias would be silently dropped at
deployment (the v0.4 train/inference mismatch documented in v0.4.2):
PLBERT input is OOD by exactly lang_embedding[row] vs. what the predictor
was trained against → "radio tuning" audio.

Folding the bias into the word-embedding table makes the saved weights
numerically equivalent to running CustomAlbert with lang_ids set to a
constant `row` for every position. Stock KModel — which only knows
vanilla AlbertModel — then just works, no lang_embedding module needed.

Limitation: single-language fold. The fold uses one row of
lang_embedding, so deployed inference is locked to that language. For
mixed-language deployment, fix the inference call site instead (thread
lang_ids through KModel.forward_with_tokens).

Idempotency: refuses to run if no lang_embedding key is present.
Catches the "already folded" / "wrong file" footgun loudly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

LANG_EMB_KEYS = ("module.lang_embedding.weight", "lang_embedding.weight")
WORD_EMB_KEYS = (
    "module.embeddings.word_embeddings.weight",
    "embeddings.word_embeddings.weight",
)


def _find_key(state_dict: dict, candidates: tuple[str, ...]) -> str | None:
    return next((k for k in candidates if k in state_dict), None)


def fold(input_path: Path, output_path: Path, lang_row: int = 0) -> None:
    weights = torch.load(str(input_path), map_location="cpu", weights_only=False)

    if "bert" not in weights:
        sys.exit(
            f"expected 'bert' top-level module in {input_path}; "
            f"got {sorted(weights)}. Is this a Kokoro-format .pth?"
        )
    bert = weights["bert"]

    lang_key = _find_key(bert, LANG_EMB_KEYS)
    if lang_key is None:
        sys.exit(
            f"no lang_embedding key in bert state dict (already folded? "
            f"pre-v0.4 ckpt? wrong file?). bert key sample: "
            f"{sorted(bert)[:6]}"
        )
    word_key = _find_key(bert, WORD_EMB_KEYS)
    if word_key is None:
        sys.exit(
            f"no word_embeddings key in bert state dict. "
            f"bert key sample: {sorted(bert)[:6]}"
        )

    lang = bert[lang_key]
    word = bert[word_key]
    if lang.dim() != 2 or word.dim() != 2:
        sys.exit(f"unexpected shapes: lang={tuple(lang.shape)} word={tuple(word.shape)}")
    if lang_row < 0 or lang_row >= lang.shape[0]:
        sys.exit(f"lang_row={lang_row} out of range; lang_embedding has {lang.shape[0]} rows")
    if lang.shape[1] != word.shape[1]:
        sys.exit(
            f"embedding-dim mismatch: lang={tuple(lang.shape)} word={tuple(word.shape)}"
        )

    bias = lang[lang_row].clone()
    print(
        f"folding bert[{lang_key!r}][{lang_row}] "
        f"(||bias|| = {bias.norm().item():.4f}, dim = {bias.shape[0]}) "
        f"into bert[{word_key!r}] ({tuple(word.shape)})"
    )
    bert[word_key] = word + bias.to(word.dtype)
    del bert[lang_key]
    print(f"  stripped bert[{lang_key!r}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"saved folded ckpt: {output_path} ({size_mb:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Kokoro-format .pth produced by scripts/upstream/test_inference.convert_checkpoint",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the folded .pth (loadable by stock Kokoro KModel)",
    )
    ap.add_argument(
        "--lang-row",
        type=int,
        default=0,
        help="Row of lang_embedding to fold (0=mr, 1=en — see Utils/PLBERT/util.py NUM_LANGUAGES). Default 0.",
    )
    args = ap.parse_args()
    fold(args.input, args.output, lang_row=args.lang_row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
