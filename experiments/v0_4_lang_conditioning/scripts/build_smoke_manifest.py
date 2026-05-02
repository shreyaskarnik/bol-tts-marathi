"""
Stub manifest builder for v0.4 architecture smoke test.

Generates a 4-column v0.4 manifest from an existing 3-column v0.2 manifest
(`wav|ipa|speaker`) by appending a stub `lang_ids` column. The stub values are
deterministic-but-pseudo-random per-token, NOT semantically aligned with
phoneme content.

Use this for architecture wiring validation only:
  - NaN check
  - Gradient flow into both lang_embedding rows (mr=0 + en=1)
  - predictor_encoder norm stability (no v0.2-attempt-1 collapse)

For semantically-correct lang_ids on real training data, build_v0_4_manifest.py
(Task #11 v2, post smoke test) does word-by-word re-phonemization with
lexicon-based loanword detection. That requires the source Devanagari text
which is NOT preserved in the v0.2 manifest pipeline.

Usage:
  python build_smoke_manifest.py \
      --input  ../../../training/springlab_mr.txt \
      --output ../data/train_list_smoke.txt \
      --limit 500 \
      --density 0.10
"""

import argparse
import hashlib
import sys
from pathlib import Path

# Resolve TextCleaner from the StyleTTS2 fork (mirrors the dataloader's tokenizer).
HERE = Path(__file__).resolve().parent
STYLETTS2_FORK = HERE.parents[3] / "kokoro-deutsch" / "StyleTTS2"
sys.path.insert(0, str(STYLETTS2_FORK))
from kokoro_symbols import TextCleaner  # noqa: E402

CLEANER = TextCleaner()


def stub_lang_ids(ipa_text: str, density: float) -> list[int]:
    """Per-token lang_ids of length len(TextCleaner(ipa)) + 2 (boundary 0s).

    Hash-seeded LCG so the same IPA string always gets the same labels —
    repeatable across runs, no rng_state plumbing needed.
    """
    n_tokens = len(CLEANER(ipa_text))
    threshold = int(density * 1000)
    seed = int.from_bytes(hashlib.md5(ipa_text.encode()).digest()[:4], "big")
    state = seed
    ids = [0]  # leading boundary token (mr)
    for _ in range(n_tokens):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        ids.append(1 if (state % 1000) < threshold else 0)
    ids.append(0)  # trailing boundary token (mr)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="3-col v0.2 manifest")
    ap.add_argument("--output", type=Path, required=True, help="4-col v0.4 smoke manifest")
    ap.add_argument("--limit", type=int, default=None, help="cap row count")
    ap.add_argument("--density", type=float, default=0.10, help="en-token fraction")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    en_total = 0
    tok_total = 0
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if args.limit is not None and written >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                skipped += 1
                continue
            wav, ipa, speaker = parts
            ids = stub_lang_ids(ipa, density=args.density)
            en_total += sum(ids)
            tok_total += len(ids)
            ids_str = " ".join(str(x) for x in ids)
            fout.write(f"{wav}|{ipa}|{speaker}|{ids_str}\n")
            written += 1

    actual_density = en_total / max(tok_total, 1)
    print(
        f"wrote {written} rows to {args.output} "
        f"(target density {args.density:.2%}, actual {actual_density:.2%}, "
        f"skipped {skipped})"
    )


if __name__ == "__main__":
    main()
