#!/usr/bin/env python3
"""preprocess_loanwords.py — Latin English loanword → Devanagari preprocessor.

Rewrites Latin-script English words in the input text to their conventional
Marathi-Devanagari spellings BEFORE TTS phonemization.

Why: when a Marathi-trained TTS receives Latin English (e.g., "Weekend"), it
routes through espeak en G2P → English-IPA neighborhood → decoder produces
"weekenda" / "amajing" patterns. Same word as conventional Devanagari
(वीकेंड) → espeak mr G2P → Marathi-IPA → decoder produces clean
Indian-Marathi rendering. See:
  feedback_devanagari_transliteration_bypasses_decoder_ceiling.md

Two source formats are supported:
  1. data/loanword_map.json — curated Latin→Devanagari map (the format the
     webgpu-demo ships; ~19,450 entries with hand-curated overrides for
     wifi/trip/zomato/etc and Indian brand names). Default if present.
  2. experiments/v0_4_lang_conditioning/data/loanword_dict_dev_to_latin.tsv
     — original IndicCMix-derived TSV, Devanagari→Latin keyed with
     frequency. The script inverts and frequency-picks. Used as fallback.

Both webgpu-demo and inference_mac_mr.py should load from (1) so the python
and browser pipelines stay consistent.

Usage:
    from preprocess_loanwords import preprocess
    preprocess("Weekend ला movie बघायचा plan आहे का?")
    # → "वीकेंड ला मूव्ही बघायचा प्लॅन आहे का?"

CLI:
    python preprocess_loanwords.py --text "Weekend ला movie बघायचा plan आहे का?"
    python preprocess_loanwords.py --dict data/loanword_map.json --text "..."
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict


# Prefer the curated JSON (kept in sync with the webgpu-demo's deployed copy).
# Fall back to the original TSV if the JSON isn't present (e.g. fresh checkout
# before data/ has been populated).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DICT_PATH = (
    _REPO_ROOT / "data" / "loanword_map.json"
    if (_REPO_ROOT / "data" / "loanword_map.json").exists()
    else _REPO_ROOT / "experiments/v0_4_lang_conditioning/data/loanword_dict_dev_to_latin.tsv"
)

# Match runs of basic-Latin letters as candidate English tokens. Apostrophes
# and hyphens kept inside the word; numbers/punctuation split the word.
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def load_loanword_map(dict_path: Path = DEFAULT_DICT_PATH) -> Dict[str, str]:
    """Load loanword map and return a Latin → Devanagari dict.

    Auto-detects the file format by extension:
      - `.json`: a flat `{latin_lower: devanagari}` dict (the curated form
        shipped with the webgpu-demo; preferred).
      - `.tsv`:  the original IndicCMix-derived TSV keyed
        `devanagari\\tlatin\\tfrequency`. The function inverts and
        frequency-picks the canonical Devanagari per lowercased Latin.
    """
    suffix = dict_path.suffix.lower()
    if suffix == ".json":
        with dict_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Curated map is already Latin (lowercase) → Devanagari. Defensively
        # lowercase the keys in case a hand edit slipped a capital in.
        return {k.lower(): v for k, v in data.items()}
    # TSV path: invert + frequency-pick.
    best_for_latin: Dict[str, tuple[int, str]] = {}
    with dict_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            latin = row["latin"].strip().lower()
            dev = row["devanagari"].strip()
            try:
                freq = int(row["frequency"])
            except (KeyError, ValueError):
                freq = 0
            if not latin or not dev:
                continue
            existing = best_for_latin.get(latin)
            if existing is None or freq > existing[0]:
                best_for_latin[latin] = (freq, dev)
    return {latin: dev for latin, (_, dev) in best_for_latin.items()}


def preprocess(text: str, lookup: Dict[str, str] | None = None) -> str:
    """Replace Latin English loanwords with their Devanagari forms.

    Case is matched lower for lookup; the original token's case is dropped
    (Devanagari has no case). Out-of-vocab Latin tokens are kept verbatim.
    Non-Latin segments (Devanagari, punctuation, numerals) are untouched.
    """
    if lookup is None:
        lookup = load_loanword_map()

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return lookup.get(token.lower(), token)

    return LATIN_WORD_RE.sub(replace, text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=False)
    ap.add_argument("--text-file", type=Path)
    ap.add_argument("--dict", type=Path, default=DEFAULT_DICT_PATH)
    ap.add_argument("--show-coverage", action="store_true",
                    help="report which Latin tokens hit/miss the dict")
    args = ap.parse_args()

    lookup = load_loanword_map(args.dict)
    print(f"# loaded {len(lookup):,} Latin→Devanagari mappings")

    if args.text_file:
        lines = args.text_file.read_text(encoding="utf-8").splitlines()
    elif args.text:
        lines = [args.text]
    else:
        lines = [
            "Weekend ला movie बघायचा plan आहे का?",
            "Coffee पिऊया का?",
            "Artificial intelligence conference ला जायचं आहे का?",
            "मी Google मध्ये काम करतो.",
            "Mobile charge करायचा आहे.",
        ]

    for line in lines:
        out = preprocess(line, lookup)
        print(f"in : {line}")
        print(f"out: {out}")
        if args.show_coverage:
            for tok in LATIN_WORD_RE.findall(line):
                hit = tok.lower() in lookup
                print(f"  {tok!r:20s} {'✓ '+lookup[tok.lower()] if hit else '✗ (no entry)'}")
        print()


if __name__ == "__main__":
    main()
