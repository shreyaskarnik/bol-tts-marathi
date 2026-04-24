"""
Generate OOD_texts.txt for Stage 2 (SLM adversarial) training on Marathi.

Pulls Marathi Wikipedia from HF, splits into sentences, filters, phonemizes
via misaki.espeak(language='mr'), confirms coverage by Kokoro's 178-token vocab,
and writes exactly ~100 IPA sentences — one per line — to training/OOD_texts.txt.

Usage:
  uv run --python 3.12 \\
      --with "misaki[en]>=0.9.4" \\
      --with datasets \\
      --with tqdm \\
      python3 scripts/generate_ood_marathi.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# --- Paths ------------------------------------------------------------------
# Script lives at scripts/data_prep/, so parents[2] = repo root.
import os
PROJECT_ROOT = Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[2]))
TRAINING_DIR = PROJECT_ROOT / "training"
OUT_PATH = TRAINING_DIR / "OOD_texts.txt"

# Make training/ importable so we can load kokoro_symbols.py
sys.path.insert(0, str(TRAINING_DIR))

from kokoro_symbols import dicts as KOKORO_SYMBOL_DICT  # noqa: E402

# --- Target ----------------------------------------------------------------

TARGET_LINES = 100

# --- Filters ---------------------------------------------------------------

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
SENTENCE_SPLIT_RE = re.compile(r"[।\.\?\!]+")
URL_HINTS = ("http", "www.")

MIN_CHARS = 30
MAX_CHARS = 300
MIN_WORDS = 5
MAX_LATIN_WORDS_IN_ROW = 3  # reject sentences with more than 3 consecutive Latin words
MIN_IPA_CHARS = 10
MAX_IPA_CHARS = 400
MIN_COVERAGE = 0.99

# --- HF token (optional; wikimedia/wikipedia is public) --------------------

def _maybe_set_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        try:
            tok = token_path.read_text().strip()
            if tok:
                os.environ["HF_TOKEN"] = tok
        except OSError:
            pass


# --- Sentence extraction ---------------------------------------------------

def extract_sentences(article_text: str) -> list[str]:
    """Split a Wikipedia article into candidate Marathi sentences."""
    if not article_text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", article_text).strip()
    raw = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


def passes_text_filters(sent: str) -> tuple[bool, str]:
    """Return (ok, reason_if_rejected)."""
    if any(h in sent.lower() for h in URL_HINTS):
        return False, "url"
    devan_chars = DEVANAGARI_RE.findall(sent)
    if len(devan_chars) < MIN_CHARS:
        return False, "too_short_devanagari"
    if len(sent) > MAX_CHARS:
        return False, "too_long"
    words = sent.split()
    if len(words) < MIN_WORDS:
        return False, "too_few_words"
    # Reject if too much Latin script
    latin_words = LATIN_WORD_RE.findall(sent)
    if len(latin_words) > MAX_LATIN_WORDS_IN_ROW:
        return False, "too_much_latin"
    # Additional check: must be mostly Devanagari
    non_space = [c for c in sent if not c.isspace()]
    if non_space:
        devan_ratio = len(devan_chars) / len(non_space)
        if devan_ratio < 0.70:
            return False, "not_mostly_devanagari"
    return True, ""


# --- Tokenizer coverage ----------------------------------------------------

def coverage_ratio(ipa: str) -> tuple[float, set[str]]:
    if not ipa:
        return 0.0, set()
    in_vocab = 0
    missing: set[str] = set()
    for ch in ipa:
        if ch in KOKORO_SYMBOL_DICT:
            in_vocab += 1
        else:
            missing.add(ch)
    return in_vocab / len(ipa), missing


# --- Main ------------------------------------------------------------------

def main() -> int:
    _maybe_set_hf_token()

    # Lazy imports (heavy)
    from datasets import load_dataset
    from tqdm import tqdm

    from misaki.espeak import EspeakG2P

    print("Loading misaki EspeakG2P(language='mr')...", flush=True)
    g2p = EspeakG2P(language="mr")

    print("Opening streaming wikimedia/wikipedia 20231101.mr...", flush=True)
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.mr",
        split="train",
        streaming=True,
    )

    rejections: dict[str, int] = {
        "url": 0,
        "too_short_devanagari": 0,
        "too_long": 0,
        "too_few_words": 0,
        "too_much_latin": 0,
        "not_mostly_devanagari": 0,
        "ipa_empty_or_short": 0,
        "ipa_too_long": 0,
        "tokenizer_coverage": 0,
        "phonemize_error": 0,
        "duplicate": 0,
    }

    accepted: list[str] = []
    seen_ipa: set[str] = set()
    seen_source: set[str] = set()
    missing_chars_total: dict[str, int] = {}

    pbar = tqdm(total=TARGET_LINES, desc="accepted")
    article_count = 0

    for article in wiki:
        if len(accepted) >= TARGET_LINES:
            break
        article_count += 1
        text = article.get("text", "")
        for sent in extract_sentences(text):
            if len(accepted) >= TARGET_LINES:
                break
            if sent in seen_source:
                rejections["duplicate"] += 1
                continue
            seen_source.add(sent)

            ok, reason = passes_text_filters(sent)
            if not ok:
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            # Phonemize
            try:
                result = g2p(sent)
                ipa = result[0] if isinstance(result, tuple) else result
            except Exception as exc:  # pragma: no cover
                rejections["phonemize_error"] += 1
                if rejections["phonemize_error"] <= 3:
                    print(f"  phonemize error: {exc!r} on: {sent[:60]}...", flush=True)
                continue

            if not ipa or len(ipa) < MIN_IPA_CHARS:
                rejections["ipa_empty_or_short"] += 1
                continue
            if len(ipa) > MAX_IPA_CHARS:
                rejections["ipa_too_long"] += 1
                continue

            ratio, missing = coverage_ratio(ipa)
            if ratio < MIN_COVERAGE:
                rejections["tokenizer_coverage"] += 1
                for m in missing:
                    missing_chars_total[m] = missing_chars_total.get(m, 0) + 1
                if rejections["tokenizer_coverage"] <= 5:
                    miss_repr = {
                        f"U+{ord(c):04X}({c!r})": 1 for c in list(missing)[:8]
                    }
                    print(
                        f"  coverage {ratio:.3f} < {MIN_COVERAGE}; missing={miss_repr}",
                        flush=True,
                    )
                continue

            if ipa in seen_ipa:
                rejections["duplicate"] += 1
                continue
            seen_ipa.add(ipa)

            accepted.append(ipa)
            pbar.update(1)

    pbar.close()

    # Write output
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for line in accepted:
            # Collapse any internal whitespace to single space, strip
            cleaned = re.sub(r"\s+", " ", line).strip()
            f.write(cleaned + "\n")

    # --- Report -----------------------------------------------------------
    print("\n=== GENERATION REPORT ===", flush=True)
    print(f"Articles scanned:    {article_count}", flush=True)
    print(f"Accepted lines:      {len(accepted)} (target {TARGET_LINES})", flush=True)
    print(f"Output path:         {OUT_PATH}", flush=True)
    print("Rejection breakdown:", flush=True)
    for k, v in sorted(rejections.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v}", flush=True)

    if missing_chars_total:
        print("\nTop missing IPA chars (char: count):", flush=True)
        items = sorted(missing_chars_total.items(), key=lambda kv: -kv[1])[:15]
        for ch, cnt in items:
            print(f"  U+{ord(ch):04X} {ch!r}  {cnt}", flush=True)

    print("\nFirst 5 lines of output:", flush=True)
    for i, line in enumerate(accepted[:5], 1):
        print(f"  [{i}] {line}", flush=True)

    return 0 if len(accepted) >= TARGET_LINES else 1


if __name__ == "__main__":
    sys.exit(main())
