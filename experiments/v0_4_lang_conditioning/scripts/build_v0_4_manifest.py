"""
v0.4 manifest builder — Approach B-lite per design doc.

Takes existing v0.2 3-col manifest (`wav|ipa|speaker`) and emits 4-col v0.4
manifest with per-token lang_ids:
    wav | ipa(unchanged) | speaker | lang_ids

Strategy: each space-separated chunk in the v0.2 IPA is one source word's
phonemes. For each Devanagari loanword in our dict, we pre-compute its
mr-G2P IPA (using misaki+espeak, same pipeline as the original prep). At
manifest-build time, we look up each IPA chunk against this set; matches
get lang=1 (en), non-matches get lang=0 (mr).

IPA itself is NOT modified — that would require re-streaming raw text
(deferred to v0.4.1 Approach A if B-lite shows improvement).

Output format mirrors the dataloader contract:
  - lang_ids has length len(text_cleaner(ipa)) + 2  (matching boundary 0s)
  - lang_ids in {0, 1}, space-separated ints

Usage:
  python build_v0_4_manifest.py \\
    --dict      ../data/loanword_dict_dev_to_latin.tsv \\
    --input     ../../../training/springlab_mr.txt \\
    --output    ../data/train_list_v0_4.txt \\
    --min-freq  20

Limit dict to entries with freq>=20 to keep noise low while covering the
common loanwords. The auto-aligned dict has tail entries (e.g., random
proper nouns appearing once or twice) that aren't true loanwords.
"""

import argparse
import re
import sys
import time
from pathlib import Path

# Resolve TextCleaner from the StyleTTS2 fork (mirrors dataloader's tokenizer).
HERE = Path(__file__).resolve().parent
STYLETTS2_FORK = HERE.parents[3] / "kokoro-deutsch" / "StyleTTS2"
sys.path.insert(0, str(STYLETTS2_FORK))
from kokoro_symbols import TextCleaner  # noqa: E402


# Common Marathi case suffixes / postpositions that get glued to noun stems
# without a space. Auto-generating these inflected forms catches "मीटिंगला"
# even when the dict only has "मीटिंग". Empty string = the base form itself.
INFLECTION_SUFFIXES = [
    "",        # base form
    "ला",     # to / at (dative)
    "चा",     # of (masc)
    "ची",     # of (fem)
    "चे",     # of (neut/pl)
    "च्या",   # of (oblique)
    "त",       # in (locative short)
    "तला",   # the-one-in
    "तली",   # the-one-in (fem)
    "तले",   # the-one-in (neut/pl)
    "मध्ये", # in (madhye)
    "ने",     # instrumental
    "नी",     # instrumental/agentive
    "स",       # to (dative short)
    "ंना",     # to-them (oblique-pl + ना)
    "ांना",   # plural+oblique+ना
    "ही",     # also/too (emphatic clitic)
    "च",       # only (emphatic clitic)
    "ः",       # honorific tag (rare)
]


def build_en_ipa_set(dict_path: Path, min_freq: int) -> set[str]:
    """Pre-compute mr-G2P IPA for each Devanagari loanword + common inflections."""
    from misaki import espeak

    g2p = espeak.EspeakG2P(language="mr")

    devanagari_words: list[str] = []
    with dict_path.open(encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            dev, _lat, freq_s = parts
            if int(freq_s) >= min_freq:
                devanagari_words.append(dev)

    print(f"[dict] {len(devanagari_words)} root entries pass min_freq={min_freq}")
    print(f"[dict] expanding by {len(INFLECTION_SUFFIXES)} suffix variants each "
          f"= {len(devanagari_words) * len(INFLECTION_SUFFIXES)} candidate forms")

    en_ipa_set: set[str] = set()
    skipped = 0
    last_log = time.time()
    for i, dev in enumerate(devanagari_words):
        for suffix in INFLECTION_SUFFIXES:
            form = dev + suffix
            try:
                ipa, _ = g2p(form)
            except Exception:
                skipped += 1
                continue
            ipa = ipa.strip()
            if ipa:
                en_ipa_set.add(ipa)
        if time.time() - last_log > 3.0:
            print(f"  phonemized {i}/{len(devanagari_words)} roots, "
                  f"{len(en_ipa_set)} unique IPA forms")
            last_log = time.time()

    print(f"[dict] {len(en_ipa_set)} unique IPA forms (skipped {skipped} G2P failures)")
    return en_ipa_set


# Edge-strip punctuation (commas, periods) but keep IPA characters
EDGE_RE = re.compile(r"^[^\wɐ-ʯÀ-ɏ̀-ͯ'ˈˌːʰʲʷ̃\-]+|[^\wɐ-ʯÀ-ɏ̀-ͯ'ˈˌːʰʲʷ̃\-]+$")


def normalize_ipa_chunk(chunk: str) -> str:
    """Strip leading/trailing punctuation from an IPA chunk. Keeps IPA chars."""
    return EDGE_RE.sub("", chunk).strip()


def derive_lang_ids(ipa: str, en_ipa_set: set[str], cleaner: TextCleaner) -> list[int]:
    """
    Walk char-by-char through ipa, marking each char's lang based on the
    word it belongs to. Then drop chars that TextCleaner skips (unknown
    chars), so the result aligns with the dataloader's tokenized text.

    Returns lang_ids of length len(cleaner(ipa)) + 2 (with boundary 0s).
    """
    cleaner_dict = cleaner.word_index_dictionary

    # Step 1: for each char position, what's its word's lang?
    char_lang = [0] * len(ipa)
    word_start = None
    for i, ch in enumerate(ipa):
        if ch.isspace():
            if word_start is not None:
                # Just hit end of a word — look up
                word = ipa[word_start:i]
                norm = normalize_ipa_chunk(word)
                if norm in en_ipa_set:
                    for j in range(word_start, i):
                        char_lang[j] = 1
                word_start = None
        else:
            if word_start is None:
                word_start = i
    # Final word (no trailing space)
    if word_start is not None:
        word = ipa[word_start:]
        norm = normalize_ipa_chunk(word)
        if norm in en_ipa_set:
            for j in range(word_start, len(ipa)):
                char_lang[j] = 1

    # Step 2: filter to chars that TextCleaner accepts (mirroring dataloader)
    inner_lang_ids = [
        char_lang[i] for i, ch in enumerate(ipa) if ch in cleaner_dict
    ]

    # Step 3: add boundary 0 tokens (the dataloader does insert(0,0); append(0))
    return [0] + inner_lang_ids + [0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="3-col v0.2 manifest")
    ap.add_argument("--output", type=Path, required=True, help="4-col v0.4 manifest")
    ap.add_argument("--min-freq", type=int, default=2,
                    help="dict freq cutoff. min=2 + inflection expansion lands "
                         "the flip rate in the design-doc 10-25% sweet spot on "
                         "the v0.2 corpus; higher cutoffs lose too much coverage")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    en_ipa_set = build_en_ipa_set(args.dict, args.min_freq)
    cleaner = TextCleaner()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_written = 0
    n_skipped = 0
    en_token_total = 0
    token_total = 0
    rows_with_en = 0

    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            if args.limit is not None and n_written >= args.limit:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                n_skipped += 1
                continue
            wav, ipa, speaker = parts
            try:
                lang_ids = derive_lang_ids(ipa, en_ipa_set, cleaner)
            except Exception as e:
                print(f"[WARN] {wav}: {e}", file=sys.stderr)
                n_skipped += 1
                continue

            # Sanity: lang_ids length must == len(cleaner(ipa)) + 2
            expected = len(cleaner(ipa)) + 2
            if len(lang_ids) != expected:
                print(
                    f"[WARN] {wav}: len(lang_ids)={len(lang_ids)} != expected={expected}",
                    file=sys.stderr,
                )
                n_skipped += 1
                continue

            n_en = sum(lang_ids)
            en_token_total += n_en
            token_total += len(lang_ids)
            if n_en > 0:
                rows_with_en += 1

            ids_str = " ".join(str(x) for x in lang_ids)
            fout.write(f"{wav}|{ipa}|{speaker}|{ids_str}\n")
            n_written += 1

    flip_pct = en_token_total / max(token_total, 1) * 100
    rows_pct = rows_with_en / max(n_written, 1) * 100
    print(
        f"\n[v0.4 manifest] wrote {n_written} rows to {args.output} "
        f"(skipped {n_skipped}, total scanned {n_total})"
    )
    print(
        f"  flip rate: {en_token_total}/{token_total} tokens = {flip_pct:.2f}% en"
    )
    print(
        f"  rows with >=1 en token: {rows_with_en}/{n_written} = {rows_pct:.2f}%"
    )
    print(
        f"  (sweet spot per design doc: 10-25% token flip rate; "
        f"<5% = lexicon too narrow, >40% = too aggressive)"
    )


if __name__ == "__main__":
    main()
