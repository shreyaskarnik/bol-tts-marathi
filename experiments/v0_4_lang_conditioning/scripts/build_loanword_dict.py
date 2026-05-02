"""
Build a Devanagari→Latin English-loanword dictionary from ai4bharat/IndicCMix.

The dataset has parallel columns:
  - native_script_codemixed: mixed-script (English in Latin + Marathi in Devanagari)
  - full_native_script:      same content with English transliterated to Devanagari

Word-by-word alignment via positional zip yields (Devanagari, Latin) loanword
pairs for free — no classifier, no model. Rows where the two columns have
different word counts are skipped (alignment ambiguous).

Output: TSV sorted by frequency, three columns: devanagari, latin, count.
This is the lexicon we use in build_v0_4_manifest.py to identify which
Devanagari tokens in Rasa/IV-R/SPRINGLab text are English loanwords (→ tag
lang=en + romanize for re-phonemization).

Usage:
  python build_loanword_dict.py \\
      --output ../data/loanword_dict_dev_to_latin.tsv \\
      [--limit 100000]   # cap rows scanned (default: all)
"""

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Devanagari range U+0900..U+097F covers Hindi, Marathi, etc.
DEV_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
# Strip leading/trailing punctuation but keep mid-word characters (e.g., apostrophes)
EDGE_PUNCT_RE = re.compile(r"^[^\wऀ-ॿ]+|[^\wऀ-ॿ]+$")


def clean_word(w: str) -> str:
    return EDGE_PUNCT_RE.sub("", w)


def is_pure_latin(w: str) -> bool:
    return bool(LATIN_RE.search(w)) and not DEV_RE.search(w)


def is_pure_devanagari(w: str) -> bool:
    return bool(DEV_RE.search(w)) and not LATIN_RE.search(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on Marathi rows scanned (default: all)")
    ap.add_argument("--min-freq", type=int, default=2,
                    help="drop pairs seen fewer than N times (default: 2)")
    args = ap.parse_args()

    import pandas as pd

    # The default/train/ parquet at ai4bharat/IndicCMix has only 110 rows
    # (an aggregate sample). The real per-language data lives at the repo
    # root as mr.parquet, hi.parquet, etc. — see commit history for those
    # files. mr.parquet has ~105K Marathi-only rows.
    parquet_url = "hf://datasets/ai4bharat/IndicCMix/mr.parquet"
    print(f"loading {parquet_url} (text columns only)...")
    df = pd.read_parquet(parquet_url, columns=[
        "id", "language", "native_script_codemixed", "full_native_script",
    ])
    print(f"loaded {len(df):,} Marathi rows")

    pairs: Counter[tuple[str, str]] = Counter()
    n_marathi = 0
    n_aligned = 0
    n_misaligned = 0
    last_log = time.time()

    for _, row in df.iterrows():
        n_marathi += 1
        if args.limit is not None and n_marathi > args.limit:
            break

        cm_words = row["native_script_codemixed"].split()
        fn_words = row["full_native_script"].split()
        if len(cm_words) != len(fn_words):
            n_misaligned += 1
            continue
        n_aligned += 1

        for cm, fn in zip(cm_words, fn_words):
            cm_c = clean_word(cm)
            fn_c = clean_word(fn)
            if is_pure_latin(cm_c) and is_pure_devanagari(fn_c):
                # Devanagari → Latin pair (case-fold the Latin side)
                pairs[(fn_c, cm_c.lower())] += 1

        # Periodic progress log every ~3 seconds
        if time.time() - last_log > 3.0:
            print(
                f"  mr_rows={n_marathi:>6d}/{len(df):>6d}  "
                f"aligned={n_aligned:>5d}  misaligned={n_misaligned:>5d}  "
                f"unique_pairs={len(pairs):>5d}"
            )
            last_log = time.time()

    print(f"\nfinal: mr_rows={n_marathi} "
          f"aligned={n_aligned} misaligned={n_misaligned}")

    # Aggregate Devanagari-side: a single Devanagari word may map to multiple
    # Latin spellings (capitalization, alt-romanizations). Keep the most
    # frequent Latin form per Devanagari word.
    by_dev: dict[str, dict[str, int]] = {}
    for (dev, lat), c in pairs.items():
        by_dev.setdefault(dev, {})[lat] = c
    consolidated: list[tuple[str, str, int]] = []
    for dev, lat_counts in by_dev.items():
        total = sum(lat_counts.values())
        if total < args.min_freq:
            continue
        # Pick most frequent Latin spelling
        top_lat = max(lat_counts.items(), key=lambda kv: kv[1])[0]
        consolidated.append((dev, top_lat, total))
    consolidated.sort(key=lambda t: -t[2])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("devanagari\tlatin\tfrequency\n")
        for dev, lat, freq in consolidated:
            f.write(f"{dev}\t{lat}\t{freq}\n")
    print(f"\nwrote {len(consolidated)} entries to {args.output} "
          f"(min_freq={args.min_freq})")


if __name__ == "__main__":
    main()
