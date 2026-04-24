"""Diagnose why training is producing NaN losses from step 10.

Checks:
  1. Symbol table has correct canonical Kokoro indices (per semidark TRAINING_GUIDE.md):
     ç=78, ʦ=20, ː=158, ɾ=125, a=43
     Plus our Marathi addition: ɭ=144.
  2. Every IPA char in train_list.txt maps to a vocab slot (no silent drops by TextCleaner).
  3. No duplicate real phonemes in the symbol table.

Run on pod:
  python3 /workspace/bol_run/scripts/diagnose_nan.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/workspace/bol_run/StyleTTS2")
from kokoro_symbols import symbols, dicts, TextCleaner  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("symbol table structural check")
    print("=" * 60)
    print(f"len(symbols) = {len(symbols)}  (expect 178)")
    expected = {
        "ç": 78, "ʦ": 20, "ː": 158,
        "ɾ": 125, "a": 43, "ə": 83,
        "ʰ": 162, "ɖ": 80, "ʈ": 132,
        "ɭ": 144,  # our Marathi addition
    }
    mismatches = 0
    for sym, idx in expected.items():
        got = dicts.get(sym)
        status = "✓" if got == idx else "✗"
        print(f"  {status} dicts[{sym!r}] = {got}  (expect {idx})")
        if got != idx:
            mismatches += 1
    if mismatches:
        print(f"\n{mismatches} mismatches — symbol table is OFF, training would produce NaN")
    else:
        print("\n✓ all canonical indices match")

    # Duplicates
    print()
    print("=" * 60)
    print("duplicate phoneme check")
    print("=" * 60)
    PUA_RANGE = range(0xE000, 0xF900)
    seen = {}
    dupes = []
    for i, s in enumerate(symbols):
        if len(s) == 1 and ord(s) in PUA_RANGE:
            continue
        if s in seen:
            dupes.append((s, seen[s], i))
        else:
            seen[s] = i
    print(f"  dupes: {len(dupes)}")
    for d in dupes[:5]:
        print(f"    {d}")

    # Tokenizer coverage on first 20 lines
    print()
    print("=" * 60)
    print("tokenizer coverage — first 20 train lines")
    print("=" * 60)
    tc = TextCleaner()
    with open("/workspace/bol_run/training/train_list.txt") as f:
        lines = f.readlines()[:20]
    bad = 0
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        wav, ipa, spk = parts
        ids = tc(ipa)
        if len(ids) != len(ipa):
            dropped = [c for c in ipa if c not in dicts]
            short = wav.split("/")[-1][:30]
            print(f"  [DROP] {short:30s}  len_ipa={len(ipa)}  len_ids={len(ids)}  missing={dropped[:5]!r}")
            bad += 1
    print(f"  {bad}/{len(lines)} lines had silently-dropped chars")

    # Scan full manifest for missing chars
    print()
    print("=" * 60)
    print("full-manifest unknown-char scan")
    print("=" * 60)
    missing: Counter[str] = Counter()
    total_chars = 0
    total_lines = 0
    with open("/workspace/bol_run/training/train_list.txt") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            total_lines += 1
            for c in parts[1]:
                total_chars += 1
                if c not in dicts:
                    missing[c] += 1
    print(f"  scanned {total_lines} lines / {total_chars} chars")
    if missing:
        print(f"  UNKNOWN CHARS FOUND:")
        for c, n in missing.most_common(10):
            print(f"    {c!r} (U+{ord(c):04X}): {n} occurrences")
    else:
        print(f"  ✓ every char in train_list is covered by symbol table")


if __name__ == "__main__":
    main()
