"""merge_manifests.py — combine Marathi manifest fragments into train/val lists
===============================================================================
Reads two manifest fragments produced by the per-source prep scripts
(``rasa_mr.txt`` and ``indicvoices_r_mr.txt``) and writes the final flat
``train_list.txt`` / ``val_list.txt`` files that the StyleTTS2 / Kokoro
fine-tuning scripts expect.

Layout (all paths relative to the bol-tts-marathi/ repo root):
  training/rasa_mr.txt              input fragment (path|ipa|speaker)
  training/indicvoices_r_mr.txt     input fragment (path|ipa|speaker)
  training/train_list.txt           output (95%, stratified by speaker)
  training/val_list.txt             output (5%,  stratified by speaker)

Behavior:
  - Fragments are optional: a missing fragment produces a warning, not a crash.
  - Lines are sanity-filtered (3 pipe-delimited fields, non-empty IPA, wav file
    must exist under dataset/audio/).
  - Duplicate wav paths are collapsed (first occurrence wins).
  - Tokenizer coverage is probed on a random sample; run aborts below 99%.
  - Split is 95/5 *stratified by speaker*, floor(5%, min 1) to val for speakers
    with >=20 utterances; speakers with <20 utterances go entirely to train.
  - Outputs are sorted by (speaker, wav_path) so repeat runs are bit-identical.

Usage:
  python3 scripts/merge_manifests.py

Requires the repo-local ``training/kokoro_symbols.py`` (TextCleaner).
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# Script lives at scripts/data_prep/, so parents[2] = repo root.
import os
REPO_ROOT = Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[2]))
AUDIO_DIR = REPO_ROOT / "dataset" / "audio"
TRAINING_DIR = REPO_ROOT / "training"

RASA_FRAGMENT = TRAINING_DIR / "rasa_mr.txt"
INDICVOICES_FRAGMENT = TRAINING_DIR / "indicvoices_r_mr.txt"

TRAIN_LIST = TRAINING_DIR / "train_list.txt"
VAL_LIST = TRAINING_DIR / "val_list.txt"

# ── Split / coverage constants ───────────────────────────────────────────────
VAL_RATIO = 0.05
MIN_UTTS_FOR_VAL_SPLIT = 20   # below this, everything goes to train
COVERAGE_SAMPLE_SIZE = 100
COVERAGE_MIN = 0.99           # abort if below
AVG_SECONDS_PER_UTT = 5.0     # rough estimate for hours summary
RANDOM_SEED = 42


# ── Fragment reading ─────────────────────────────────────────────────────────

def _read_fragment(path: Path) -> list[tuple[str, str, str, int]]:
    """Return list of (wav_rel, ipa, speaker, source_line_no) tuples.

    Performs format sanity-check (3 pipe-delimited fields, non-empty IPA) but
    does NOT yet check audio existence — that is done once across the merged
    corpus so the warning aggregates across sources.
    """
    if not path.exists():
        print(f"WARN: fragment not found, skipping: {path.relative_to(REPO_ROOT)}")
        return []

    entries: list[tuple[str, str, str, int]] = []
    dropped_format = 0
    dropped_empty_ipa = 0
    dropped_format_examples: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 3:
                dropped_format += 1
                if len(dropped_format_examples) < 3:
                    dropped_format_examples.append(
                        f"{path.name}:{lineno} ({len(parts)} fields)"
                    )
                continue
            wav_rel, ipa, speaker = parts
            wav_rel = wav_rel.strip()
            ipa = ipa.strip()
            speaker = speaker.strip()
            if not ipa:
                dropped_empty_ipa += 1
                continue
            if not wav_rel or not speaker:
                dropped_format += 1
                continue
            entries.append((wav_rel, ipa, speaker, lineno))

    print(f"  read {path.name}: {len(entries):,} entries")
    if dropped_format:
        print(f"    dropped (bad format): {dropped_format}")
        for ex in dropped_format_examples:
            print(f"      e.g. {ex}")
    if dropped_empty_ipa:
        print(f"    dropped (empty IPA): {dropped_empty_ipa}")
    return entries


# ── Coverage check ───────────────────────────────────────────────────────────

def _coverage_check(entries: list[tuple[str, str, str, int]]) -> float:
    """Sample ``COVERAGE_SAMPLE_SIZE`` IPA lines, report fraction of chars that
    map to a Kokoro vocab index via ``TextCleaner``.

    Aborts (SystemExit) if coverage < COVERAGE_MIN.
    """
    # Import lazily and insert the training/ dir so ``kokoro_symbols`` resolves
    # against the repo-local copy regardless of cwd.
    sys.path.insert(0, str(TRAINING_DIR))
    try:
        from kokoro_symbols import TextCleaner  # type: ignore
    except ImportError as e:
        print(f"ERROR: cannot import kokoro_symbols from {TRAINING_DIR}: {e}")
        sys.exit(1)

    cleaner = TextCleaner()

    rng = random.Random(RANDOM_SEED)
    sample_size = min(COVERAGE_SAMPLE_SIZE, len(entries))
    if sample_size == 0:
        print("ERROR: no entries to coverage-check (empty corpus).")
        sys.exit(1)
    sample = rng.sample(entries, sample_size)

    total_chars = 0
    mapped_chars = 0
    unknown_counter: Counter[str] = Counter()
    for _, ipa, _, _ in sample:
        total_chars += len(ipa)
        mapped_chars += len(cleaner(ipa))
        for ch in ipa:
            if ch not in cleaner.word_index_dictionary:
                unknown_counter[ch] += 1

    coverage = mapped_chars / total_chars if total_chars else 0.0
    print(
        f"  tokenizer coverage on {sample_size} sampled lines: "
        f"{coverage * 100:.2f}% ({mapped_chars}/{total_chars} chars)"
    )
    if unknown_counter:
        top = unknown_counter.most_common(10)
        print("    top unknown chars: "
              + ", ".join(f"{repr(c)}(U+{ord(c):04X})x{n}" for c, n in top))

    if coverage < COVERAGE_MIN:
        print(
            f"ERROR: tokenizer coverage {coverage * 100:.2f}% < "
            f"{COVERAGE_MIN * 100:.0f}% — aborting. Fix the phonemizer or "
            f"extend kokoro_symbols.py before merging."
        )
        sys.exit(1)
    return coverage


# ── Stratified split ─────────────────────────────────────────────────────────

def _stratified_split(
    entries: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Split by speaker: speakers with >=MIN_UTTS_FOR_VAL_SPLIT get floor(5%)
    (min 1) into val, everyone else goes fully to train.

    Deterministic via seeded per-speaker shuffle.
    """
    by_speaker: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for e in entries:
        by_speaker[e[2]].append(e)

    train: list[tuple[str, str, str]] = []
    val: list[tuple[str, str, str]] = []
    rng = random.Random(RANDOM_SEED)
    # Iterate speakers in sorted order for determinism
    for speaker in sorted(by_speaker.keys()):
        utts = list(by_speaker[speaker])
        # Shuffle a deterministic copy — seeded per speaker to avoid having
        # one speaker consume the global RNG state in a way that changes other
        # speakers' splits when the corpus grows.
        local_rng = random.Random((RANDOM_SEED, speaker).__hash__() & 0xFFFFFFFF)
        local_rng.shuffle(utts)
        if len(utts) >= MIN_UTTS_FOR_VAL_SPLIT:
            n_val = max(1, int(len(utts) * VAL_RATIO))
        else:
            n_val = 0
        val.extend(utts[:n_val])
        train.extend(utts[n_val:])
    # rng is seeded but unused past initialization — kept to document the
    # global seed even though per-speaker rngs do the actual shuffling.
    _ = rng
    return train, val


# ── Writers / stats ──────────────────────────────────────────────────────────

def _write_list(path: Path, rows: list[tuple[str, str, str]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r[2], r[0]))
    with path.open("w", encoding="utf-8") as f:
        for wav_rel, ipa, speaker in rows_sorted:
            f.write(f"{wav_rel}|{ipa}|{speaker}\n")


def _print_speaker_summary(
    train: list[tuple[str, str, str]], val: list[tuple[str, str, str]]
) -> None:
    per_speaker_total: Counter[str] = Counter()
    per_speaker_train: Counter[str] = Counter()
    per_speaker_val: Counter[str] = Counter()
    for _, _, s in train:
        per_speaker_total[s] += 1
        per_speaker_train[s] += 1
    for _, _, s in val:
        per_speaker_total[s] += 1
        per_speaker_val[s] += 1

    print("\n── Per-speaker distribution ────────────────────────────────────")
    print(f"  distinct speakers: {len(per_speaker_total)}")
    buckets = {"<5": 0, "<10": 0, "<20": 0}
    for count in per_speaker_total.values():
        if count < 5:
            buckets["<5"] += 1
        if count < 10:
            buckets["<10"] += 1
        if count < 20:
            buckets["<20"] += 1
    print(
        f"  speakers with <5 utts: {buckets['<5']}  "
        f"<10: {buckets['<10']}  <20: {buckets['<20']}"
    )

    top = per_speaker_total.most_common(20)
    print("  top 20 speakers by utterance count:")
    print(f"    {'speaker':<24s} {'total':>7s} {'train':>7s} {'val':>5s}")
    for speaker, total in top:
        print(
            f"    {speaker:<24s} {total:>7d} "
            f"{per_speaker_train[speaker]:>7d} {per_speaker_val[speaker]:>5d}"
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TRAINING_DIR.exists():
        print(f"ERROR: training directory missing: {TRAINING_DIR}")
        sys.exit(1)

    print("── Reading manifest fragments ─────────────────────────────────")
    raw_entries: list[tuple[str, str, str, int]] = []
    raw_entries.extend(_read_fragment(RASA_FRAGMENT))
    raw_entries.extend(_read_fragment(INDICVOICES_FRAGMENT))

    if not raw_entries:
        print("ERROR: no fragments found — nothing to merge.")
        sys.exit(1)

    print(f"  combined raw entries: {len(raw_entries):,}")

    # ── Filter: audio must exist ────────────────────────────────────────
    missing_audio: list[str] = []
    present: list[tuple[str, str, str]] = []
    for wav_rel, ipa, speaker, _ in raw_entries:
        full = AUDIO_DIR / wav_rel
        if not full.is_file():
            missing_audio.append(wav_rel)
            continue
        present.append((wav_rel, ipa, speaker))

    if missing_audio:
        print(
            f"WARN: {len(missing_audio)} lines dropped due to missing audio "
            f"under {AUDIO_DIR.relative_to(REPO_ROOT)}/"
        )
        for ex in missing_audio[:3]:
            print(f"    e.g. {ex}")

    if not present:
        print("ERROR: no entries with existing audio.")
        sys.exit(1)

    # ── Dedup by wav_rel (first occurrence wins) ─────────────────────────
    seen: set[str] = set()
    deduped: list[tuple[str, str, str]] = []
    dup_count = 0
    for row in present:
        if row[0] in seen:
            dup_count += 1
            continue
        seen.add(row[0])
        deduped.append(row)
    if dup_count:
        print(f"  deduplicated: {dup_count} duplicate wav paths collapsed")

    print(f"  post-filter entries: {len(deduped):,}")

    # ── Tokenizer coverage check (aborts on failure) ────────────────────
    print("\n── Tokenizer coverage check ───────────────────────────────────")
    # Convert to 4-tuples for _coverage_check signature
    _coverage_check([(w, i, s, 0) for (w, i, s) in deduped])

    # ── Stratified split ────────────────────────────────────────────────
    print("\n── Stratified 95/5 split ──────────────────────────────────────")
    train, val = _stratified_split(deduped)
    print(f"  train: {len(train):,}   val: {len(val):,}   "
          f"(val {100 * len(val) / max(1, len(train) + len(val)):.2f}%)")

    # ── Write outputs ───────────────────────────────────────────────────
    _write_list(TRAIN_LIST, train)
    _write_list(VAL_LIST, val)
    print(f"  wrote {TRAIN_LIST.relative_to(REPO_ROOT)}")
    print(f"  wrote {VAL_LIST.relative_to(REPO_ROOT)}")

    # ── Summary ─────────────────────────────────────────────────────────
    total = len(train) + len(val)
    est_hours = total * AVG_SECONDS_PER_UTT / 3600.0
    print("\n── Summary ────────────────────────────────────────────────────")
    print(f"  total utterances : {total:,}")
    print(
        f"  estimated hours  : ~{est_hours:.1f} h "
        f"(at {AVG_SECONDS_PER_UTT:.1f}s/utt avg)"
    )
    print(f"  train / val      : {len(train):,} / {len(val):,}")
    _print_speaker_summary(train, val)


if __name__ == "__main__":
    main()
