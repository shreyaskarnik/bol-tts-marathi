"""Convert string speaker names in manifests to integer IDs.

StyleTTS2's meldataset.py does `int(speaker_id)` on the 3rd column, which fails on
our string speaker names (marathi_female, marathi_male, mr_sXXXX). We rewrite to
sequential integer IDs and persist the mapping to speaker_map.json for voicepack
extraction post-training.

Usage (on pod):
  python3 /workspace/bol_run/scripts/fix_speaker_ids.py

Idempotent: if train_list.named.txt already exists, uses it as source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path


TRAINING_DIR = Path("/workspace/bol_run/training")


def main() -> None:
    train = TRAINING_DIR / "train_list.txt"
    val = TRAINING_DIR / "val_list.txt"
    train_named = TRAINING_DIR / "train_list.named.txt"
    val_named = TRAINING_DIR / "val_list.named.txt"
    speaker_map_path = TRAINING_DIR / "speaker_map.json"

    # Back up originals if not yet backed up
    if not train_named.exists():
        train_named.write_text(train.read_text())
        print(f"[fix] backed up {train.name} → {train_named.name}")
    if not val_named.exists():
        val_named.write_text(val.read_text())
        print(f"[fix] backed up {val.name} → {val_named.name}")

    # Build stable speaker_name → int_id mapping from train set
    # Deterministic: sorted by name, so order is stable across runs
    all_speakers: set[str] = set()
    for line in train_named.read_text().splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[2].strip():
            all_speakers.add(parts[2].strip())

    # Marathi anchors first (0=female, 1=male), then IV-R speakers sorted
    ordered = []
    if "marathi_female" in all_speakers:
        ordered.append("marathi_female")
        all_speakers.discard("marathi_female")
    if "marathi_male" in all_speakers:
        ordered.append("marathi_male")
        all_speakers.discard("marathi_male")
    ordered.extend(sorted(all_speakers))

    speaker_to_id = {name: i for i, name in enumerate(ordered)}
    speaker_map_path.write_text(json.dumps(speaker_to_id, indent=2, ensure_ascii=False))
    print(f"[fix] wrote {speaker_map_path.name}: {len(speaker_to_id)} speakers")
    print(f"       marathi_female={speaker_to_id.get('marathi_female')}")
    print(f"       marathi_male={speaker_to_id.get('marathi_male')}")
    print(f"       first IV-R: {ordered[2] if len(ordered) > 2 else 'n/a'}={speaker_to_id.get(ordered[2] if len(ordered) > 2 else '')}")

    # Rewrite manifests with int IDs
    for named, out in [(train_named, train), (val_named, val)]:
        new_lines = []
        missing = 0
        for line in named.read_text().splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            name = parts[2].strip()
            if name not in speaker_to_id:
                # Speaker only in val but not train — assign a new id
                speaker_to_id[name] = len(speaker_to_id)
                missing += 1
            parts[2] = str(speaker_to_id[name])
            new_lines.append("|".join(parts))
        out.write_text("\n".join(new_lines) + "\n")
        print(f"[fix] rewrote {out.name}: {len(new_lines)} lines{', ' + str(missing) + ' new speakers' if missing else ''}")

    # Re-persist in case new val speakers were added
    speaker_map_path.write_text(json.dumps(speaker_to_id, indent=2, ensure_ascii=False))
    print(f"[fix] final speaker_map.json: {len(speaker_to_id)} speakers total")


if __name__ == "__main__":
    main()
