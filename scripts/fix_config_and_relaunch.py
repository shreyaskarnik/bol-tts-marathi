"""Fix the TOP-LEVEL batch_size in config_marathi_ft.yml and print next steps.

Context: train_first.py reads config.get('batch_size') from the TOP LEVEL of the
YAML (per TRAINING_GUIDE.md). Our earlier sed patterns matched a nested
batch_size at line 126 (inside a sub-section) instead of the real top-level one
at line 2, so every previous "batch_size drop" was silently ignored.

This script uses PyYAML to read + edit + write the config safely — no regex hell.

Run on pod:
    python3 /workspace/bol_run/scripts/fix_config_and_relaunch.py [--bs N]

Default new batch_size is 4. Override with --bs 2 if 4 still OOMs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[fail] pyyaml not installed; run: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


CFG = Path("/workspace/bol_run/configs/config_marathi_ft.yml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=4, help="new top-level batch_size")
    args = ap.parse_args()

    if not CFG.exists():
        print(f"[fail] config not found: {CFG}", file=sys.stderr)
        return 2

    # Read as text so we preserve comments/ordering via simple line-level rewrite
    lines = CFG.read_text().splitlines(keepends=True)

    changed = False
    for i, line in enumerate(lines):
        # Match top-level batch_size (no leading whitespace)
        stripped = line.rstrip("\n")
        if stripped.startswith("batch_size:") and not stripped.startswith(" "):
            old_val = stripped.split(":", 1)[1].strip().split()[0]
            new_line = f"batch_size: {args.bs}\n"
            lines[i] = new_line
            print(f"[fix] line {i+1}: batch_size {old_val} → {args.bs}")
            changed = True
            break

    if not changed:
        print("[warn] no top-level batch_size found to change")

    CFG.write_text("".join(lines))

    # Verify using PyYAML parse
    parsed = yaml.safe_load(CFG.read_text())
    top_level_bs = parsed.get("batch_size", "<missing>")
    print(f"[verify] parsed top-level batch_size = {top_level_bs}")

    # Show what train_first.py will actually see
    if top_level_bs != args.bs:
        print(f"[fail] top-level batch_size is {top_level_bs}, expected {args.bs}")
        return 1

    print()
    print("─" * 60)
    print("Next steps — paste these commands to relaunch Stage 1:")
    print("─" * 60)
    print("""
cd /workspace
pkill -9 -f 'accelerate launch' 2>/dev/null; pkill -9 -f 'train_first.py' 2>/dev/null; sleep 2
nvidia-smi --query-gpu=memory.used --format=csv
rm -f training_s1.log
STAGE=1 setsid nohup ./launch_training.sh > training_s1.log 2>&1 < /dev/null &
disown
echo "PID=$!"
sleep 5 && pgrep -fa 'accelerate|train_first' | head -3
tail -f training_s1.log
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
