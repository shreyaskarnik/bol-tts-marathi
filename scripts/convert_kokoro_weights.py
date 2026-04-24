#!/usr/bin/env python3
"""Convert Kokoro-82M pretrained weights to StyleTTS2-compatible format.

Kokoro's raw checkpoint format:
    {'bert': {'module.X': tensor, ...},
     'bert_encoder': {...},
     'predictor': {...},
     'decoder': {...},
     'text_encoder': {...}}

StyleTTS2's `load_checkpoint` expects:
    {'net': {component_name: state_dict_with_module_prefix_stripped, ...}}

This script strips the 'module.' prefix from every key and wraps the whole
thing in {'net': ...}. The source checkpoint is read from the already-
downloaded local copy; no HF re-download is performed.

Also copies the upstream semidark `config.json` into our training/ dir so
downstream training scripts find the expected vocab/config.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# Script lives at scripts/, so parents[1] = repo root.
import os
REPO_ROOT = Path(os.environ.get("BOL_REPO", Path(__file__).resolve().parents[1]))
# Source weights: where the raw Kokoro-82M .pth lives. Override via BOL_KOKORO_BASE.
SRC_WEIGHTS = Path(
    os.environ.get("BOL_KOKORO_BASE", REPO_ROOT.parent / "Models" / "Kokoro" / "kokoro-v1_0.pth")
)
SRC_CONFIG = REPO_ROOT / "configs" / "config_mr.json"
TRAINING_DIR = REPO_ROOT / "training"
OUT_WEIGHTS = TRAINING_DIR / "kokoro_base.pth"
OUT_CONFIG = TRAINING_DIR / "config.json"

EXPECTED_MODULES = {"bert", "bert_encoder", "predictor", "decoder", "text_encoder"}


def convert(force: bool = False) -> None:
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required. Install: pip install torch")
        sys.exit(1)

    if not SRC_WEIGHTS.exists():
        print(f"ERROR: source checkpoint not found: {SRC_WEIGHTS}")
        sys.exit(1)
    if not SRC_CONFIG.exists():
        print(f"ERROR: upstream config.json not found: {SRC_CONFIG}")
        sys.exit(1)

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_WEIGHTS.exists() and not force:
        print(f"{OUT_WEIGHTS} already exists, skipping (use --force to regenerate)")
    else:
        print(f"Loading Kokoro weights from {SRC_WEIGHTS}...")
        # weights_only=False required: the upstream `weights_only=True` flag is
        # fragile on torch 2.6+ for this checkpoint.
        kokoro_state = torch.load(
            str(SRC_WEIGHTS), map_location="cpu", weights_only=False
        )

        top_keys = set(kokoro_state.keys())
        if top_keys != EXPECTED_MODULES:
            print(
                f"WARNING: unexpected top-level keys. "
                f"got={sorted(top_keys)} expected={sorted(EXPECTED_MODULES)}"
            )

        net: dict[str, dict] = {}
        total_params = 0
        per_component_params: dict[str, int] = {}

        for component, state_dict in kokoro_state.items():
            cleaned: dict = {}
            comp_params = 0
            for key, tensor in state_dict.items():
                clean_key = key.removeprefix("module.")
                cleaned[clean_key] = tensor
                comp_params += tensor.numel()
            net[component] = cleaned
            per_component_params[component] = comp_params
            total_params += comp_params
            print(
                f"  {component:14s} tensors={len(cleaned):4d}  "
                f"params={comp_params / 1e6:7.3f}M"
            )

        checkpoint = {"net": net}
        torch.save(checkpoint, str(OUT_WEIGHTS))
        print(f"\nSaved converted weights: {OUT_WEIGHTS}")
        print(
            "  Format: {'net': {bert, bert_encoder, predictor, decoder, text_encoder}}"
        )
        print(f"  Total parameters: {total_params / 1e6:.3f}M")

    # Always ensure config.json is copied (idempotent, cheap)
    if OUT_CONFIG.exists() and not force:
        print(f"{OUT_CONFIG} already exists, skipping config copy")
    else:
        shutil.copy2(SRC_CONFIG, OUT_CONFIG)
        print(f"Copied config: {SRC_CONFIG} -> {OUT_CONFIG}")

    # Verification: reload and check invariants
    print("\nVerifying output...")
    ckpt = torch.load(str(OUT_WEIGHTS), map_location="cpu", weights_only=False)

    assert "net" in ckpt, f"expected top-level key 'net', got {list(ckpt.keys())}"
    net = ckpt["net"]

    modules = set(net.keys())
    assert modules == EXPECTED_MODULES, (
        f"expected modules {sorted(EXPECTED_MODULES)}, got {sorted(modules)}"
    )
    assert len(modules) == 5, f"expected 5 modules under 'net', got {len(modules)}"

    total = 0
    per_component: dict[str, int] = {}
    for name, sd in net.items():
        comp_total = 0
        for k, v in sd.items():
            assert not k.startswith("module."), (
                f"key {name}.{k} still has module. prefix"
            )
            comp_total += v.numel()
        per_component[name] = comp_total
        total += comp_total

    print("  Per-component parameters:")
    for name in sorted(per_component):
        print(f"    {name:14s} {per_component[name] / 1e6:7.3f}M")
    print(f"  Total parameters: {total / 1e6:.3f}M")

    total_m = total / 1e6
    assert 80.0 <= total_m <= 83.0, (
        f"total params {total_m:.3f}M not in expected 80-83M range"
    )
    print("Verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Kokoro-82M weights to StyleTTS2-compatible format"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate outputs even if they already exist",
    )
    args = parser.parse_args()
    convert(force=args.force)


if __name__ == "__main__":
    main()
