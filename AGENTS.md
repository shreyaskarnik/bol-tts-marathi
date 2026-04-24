# AGENTS.md — briefing for coding agents working in this repo

## What this repo does

Marathi fine-tune of Kokoro-82M. We follow [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch)'s training recipe and add a Marathi overlay (symbol table, phonemizer language, TB utils, test sentences). See `README.md` for the layout and `docs/TRAINING_GUIDE.md` for the pipeline.

## What to read before editing

1. **`NOTICE`** — attribution requirements. Upstream code (kokoro, StyleTTS2, kokoro-deutsch) is used under its own licenses; don't strip attribution from docstrings that reference those projects.
2. **`docs/TROUBLESHOOTING.md`** — every gotcha we already hit. Before proposing a debugging idea, check that it isn't already resolved here.
3. **`docs/ARCHITECTURE.md`** — Marathi-specific invariants. In particular: `ɭ` MUST be at slot 144 of the Kokoro vocab; any rearrangement of `kokoro_symbols.py` or `configs/config_mr.json` vocab invalidates every trained checkpoint.

## Things that are easy to break

- **Symbol table alignment**: `training/kokoro_symbols.py` (used at training time by StyleTTS2) and `configs/config_mr.json` (used at inference time by Kokoro KModel) must agree on EVERY slot, not just `ɭ`. A mismatch produces silent phoneme drops at inference time.
- **Stage 2 needs `set_detect_anomaly(False)`**: semidark's upstream `StyleTTS2/train_second.py` has `torch.autograd.set_detect_anomaly(True)` hardcoded. It hangs the first forward pass on Kokoro-scale graphs. We disable it (env-gated or commented out). Don't re-enable by accident.
- **Stage 2 OOM at `joint_epoch`**: On single A100 80GB, `batch_size: 12` runs Stage 2 epochs 0–2 fine but OOMs at epoch 3 when adversarial losses engage. Use `batch_size: 8` plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **SSH pkill self-kill**: `pkill -f "train_second"` in an SSH one-liner matches the SSH's own bash shell (the command string contains "train_second") and kills the session. Prefer `kill <pid>` with specific PIDs, or `pkill -f "^python3 train_second"` with a command-anchor regex.
- **Paths in scripts**: scripts use `${BOL_REPO}`, `${BOL_CHECKPOINTS}`, `${POD_HOST}`, `${POD_PORT}` env vars. Don't inline absolute paths. Upstream semidark scripts (`extract_voicepack.py`, `test_inference.py`) are vendored under `scripts/upstream/` — scripts reference them via `${BOL_REPO}/scripts/upstream/`, not an external clone.

## Environment variables

| Variable | Purpose | Typical value |
|---|---|---|
| `BOL_REPO` | root of this clone | `$(pwd)` |
| `BOL_CHECKPOINTS` | where Mac-side .pth files live | `$BOL_REPO/checkpoints` |
| `POD_HOST`, `POD_PORT` | training pod SSH endpoint | pod-specific |
| `STYLETTS2_DETECT_ANOMALY` | enables autograd anomaly detection in Stage 2 (debug only) | unset (False) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` mandatory for Stage 2 | see above |

## Testing changes

No automated test suite yet. Before declaring a change safe:

1. Run `scripts/diagnostics/diagnose_nan.py` if you touched the symbol table or vocab.
2. Run `scripts/diagnostics/check_ipa_lengths.py` if you touched the G2P or added/removed training utterances.
3. Run `scripts/inference_mac_mr.py --text "नमस्कार"` end-to-end after any Kokoro-side change.

## Style

Follow semidark's style: concise docstrings, argparse CLIs, default paths set for a pod layout where these scripts were originally intended. Override points always exposed via `--flag` or env var.
