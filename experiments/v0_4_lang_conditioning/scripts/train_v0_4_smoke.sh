#!/usr/bin/env bash
# v0.4 architecture smoke test launcher (~5-10 min on A100).
#
# Pre-flight requirements (must already be on the pod):
#   1. kokoro-deutsch/StyleTTS2 fork checked out to v0_4_lang_conditioning branch
#      (or main + 01_lang_conditioning.patch applied)
#   2. logs/kokoro-marathi/epoch_2nd_v0_2_FINAL.pth (v0.2 final ckpt)
#   3. dataset/audio/springlab_mr/ (the 500 wavs the smoke manifest references)
#   4. Smoke manifest files (data/train_list_smoke.txt + val_list_smoke.txt)
#      — generate locally via build_smoke_manifest.py and rsync to pod, OR
#        regenerate on pod with the same script.
#
# Validation criteria (look for these in the training log):
#   1. NO "NaN detected in bert_dur" prints
#   2. predictor_encoder norm stays > 0.5 across the epoch
#      (extract via scripts/extract_voicepacks_mr.py at end of epoch 1)
#   3. lang_embedding has nonzero gradient (manually inspect via:
#        torch.load('logs/kokoro-marathi-v0_4-smoke/epoch_2nd_00000.pth')['net']['bert']['module.lang_embedding.weight'])
#   4. Validation loss line appears at end of epoch (no crash)
#
# Usage:
#   bash train_v0_4_smoke.sh

set -euo pipefail

EXPERIMENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CONFIG="$EXPERIMENT_DIR/configs/config_marathi_v0_4_smoke.yml"
TRAIN_MANIFEST="$EXPERIMENT_DIR/data/train_list_smoke.txt"
VAL_MANIFEST="$EXPERIMENT_DIR/data/val_list_smoke.txt"

# Pre-flight checks
[[ -f "$CONFIG" ]] || { echo "ERROR: missing $CONFIG"; exit 1; }
[[ -f "$TRAIN_MANIFEST" ]] || { echo "ERROR: missing $TRAIN_MANIFEST — run build_smoke_manifest.py first"; exit 1; }
[[ -f "$VAL_MANIFEST" ]] || { echo "ERROR: missing $VAL_MANIFEST — run build_smoke_manifest.py first"; exit 1; }

# Disable upstream's hardcoded autograd anomaly detect (slow + log-spammy).
# This env var is consumed by the patched train_second.py:10 in the v0.2 fork.
export STYLETTS2_DETECT_ANOMALY=0

# Single-GPU memory tuning carried over from v0.2 — required for bs=8.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(git -C "$EXPERIMENT_DIR/../../../kokoro-deutsch/StyleTTS2" rev-parse --show-toplevel 2>/dev/null || echo /workspace/bol_run/StyleTTS2)"

echo "[v0.4 smoke] config: $CONFIG"
echo "[v0.4 smoke] cwd:    $(pwd)"
echo "[v0.4 smoke] branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

STAGE=2 CONFIG="$CONFIG" ./launch_training.sh 2>&1 | tee "logs/kokoro-marathi-v0_4-smoke/training.log"

echo ""
echo "[v0.4 smoke] DONE. Inspect:"
echo "  logs/kokoro-marathi-v0_4-smoke/epoch_2nd_00000.pth   — saved ckpt"
echo "  logs/kokoro-marathi-v0_4-smoke/training.log         — full log"
echo ""
echo "Quick gradient-flow check on the saved ckpt:"
cat <<'PY'
python -c "
import torch
ck = torch.load('logs/kokoro-marathi-v0_4-smoke/epoch_2nd_00000.pth', map_location='cpu')
le = ck['net']['bert']['module.lang_embedding.weight']
print('lang_embedding norm per row:', le.norm(dim=1).tolist())
print('lang_embedding diff (mr vs en):', (le[0] - le[1]).abs().mean().item())
print('  (>0.0001 means the two rows have started to diverge — wiring works)')
"
PY
