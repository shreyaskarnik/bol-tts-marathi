#!/usr/bin/env bash
# Persist Python deps to /workspace/.local so they survive pod stop/restart.
# /workspace is RunPod's network-attached persistent volume; container disk
# (where pip installs default to) gets wiped on every pod restart.
#
# Idempotent — re-running detects existing installs and skips. Call this
# at the top of any v0.4 launch script before invoking torch/transformers.
#
# Usage on the pod:
#     bash setup_pod_env.sh
#     source <(grep -E '^export' setup_pod_env.sh)   # also: shell-local PATH

set -euo pipefail

export PYTHONUSERBASE=/workspace/.local
export PATH=/workspace/.local/bin:$PATH

mkdir -p "$PYTHONUSERBASE"

# Idempotency check — if torch 2.6 + transformers + monotonic_align all import,
# we're done. (Probing all three since one of them missing would still break.)
if python -c "
import sys
sys.path.insert(0, '$PYTHONUSERBASE/lib/python3.11/site-packages')
import torch, transformers, monotonic_align
assert torch.__version__.startswith('2.6'), f'wrong torch: {torch.__version__}'
" >/dev/null 2>&1; then
    echo "[setup_pod_env] deps already in $PYTHONUSERBASE — skipping reinstall"
    echo "                torch + transformers + monotonic_align all importable"
    exit 0
fi

echo "[setup_pod_env] installing deps to $PYTHONUSERBASE (~3-5 min)"

# 1. torch 2.6.0 from cu124 index — must be first so other compiled extensions
#    (monotonic_align) bind against the right ABI.
pip install --user --force-reinstall --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0

# 2. numpy<2 — librosa/numba chain isn't numpy-2 clean; force the downgrade
#    since pip won't auto-downgrade after torch's transitive numpy 2.x install.
pip install --user --force-reinstall --no-cache-dir 'numpy<2'

# 3. Remaining python deps — transformers<5 to dodge masking_utils ONNX bug
#    encountered in v0.2 export. tensorboard for train logs.
pip install --user --no-cache-dir \
    'transformers<5' \
    librosa soundfile munch accelerate \
    einops einops-exts \
    pandas tqdm pydub matplotlib nltk \
    tensorboard pyyaml \
    misaki \
    'datasets<3.0'

# 4. monotonic_align — MUST install last (after torch 2.6) so its CUDA
#    extension compiles against the correct headers. Skipping this is what
#    caused "CUDA illegal memory access" in the v0.2 prep.
pip install --user --force-reinstall --no-cache-dir \
    git+https://github.com/resemble-ai/monotonic_align.git

echo ""
echo "[setup_pod_env] DONE. deps in $PYTHONUSERBASE"
echo ""
echo "Add to ~/.bashrc (or every fresh shell) so future pod sessions find these:"
echo "  export PYTHONUSERBASE=/workspace/.local"
echo "  export PATH=/workspace/.local/bin:\$PATH"
