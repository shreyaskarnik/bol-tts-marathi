#!/usr/bin/env bash
# Make sure torch + transformers + monotonic_align are importable. On RunPod
# pods that already ship them system-wide (current Kokoro pod image), this is
# a no-op. On a fresh pod, install to /root/.local — container disk is fast,
# /workspace is glacial MooseFS (~22 KB/s for pip), so we accept the
# pod-restart-wipe tradeoff.
#
# Idempotent — re-running detects existing installs and skips.
#
# Usage on the pod:
#     bash setup_pod_env.sh

set -euo pipefail

# Idempotency check FIRST — probe whatever Python sees right now (system,
# /root/.local, anywhere). If all three deps import with a torch 2.6 build,
# we're done. The previous version of this check inserted a hardcoded
# /workspace/.local path that broke when /workspace was wiped.
if python3 -c "
import torch, transformers, monotonic_align
assert torch.__version__.startswith('2.6'), f'wrong torch: {torch.__version__}'
" >/dev/null 2>&1; then
    echo "[setup_pod_env] deps already importable system-wide — skipping reinstall"
    python3 -c "import torch, transformers; print(f'                torch={torch.__version__}  transformers={transformers.__version__}')"
    exit 0
fi

# Fresh pod path — install to /root/.local (container disk, fast).
export PYTHONUSERBASE=/root/.local
export PATH=/root/.local/bin:$PATH
mkdir -p "$PYTHONUSERBASE"

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
echo "  export PYTHONUSERBASE=/root/.local"
echo "  export PATH=/root/.local/bin:\$PATH"
