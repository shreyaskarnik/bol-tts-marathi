#!/usr/bin/env bash
# Push bol-tts-marathi model artifacts to HuggingFace Hub.
#
# Expects env vars:
#   BOL_REPO        — path to this repo
#   HF_USER         — your HF username / org
#   HF_MODEL_REPO   — default: $HF_USER/bol-tts-marathi
#   HF_SPACE_REPO   — default: $HF_USER/bol-tts-marathi (space), same name OK
#   BOL_CHECKPOINTS — where the .pth + .pt files live
#
# Requires: `huggingface-cli login` done once first.

set -eu

: "${BOL_REPO:?set BOL_REPO to repo root}"
: "${HF_USER:?set HF_USER}"
: "${BOL_CHECKPOINTS:?set BOL_CHECKPOINTS to your checkpoints dir}"

MODEL_REPO="${HF_MODEL_REPO:-$HF_USER/bol-tts-marathi}"
SPACE_REPO="${HF_SPACE_REPO:-$HF_USER/bol-tts-marathi}"

# Staging area
STAGE="$BOL_REPO/hf_model"
mkdir -p "$STAGE/voices"

# 1. Copy model card (already in repo)
cp "$BOL_REPO/hf_model/README.md" "$STAGE/README.md" 2>/dev/null || true
# Inject actual username into the template
sed -i '' "s|<your-user>|$HF_USER|g" "$STAGE/README.md" 2>/dev/null || \
  sed -i "s|<your-user>|$HF_USER|g" "$STAGE/README.md"

# 2. Copy inference config
cp "$BOL_REPO/configs/config_mr.json" "$STAGE/config.json"

# 3. Copy Kokoro-format model weights
test -s "$BOL_CHECKPOINTS/kokoro_mr_final.pth" || {
  echo "[abort] $BOL_CHECKPOINTS/kokoro_mr_final.pth not found."
  echo "        Run scripts/inference_mac_mr.py once (it produces this file) or"
  echo "        convert via scripts/upstream/test_inference.py convert_checkpoint()."
  exit 1
}
cp "$BOL_CHECKPOINTS/kokoro_mr_final.pth" "$STAGE/kokoro-mr-v1_0.pth"

# 4. Copy the four named voicepacks
for v in mf_asha mm_vivek mf_mukta mm_dnyanesh; do
  src="$BOL_CHECKPOINTS/voices/${v}_final.pt"
  if [[ -s "$src" ]]; then
    cp "$src" "$STAGE/voices/${v}.pt"
  else
    echo "[warn] voices/${v}_final.pt not found — skipping ${v}"
  fi
done

# 5. Copy per-voice speed config alongside
cp "$BOL_REPO/configs/voice_speeds.json" "$STAGE/voice_speeds.json"

# 6. Push model repo
echo; echo "=== pushing model → $MODEL_REPO ==="
huggingface-cli upload "$MODEL_REPO" "$STAGE" . \
  --repo-type model \
  --commit-message "publish bol-tts-marathi v0.1 (Stage 2 finetune of Kokoro-82M on Rasa + IV-R Marathi)"

# 7. Push space repo (optional — comment out if you don't want the Gradio demo up)
if [[ "${PUSH_SPACE:-1}" == "1" ]]; then
  SPACE_STAGE="$BOL_REPO/hf_space"
  # patch the repo_id in app.py so the Space uses YOUR model repo, not the placeholder
  sed -i.bak "s|<your-user>/bol-tts-marathi|$MODEL_REPO|g" "$SPACE_STAGE/app.py"
  sed -i.bak "s|<your-user>/bol-tts-marathi|$MODEL_REPO|g" "$SPACE_STAGE/README.md"
  rm -f "$SPACE_STAGE"/*.bak
  echo; echo "=== pushing space → $SPACE_REPO ==="
  huggingface-cli upload "$SPACE_REPO" "$SPACE_STAGE" . \
    --repo-type space \
    --commit-message "deploy bol-tts-marathi Gradio demo"
fi

echo; echo "=== done ==="
echo "Model: https://huggingface.co/$MODEL_REPO"
[[ "${PUSH_SPACE:-1}" == "1" ]] && echo "Space: https://huggingface.co/spaces/$SPACE_REPO"
