#!/usr/bin/env bash
# stage2_auto_monitor.sh — poll pod until Stage 2 finishes, run post-Stage-2
# work (ckpt download + voicepack extraction + test synthesis), then LOUDLY
# NOTIFY (macOS banner + terminal bell + log) to stop pod manually.
#
# Auto-stop path: set RUNPOD_POD_ID + RUNPOD_API_KEY and uncomment the curl
# block at the bottom of this script. Otherwise we just notify.
#
# Requires env vars: BOL_REPO, BOL_CHECKPOINTS, POD_HOST, POD_PORT
#
# Usage (detached, survives terminal close; caffeinate -dim keeps the Mac awake):
#   nohup caffeinate -dim bash ${BOL_REPO}/scripts/stage2_auto_monitor.sh \
#     > ${BOL_CHECKPOINTS}/stage2_monitor.out 2>&1 &
#   disown
#
# Stop it:
#   pgrep -f stage2_auto_monitor.sh | xargs kill

set -u
LOG="${BOL_CHECKPOINTS}/stage2_monitor.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

POD="root@${POD_HOST}"
PORT=${POD_PORT}
POLL=900  # 15 min
MAC_CKPT_DIR="${BOL_CHECKPOINTS}"
VENV="${BOL_REPO}/.venv"
SEMIDARK_SCRIPTS="${BOL_REPO}/scripts/upstream"
MR_SCRIPTS="${BOL_REPO}/scripts"

notify() {
  local msg="$1"
  osascript -e "display notification \"$msg\" with title \"bol-tts Stage 2\" sound name \"Ping\"" 2>/dev/null || true
  printf '\a'
  echo "[$(date)] NOTIFY: $msg"
}

echo "=== $(date): stage2 monitor starting, poll every ${POLL}s ==="
notify "Stage 2 monitor started. Will alert when training completes."

while true; do
  STATE=$(ssh -p "$PORT" -o ConnectTimeout=10 "$POD" '
    ls -1 /workspace/bol_run/StyleTTS2/logs/kokoro-marathi/epoch_2nd_*.pth 2>/dev/null | sort | tail -1
    echo "---"
    pgrep -f "train_second.py" | head -1 || echo "NO_TRAIN_PROC"
    echo "---"
    tail -1 /workspace/bol_run/StyleTTS2/logs/kokoro-marathi/train.log
  ' 2>&1) || { echo "[$(date)] ssh failed, retry next cycle"; sleep "$POLL"; continue; }

  LATEST=$(echo "$STATE" | grep -E 'epoch_2nd_[0-9]+\.pth' | tail -1)
  LATEST_N=$(echo "$LATEST" | sed -E 's|.*epoch_2nd_0+([0-9]+)\.pth|\1|' | grep -oE '[0-9]+' | head -1)
  NO_PROC=$(echo "$STATE" | grep -c "NO_TRAIN_PROC" || echo 0)

  echo "[$(date)] latest epoch_2nd N=${LATEST_N:-none}  no_proc=${NO_PROC}"

  # Stage 2 is done if: final epoch (9) exists AND no train_second process
  if [[ "${LATEST_N:-0}" -ge 9 && "$NO_PROC" -ge 1 ]]; then
    echo "[$(date)] STAGE 2 COMPLETE — acting"
    break
  fi
  sleep "$POLL"
done

notify "Stage 2 finished — downloading final ckpt + extracting voicepacks..."

# ── post-Stage-2: download final ckpt + voicepacks ──────────────────────────
FINAL_CKPT="$MAC_CKPT_DIR/epoch_2nd_00009.pth"
scp -P "$PORT" "$POD:/workspace/bol_run/StyleTTS2/logs/kokoro-marathi/epoch_2nd_00009.pth" "$FINAL_CKPT" || {
  notify "SCP of final ckpt failed — check pod manually"
  exit 1
}

source "$VENV/bin/activate"

# Stage-1 checkpoint was already downloaded earlier; reuse as style-encoder source
STAGE1_CKPT="$MAC_CKPT_DIR/epoch_1st_00009.pth"
[[ -s "$STAGE1_CKPT" ]] || scp -P "$PORT" "$POD:/workspace/bol_run/StyleTTS2/logs/kokoro-marathi/epoch_1st_00009.pth" "$STAGE1_CKPT"

mkdir -p "$MAC_CKPT_DIR/voices"

for gender in female male; do
  voice=$([ "$gender" = female ] && echo mf_asha || echo mm_vivek)
  python "$SEMIDARK_SCRIPTS/extract_voicepack.py" \
    --model "$FINAL_CKPT" \
    --style-encoder-model "$STAGE1_CKPT" \
    --audio-dir "${BOL_REPO}/dataset/audio/rasa_${gender}" \
    --output "$MAC_CKPT_DIR/voices/${voice}_final.pt" \
    --num-samples 200 --device cpu
done

# ── convert + synth for ear-test ────────────────────────────────────────────
KOKORO_OUT="$MAC_CKPT_DIR/kokoro_mr_final.pth"
python - <<PY
import sys
sys.path.insert(0, "$SEMIDARK_SCRIPTS")
from test_inference import convert_checkpoint
convert_checkpoint("$FINAL_CKPT", "$KOKORO_OUT")
PY

for voice in mf_asha mm_vivek; do
  python "$MR_SCRIPTS/inference_mac_mr.py" \
    --model "$KOKORO_OUT" \
    --voicepack "$MAC_CKPT_DIR/voices/${voice}_final.pt" \
    --output-dir "$MAC_CKPT_DIR/test_output_final/$voice"
done

# ── STOP POD ────────────────────────────────────────────────────────────────
# Option 1: runpodctl (install via brew or `pip install runpod`, configure with API key)
# if command -v runpodctl >/dev/null; then
#   runpodctl stop pod "$RUNPOD_POD_ID" && notify "Pod stopped." && exit 0
# fi
#
# Option 2: REST API (set RUNPOD_POD_ID + RUNPOD_API_KEY)
# if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
#   curl -sS -X POST "https://api.runpod.io/v2/${RUNPOD_POD_ID}/stop" \
#     -H "Authorization: Bearer $RUNPOD_API_KEY" && notify "Pod stopped via API." && exit 0
# fi

# Fallback: loud notification repeated 3x over 6 min
for i in 1 2 3; do
  notify "STAGE 2 DONE — STOP POD MANUALLY (runpod.io dashboard). Voices + tests written to checkpoints/test_output_final/. Burn is $1.506/hr."
  [[ $i -lt 3 ]] && sleep 120
done

echo "[$(date)] monitor finished"
