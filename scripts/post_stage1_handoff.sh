#!/usr/bin/env bash
# Post-Stage-1 handoff: verify Stage 1 done → test latest ckpt locally → kick off Stage 2 on pod.
#
# Idempotent: skips the ckpt download if it's already on Mac, refuses to launch
# Stage 2 if it's already running.
#
# Requires env vars: BOL_REPO, BOL_CHECKPOINTS, POD_HOST, POD_PORT
#
# Usage:
#   bash ${BOL_REPO}/scripts/post_stage1_handoff.sh
#
# Log: ${BOL_CHECKPOINTS}/post_stage1_handoff.log

set -u  # fail on unset vars; no -e so we can handle errors inline
LOG="${BOL_CHECKPOINTS}/post_stage1_handoff.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date): starting post-stage-1 handoff ==="

POD="root@${POD_HOST}"
PORT=${POD_PORT}
POD_LOGDIR="/workspace/bol_run/StyleTTS2/logs/kokoro-marathi"
MAC_CKPT_DIR="${BOL_CHECKPOINTS}"
VENV="${BOL_REPO}/.venv"
SEMIDARK_SCRIPTS="${BOL_REPO}/scripts/upstream"
MR_SCRIPTS="${BOL_REPO}/scripts"

# ── STEP 1: verify Stage 1 is done ────────────────────────────────────────
echo; echo "--- STEP 1: pod state ---"
POD_STATE=$(ssh -p "$PORT" -o ConnectTimeout=10 "$POD" '
  ls -1 /workspace/bol_run/StyleTTS2/logs/kokoro-marathi/epoch_1st_*.pth 2>/dev/null | sort | tail -5
  echo "---"
  pgrep -af "accelerate launch|train_first|train_second" | grep -v grep || echo "NO_TRAIN_PROC"
  echo "---"
  tail -3 /workspace/bol_run/StyleTTS2/logs/kokoro-marathi/train.log
')
echo "$POD_STATE"

# Parse: last line of ckpt list, presence of NO_TRAIN_PROC
LATEST_CKPT=$(echo "$POD_STATE" | grep -E 'epoch_1st_[0-9]+\.pth' | tail -1)
LATEST_N=$(echo "$LATEST_CKPT" | sed -E 's|.*epoch_1st_0+([0-9]+)\.pth|\1|' | grep -oE '[0-9]+' | head -1)

if [[ -z "$LATEST_N" ]]; then
  echo "[abort] no checkpoint found on pod; is training still initializing?"
  exit 2
fi
echo "[ok] latest pod epoch = $LATEST_N"

if echo "$POD_STATE" | grep -q "NO_TRAIN_PROC"; then
  echo "[ok] no active training process"
  TRAIN_DONE=1
else
  echo "[warn] a training process IS still running — Stage 1 not done yet"
  TRAIN_DONE=0
fi

if [[ "$TRAIN_DONE" -ne 1 ]] || [[ "$LATEST_N" -lt 9 ]]; then
  echo "[abort] Stage 1 not complete (latest=$LATEST_N, train_done=$TRAIN_DONE). Re-run this script later."
  exit 3
fi

# ── STEP 2: download latest ckpt ──────────────────────────────────────────
echo; echo "--- STEP 2: downloading epoch $LATEST_N ---"
LOCAL_CKPT="$MAC_CKPT_DIR/epoch_1st_$(printf '%05d' "$LATEST_N").pth"
if [[ -s "$LOCAL_CKPT" ]]; then
  echo "[skip] $LOCAL_CKPT already exists"
else
  scp -P "$PORT" "$POD:$POD_LOGDIR/epoch_1st_$(printf '%05d' "$LATEST_N").pth" "$LOCAL_CKPT"
  echo "[ok] scp done"
fi

# ── STEP 3: convert + extract voicepack + synthesize ──────────────────────
echo; echo "--- STEP 3: test battery ---"
source "$VENV/bin/activate"

KOKORO_OUT="$MAC_CKPT_DIR/kokoro_mr_final.pth"
python - <<PY
import sys
sys.path.insert(0, "$SEMIDARK_SCRIPTS")
from test_inference import convert_checkpoint
convert_checkpoint("$LOCAL_CKPT", "$KOKORO_OUT")
PY

VOICE_OUT="$MAC_CKPT_DIR/voices/mf_asha_final.pt"
mkdir -p "$(dirname "$VOICE_OUT")"
python "$SEMIDARK_SCRIPTS/extract_voicepack.py" \
  --model "$LOCAL_CKPT" \
  --audio-dir ${BOL_REPO}/dataset/audio/rasa_female \
  --output "$VOICE_OUT" \
  --num-samples 200 --device cpu

python "$MR_SCRIPTS/inference_mac_mr.py" \
  --model "$KOKORO_OUT" \
  --voicepack "$VOICE_OUT" \
  --output-dir "$MAC_CKPT_DIR/test_output_final/asha"

echo "[ok] test WAVs at $MAC_CKPT_DIR/test_output_final/asha/"

# ── STEP 4: kick off Stage 2 ──────────────────────────────────────────────
echo; echo "--- STEP 4: launching Stage 2 ---"

# scp Marathi TB utils so Stage 2 logs Marathi previews (optional but nice)
scp -P "$PORT" "$MR_SCRIPTS/kokoro_tb_utils_mr.py" \
  "$POD:/workspace/bol_run/StyleTTS2/kokoro_tb_utils.py"

# guard: don't double-launch
ALREADY=$(ssh -p "$PORT" "$POD" 'pgrep -af "accelerate launch|train_second" | grep -v grep || true')
if [[ -n "$ALREADY" ]]; then
  echo "[abort] Stage 2 already running? $ALREADY"
  exit 4
fi

ssh -p "$PORT" "$POD" 'cd /workspace && rm -f training_s2.log && STAGE=2 setsid nohup ./launch_training.sh > training_s2.log 2>&1 < /dev/null & disown'

echo "[ok] Stage 2 launched. Verifying in 90s..."
sleep 90

ssh -p "$PORT" "$POD" '
  echo "--- process ---"
  pgrep -af "accelerate launch|train_second" | grep -v grep
  echo "--- log tail ---"
  tail -20 /workspace/bol_run/StyleTTS2/logs/kokoro-marathi/train.log
'

echo; echo "=== $(date): handoff complete ==="
echo "Stage 2 expected Mel start ~0.43 (if ~7.5, Stage 1 didn't load — abort and debug)"
echo "Log: $LOG"
