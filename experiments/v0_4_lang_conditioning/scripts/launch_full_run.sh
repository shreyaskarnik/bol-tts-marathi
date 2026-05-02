#!/usr/bin/env bash
# launch_full_run.sh — start v0.4 full Stage 2.5 training + auto-eval loop
# as nohup'd background processes. Both survive SSH disconnect (RunPod has no
# tmux out of the box; nohup is the equivalent for "fire-and-forget").
#
# Usage on the pod:
#     bash launch_full_run.sh
#
# Then disconnect SSH freely. Reconnect anytime to:
#     tail -f $TRAIN_LOG    # training progress
#     tail -f $EVAL_LOG     # eval-loop progress (synth after each epoch)
#     ls $SAMPLES_DIR       # list synthesized WAVs

set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────────
RUN_DIR=/workspace/bol_run
EXP_DIR="$RUN_DIR/experiments/v0_4_lang_conditioning"
STYLETTS2_DIR="$RUN_DIR/StyleTTS2"
LOG_DIR="$STYLETTS2_DIR/logs/kokoro-marathi-v0_4"
TRAIN_LOG="$LOG_DIR/training.log"
EVAL_LOG="$LOG_DIR/eval_loop.log"
SAMPLES_DIR="$LOG_DIR/audio_samples"
PID_FILE="$LOG_DIR/launch_pids.txt"
CONFIG="$EXP_DIR/configs/config_marathi_v0_4_langcond.yml"

mkdir -p "$LOG_DIR" "$SAMPLES_DIR"

# ─── Pre-flight checks ──────────────────────────────────────────────────────
[[ -f "$CONFIG" ]] || { echo "ERROR: missing $CONFIG"; exit 1; }
[[ -f "$EXP_DIR/scripts/auto_eval_loop.sh" ]] || { echo "ERROR: missing auto_eval_loop.sh"; exit 1; }
[[ -d "$STYLETTS2_DIR" ]] || { echo "ERROR: missing $STYLETTS2_DIR"; exit 1; }

# Voicepack for eval (path mirrors auto_eval_loop.sh's default)
VOICEPACK="${VOICEPACK:-$RUN_DIR/mf_mukta.bin}"
[[ -f "$VOICEPACK" ]] || {
    echo "ERROR: voicepack not found at $VOICEPACK"
    echo "  upload one before launch:"
    echo "    scp -P \$PORT mf_mukta.bin root@\$IP:$VOICEPACK"
    exit 1
}

# Manifests must exist (run build_v0_4_manifest.py first)
TRAIN_MANIFEST=$(grep -E '^\s*train_data:' "$CONFIG" | head -1 | awk -F'"' '{print $2}')
TRAIN_MANIFEST_RESOLVED="$STYLETTS2_DIR/$TRAIN_MANIFEST"
[[ -f "$TRAIN_MANIFEST_RESOLVED" ]] || {
    echo "ERROR: train manifest not at $TRAIN_MANIFEST_RESOLVED (config says $TRAIN_MANIFEST)"
    echo "  run build_v0_4_manifest.py first"
    exit 1
}
echo "[launch] config: $CONFIG"
echo "[launch] train manifest: $TRAIN_MANIFEST_RESOLVED ($(wc -l < "$TRAIN_MANIFEST_RESOLVED") rows)"

# ─── Persistent env (idempotent) ────────────────────────────────────────────
export PYTHONUSERBASE=/workspace/.local
export PATH=/workspace/.local/bin:$PATH
bash "$EXP_DIR/scripts/setup_pod_env.sh"

# Training-specific env vars
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STYLETTS2_DETECT_ANOMALY=0

# Already-running guard — refuse to double-launch
if [[ -f "$PID_FILE" ]]; then
    while read -r pid name; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $name (PID $pid) already running. Stop it first:"
            echo "  kill $pid    # or: bash $EXP_DIR/scripts/stop_full_run.sh"
            exit 1
        fi
    done < "$PID_FILE"
fi
> "$PID_FILE"

# ─── Launch training (nohup'd) ──────────────────────────────────────────────
echo ""
echo "[launch] starting training (logs → $TRAIN_LOG)"
cd "$STYLETTS2_DIR"
nohup accelerate launch --mixed_precision=bf16 --num_processes=1 \
    train_second.py --config_path "$CONFIG" \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
disown $TRAIN_PID
echo "$TRAIN_PID train" >> "$PID_FILE"
echo "[launch] training PID: $TRAIN_PID"

# Sleep briefly to verify training process didn't die immediately
sleep 5
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "ERROR: training process died within 5s. Check $TRAIN_LOG:"
    tail -20 "$TRAIN_LOG"
    exit 1
fi

# ─── Launch eval loop (nohup'd, with VOICEPACK env) ─────────────────────────
echo ""
echo "[launch] starting eval loop (logs → $EVAL_LOG)"
LOG_DIR="$LOG_DIR" VOICEPACK="$VOICEPACK" \
    nohup bash "$EXP_DIR/scripts/auto_eval_loop.sh" \
    > "$EVAL_LOG" 2>&1 < /dev/null &
EVAL_PID=$!
disown $EVAL_PID
echo "$EVAL_PID eval" >> "$PID_FILE"
echo "[launch] eval-loop PID: $EVAL_PID"

# ─── Final summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[launch] DONE. Both processes nohup'd; SSH disconnect-safe."
echo ""
echo "Monitor:"
echo "  tail -f $TRAIN_LOG"
echo "  tail -f $EVAL_LOG"
echo "  ls $SAMPLES_DIR     # WAVs accumulate here as epochs save"
echo ""
echo "Stop both processes:"
echo "  bash $EXP_DIR/scripts/stop_full_run.sh"
echo ""
echo "First epoch eval lands in ~2.5h. Then SCP the WAV down:"
echo "  scp -P \$PORT root@\$IP:$SAMPLES_DIR/epoch_00000.wav ~/Downloads/"
echo "============================================================"
