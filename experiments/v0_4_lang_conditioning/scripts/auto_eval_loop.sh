#!/usr/bin/env bash
# auto_eval_loop.sh — watch ckpt dir during training; synthesize audio sample
# from each new epoch ckpt as it's saved. Run in a separate tmux pane next to
# the training process. Output WAVs accumulate in <log_dir>/audio_samples/.
#
# Workflow:
#   tmux new -s train       # pane 1: training
#   tmux new -s eval        # pane 2: this script
#
# Then in pane 2:
#   bash auto_eval_loop.sh
#
# Catches the v0.4 "radio tuning" failure mode early — if epoch 1's WAV is
# garbled, kill the training pane (~$3.75 spent) instead of running the full
# 25h ($38). Inference is GPU-fast (~5s per sample on A100).

set -euo pipefail

LOG_DIR="${LOG_DIR:-/workspace/bol_run/StyleTTS2/logs/kokoro-marathi-v0_4}"
CKPT_PATTERN="$LOG_DIR/epoch_2nd_*.pth"
SAMPLES_DIR="$LOG_DIR/audio_samples"

EXP_DIR=/workspace/bol_run/experiments/v0_4_lang_conditioning
STYLETTS2_DIR=/workspace/bol_run/StyleTTS2
CONFIG="$EXP_DIR/configs/config_marathi_v0_4_langcond.yml"
SYNTH="$EXP_DIR/scripts/synth_v0_4.py"
VOICEPACK="${VOICEPACK:-/workspace/bol_run/mf_mukta.bin}"  # upload one before launch

# Sanity (script writers) — validate config / voicepack present at start
[[ -f "$CONFIG" ]]    || { echo "ERROR: config not at $CONFIG"; exit 1; }
[[ -f "$SYNTH" ]]     || { echo "ERROR: synth script not at $SYNTH"; exit 1; }
[[ -f "$VOICEPACK" ]] || { echo "ERROR: voicepack not at $VOICEPACK; upload first"; exit 1; }

mkdir -p "$SAMPLES_DIR"

# Persistent env (in case pane started fresh)
# Default to /root/.local for container-disk installs (faster than MooseFS).
# Override PYTHONUSERBASE in caller env if you've installed to /workspace/.local.
export PYTHONUSERBASE="${PYTHONUSERBASE:-/root/.local}"
export PATH="${PYTHONUSERBASE}/bin:$PATH"

echo "[auto_eval] watching $LOG_DIR for new ckpts..."
echo "[auto_eval] writing samples to $SAMPLES_DIR"
echo "[auto_eval] using voicepack $VOICEPACK"
echo ""

# Loop until killed
while true; do
    for ckpt in $CKPT_PATTERN; do
        [[ -e "$ckpt" ]] || continue  # glob didn't match anything
        epoch=$(basename "$ckpt" .pth | sed 's/epoch_2nd_//')
        wav="$SAMPLES_DIR/epoch_${epoch}.wav"
        [[ -f "$wav" ]] && continue  # already done

        # Skip if ckpt is still being written (size changing).
        # Stage 2 ckpts are ~1.8GB so the save can take 30-60s.
        size1=$(stat -c%s "$ckpt" 2>/dev/null || echo 0)
        sleep 10
        size2=$(stat -c%s "$ckpt" 2>/dev/null || echo 0)
        if [[ "$size1" != "$size2" ]] || [[ "$size1" -lt 100000000 ]]; then
            echo "[auto_eval] $ckpt still being written (size $size1 → $size2); skipping for now"
            continue
        fi

        echo ""
        echo "[auto_eval] new ckpt: $ckpt → $wav"
        # cd into StyleTTS2 dir so the v0.4 config's relative ASR/F0 paths resolve.
        # synth_v0_4.py imports models.py which calls _load_config('Utils/ASR/config.yml').
        (cd "$STYLETTS2_DIR" && time python "$SYNTH" \
            --ckpt      "$ckpt" \
            --voicepack "$VOICEPACK" \
            --config    "$CONFIG" \
            --output    "$wav" \
            --device    cuda) \
        || { echo "[auto_eval] synth FAILED for $ckpt — see error above"; continue; }

        # Quick stats so user can scan log without listening
        python -c "
import soundfile as sf
data, sr = sf.read('$wav')
peak = abs(data).max()
rms = (data**2).mean()**0.5
print(f'  duration {len(data)/sr:.2f}s  peak {peak:.3f}  rms {rms:.3f}'
      f'  {\"⚠ HIGH PEAK (clipping?)\" if peak > 0.97 else \"\"}')
"

        echo "[auto_eval] download from your laptop with:"
        echo "  scp -P \$PORT root@\$IP:$wav ~/Downloads/v0_4_epoch_${epoch}.wav"
    done
    sleep 60
done
