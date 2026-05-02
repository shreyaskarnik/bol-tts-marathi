#!/usr/bin/env bash
# stop_full_run.sh — kill the v0.4 full training + eval-loop processes
# launched by launch_full_run.sh. Reads PIDs from launch_pids.txt.
#
# Usage on the pod:
#     bash stop_full_run.sh
#
# To abort because epoch-1 audio sounds like radio tuning:
#     bash stop_full_run.sh
#     # (cost-bounded at ~$3.75 of A100 spent)

set -euo pipefail

LOG_DIR=/workspace/bol_run/StyleTTS2/logs/kokoro-marathi-v0_4
PID_FILE="$LOG_DIR/launch_pids.txt"

if [[ ! -f "$PID_FILE" ]]; then
    echo "no PID file at $PID_FILE — nothing to stop"
    exit 0
fi

while read -r pid name; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "stopping $name (PID $pid)..."
        kill -TERM "$pid"
        sleep 2
        # Force-kill if still alive
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid"
        fi
    else
        echo "$name (PID $pid) not running"
    fi
done < "$PID_FILE"

# Also catch any child train_second.py processes (accelerate spawns one)
ps -eo pid,cmd | grep -v grep | grep -E '[t]rain_second\.py|[a]uto_eval_loop\.sh' \
    | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true

> "$PID_FILE"
echo "done."
