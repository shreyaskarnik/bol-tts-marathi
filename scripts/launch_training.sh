#!/usr/bin/env bash
# launch_training.sh — RunPod 4090 driver for Marathi Kokoro-82M fine-tune
#
# Usage:
#   From Mac: scp -P <port> launch_training.sh bol_training_v5.tar.gz bol_data_v2.tar.gz CHECKSUMS.txt root@<ip>:/workspace/
#   Then on pod:
#     chmod +x launch_training.sh
#     STAGE=setup ./launch_training.sh   # extracts + installs deps + sanity-checks, no training
#     STAGE=1 ./launch_training.sh 2>&1 | tee training_s1.log
#     STAGE=2 ./launch_training.sh 2>&1 | tee training_s2.log
#     STAGE=both ./launch_training.sh    # runs 1 then 2 sequentially
#
# Env overrides:
#   CODE_TAR     — code bundle path (auto-detect newest bol_training_v*.tar.gz)
#   DATA_TAR     — data bundle path (auto-detect newest bol_data_v*.tar.gz)
#   RUN_DIR      — extraction target dir (default: bol_run)
#   CONFIG       — config path relative to StyleTTS2/ (default: ../configs/config_marathi_ft.yml)
#   STAGE        — setup | 1 | 2 | both  (default: setup)
#   SKIP_TRAIN   — legacy alias for STAGE=setup when set to 1
#   PY           — python interpreter (default: python3)
#
# Critical lessons encoded:
#   - torch 2.6.0 + torchaudio 2.6.0 + torchvision 0.21.0 force-installed from cu124 index
#     (avoids WavLM CVE-2025-32434 reject by transformers; avoids cu126/cu128 driver mismatches).
#   - monotonic_align re-installed AFTER torch upgrade (avoids CUDA illegal memory access).
#   - tar --no-same-owner (Mac-built tarballs fail chown on RunPod network volume).
#   - No sudo on RunPod — use $SUDO which is empty when running as root.
#   - apt update may fail on flaky nvidia mirror — treat as warning.

set -euo pipefail

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

RUN_DIR="${RUN_DIR:-bol_run}"
CONFIG="${CONFIG:-../configs/config_marathi_ft.yml}"
STAGE="${STAGE:-setup}"
PY="${PY:-python3}"

# Legacy alias: SKIP_TRAIN=1 means the old setup-only behavior.
if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    STAGE="setup"
fi

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

warn() {
    printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2
}

die() {
    printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2
    exit 1
}

# Detect sudo requirement. On RunPod we're root with no sudo binary.
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    SUDO=""
else
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        die "not root and no sudo available"
    fi
fi

# Auto-detect tarballs if not specified. Pick the newest matching file.
pick_newest() {
    local pattern="$1"
    local found
    # shellcheck disable=SC2012
    found=$(ls -1t $pattern 2>/dev/null | head -n1 || true)
    printf '%s' "$found"
}

CODE_TAR="${CODE_TAR:-$(pick_newest 'bol_training_v*.tar.gz')}"
DATA_TAR="${DATA_TAR:-$(pick_newest 'bol_data_v*.tar.gz')}"

# ----------------------------------------------------------------------------
# Step 1: Preflight
# ----------------------------------------------------------------------------

preflight() {
    log "=== preflight ==="

    [[ -n "$CODE_TAR" && -f "$CODE_TAR" ]] || die "code tarball not found (CODE_TAR='$CODE_TAR')"
    [[ -n "$DATA_TAR" && -f "$DATA_TAR" ]] || die "data tarball not found (DATA_TAR='$DATA_TAR')"
    log "code bundle: $CODE_TAR"
    log "data bundle: $DATA_TAR"

    if [[ -f CHECKSUMS.txt ]]; then
        log "verifying CHECKSUMS.txt"
        if ! sha256sum -c CHECKSUMS.txt; then
            die "checksum verification failed"
        fi
    else
        warn "no CHECKSUMS.txt found, skipping integrity check"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L || warn "nvidia-smi -L failed"
    else
        warn "nvidia-smi not found — GPU may not be visible"
    fi

    # Disk-free check on /workspace (fall back to current partition if /workspace missing).
    local target_dir
    if [[ -d /workspace ]]; then
        target_dir=/workspace
    else
        target_dir="$(pwd)"
    fi
    local free_kb
    free_kb=$(df -Pk "$target_dir" | awk 'NR==2 {print $4}')
    local free_gb=$((free_kb / 1024 / 1024))
    log "free space on $target_dir: ${free_gb} GB"
    if (( free_gb < 10 )); then
        die "less than 10 GB free on $target_dir — aborting"
    fi
}

# ----------------------------------------------------------------------------
# Step 2: System deps (apt)
# ----------------------------------------------------------------------------

install_system_deps() {
    log "=== system deps ==="

    local missing=()
    for pkg in espeak-ng build-essential libsndfile1; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            missing+=("$pkg")
        fi
    done

    if (( ${#missing[@]} == 0 )); then
        log "all apt packages already installed"
        return 0
    fi

    log "installing: ${missing[*]}"

    # apt-get update — may fail on flaky nvidia mirror. Don't abort.
    if ! $SUDO apt-get update -y; then
        warn "apt-get update had errors, continuing with cached index"
    fi

    # Install without recommends to keep image lean; tolerate partial failure path via exit code.
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends "${missing[@]}" \
        || die "apt-get install failed for: ${missing[*]}"
}

# ----------------------------------------------------------------------------
# Step 3: Python deps (pip)
# ----------------------------------------------------------------------------

install_python_deps() {
    log "=== python deps ==="

    # Force torch 2.6.0 + torchaudio 2.6.0 + torchvision 0.21.0 from cu124 index.
    # This MUST happen before monotonic_align so it compiles against 2.6 headers/ABI.
    log "installing torch 2.6.0 / torchaudio 2.6.0 / torchvision 0.21.0 (cu124)"
    $PY -m pip install --force-reinstall --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
        || die "torch/torchaudio/torchvision install failed"

    log "installing remaining python deps"
    # numpy<2 pinned because some of our deps (librosa/numba chain) aren't numpy-2 clean yet.
    $PY -m pip install --no-cache-dir \
        'numpy<2' \
        scipy \
        accelerate \
        transformers \
        librosa \
        soundfile \
        pyyaml \
        tensorboard \
        munch \
        phonemizer \
        huggingface_hub \
        Cython \
        nltk \
        tqdm \
        matplotlib \
        pandas \
        einops \
        einops-exts \
        'misaki[en]>=0.9.4' \
        click \
        pydub \
        typing-extensions \
        || die "pip install of core python deps failed"

    # monotonic_align: MUST install AFTER torch 2.6 upgrade. Force reinstall + no cache
    # so it's rebuilt against the newly installed torch. Skipping this is what caused
    # "CUDA illegal memory access" last session.
    log "installing monotonic_align (rebuilt against torch 2.6)"
    $PY -m pip install --force-reinstall --no-cache-dir \
        git+https://github.com/resemble-ai/monotonic_align.git \
        || die "monotonic_align install failed"

    log "python deps installed"
}

# ----------------------------------------------------------------------------
# Step 4: Extract bundles
# ----------------------------------------------------------------------------

extract_bundles() {
    log "=== extract bundles ==="

    mkdir -p "$RUN_DIR"

    # Code bundle: strip the top-level bol_training_v5/ dir so contents land directly in $RUN_DIR.
    # --no-same-owner prevents chown-fail on Mac-built tarballs (uid=501).
    log "extracting code bundle into $RUN_DIR/"
    tar --no-same-owner --strip-components=1 -xzf "$CODE_TAR" -C "$RUN_DIR" \
        || die "code bundle extraction failed"

    # Data bundle: extract to a tempdir then merge in (don't clobber kokoro_base.pth or other
    # files already placed by the code bundle).
    local data_tmp
    data_tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$data_tmp'" EXIT

    log "extracting data bundle into tempdir $data_tmp"
    tar --no-same-owner -xzf "$DATA_TAR" -C "$data_tmp" \
        || die "data bundle extraction failed"

    # Find the actual root of the data bundle (it may or may not have a top-level dir).
    local data_root="$data_tmp"
    local nested
    nested=$(find "$data_tmp" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)
    if [[ -n "$nested" ]]; then
        # If there's exactly one top-level dir and no other entries, use it as root.
        local top_count
        top_count=$(find "$data_tmp" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')
        if [[ "$top_count" == "1" && -d "$nested" ]]; then
            # Heuristic: if this dir contains "dataset" or "training", treat it as the root.
            if [[ -d "$nested/dataset" || -d "$nested/training" ]]; then
                data_root="$nested"
            fi
        fi
    fi
    log "data bundle root: $data_root"

    # Merge in, non-destructively (-n = no clobber). kokoro_base.pth in RUN_DIR/training/
    # stays intact since data bundle shouldn't have it; train/val/rasa manifests come in fresh.
    log "merging data bundle into $RUN_DIR/ (no-clobber)"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --ignore-existing "$data_root"/ "$RUN_DIR"/ \
            || die "rsync merge failed"
    else
        # Fallback: cp -R -n. -n may emit warnings when skipping; don't abort on those.
        cp -R -n "$data_root"/. "$RUN_DIR"/ || warn "cp -n merge had skips (expected)"
    fi

    rm -rf "$data_tmp"
    trap - EXIT
    log "bundles extracted"
}

# ----------------------------------------------------------------------------
# Step 5: Sanity checks
# ----------------------------------------------------------------------------

sanity_checks() {
    log "=== sanity checks ==="

    local ckpt="$RUN_DIR/training/kokoro_base.pth"
    [[ -f "$ckpt" ]] || die "missing $ckpt"
    local ckpt_bytes
    ckpt_bytes=$(stat -c%s "$ckpt" 2>/dev/null || stat -f%z "$ckpt")
    local ckpt_mb=$((ckpt_bytes / 1024 / 1024))
    log "kokoro_base.pth = ${ckpt_mb} MB"
    (( ckpt_mb > 300 )) || die "kokoro_base.pth is only ${ckpt_mb} MB (expected > 300)"

    [[ -f "$RUN_DIR/training/kokoro_symbols.py" ]] \
        || die "missing $RUN_DIR/training/kokoro_symbols.py"
    [[ -f "$RUN_DIR/StyleTTS2/kokoro_symbols.py" ]] \
        || die "missing $RUN_DIR/StyleTTS2/kokoro_symbols.py (text_utils imports this at train time)"

    # Manifest line counts — warn only, don't abort.
    local train_lines val_lines
    if [[ -f "$RUN_DIR/training/train_list.txt" ]]; then
        train_lines=$(wc -l <"$RUN_DIR/training/train_list.txt" | tr -d ' ')
    else
        die "missing $RUN_DIR/training/train_list.txt"
    fi
    if [[ -f "$RUN_DIR/training/val_list.txt" ]]; then
        val_lines=$(wc -l <"$RUN_DIR/training/val_list.txt" | tr -d ' ')
    else
        die "missing $RUN_DIR/training/val_list.txt"
    fi
    log "train_list.txt = $train_lines lines (expected 24676)"
    log "val_list.txt = $val_lines lines (expected 1134)"
    if [[ "$train_lines" != "24676" ]]; then
        warn "train_list.txt line count differs from expected 24676"
    fi
    if [[ "$val_lines" != "1134" ]]; then
        warn "val_list.txt line count differs from expected 1134"
    fi

    # WAV counts.
    local rasa_count iv_count
    rasa_count=$(find "$RUN_DIR/dataset/audio/rasa" -type f -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
    iv_count=$(find "$RUN_DIR/dataset/audio/indicvoices_r" -type f -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
    log "rasa wavs = $rasa_count (expected >= 13000)"
    log "indicvoices_r wavs = $iv_count (expected >= 11000)"
    (( rasa_count >= 13000 )) || die "rasa wav count too low ($rasa_count)"
    (( iv_count >= 11000 )) || die "indicvoices_r wav count too low ($iv_count)"

    # Test-import kokoro_symbols from the StyleTTS2 cwd (that's how text_utils resolves it).
    log "test-importing kokoro_symbols from StyleTTS2/"
    ( cd "$RUN_DIR/StyleTTS2" && "$PY" -c "import kokoro_symbols as k; assert len(k.symbols)==178 and k.dicts['ɭ']==144; print('symbols OK')" ) \
        || die "kokoro_symbols import/assert failed"

    # Test-load the checkpoint.
    log "test-loading kokoro_base.pth"
    "$PY" -c "import torch; c = torch.load('$RUN_DIR/training/kokoro_base.pth', map_location='cpu', weights_only=False); assert 'net' in c and set(c['net']) == {'bert','bert_encoder','predictor','decoder','text_encoder'}; print('ckpt OK')" \
        || die "kokoro_base.pth load/assert failed"

    log "all sanity checks passed"
}

# ----------------------------------------------------------------------------
# Step 6: Build monotonic_align Cython ext (if in-tree copy exists)
# ----------------------------------------------------------------------------

build_monotonic_align() {
    log "=== monotonic_align cython ext ==="
    local ma_dir="$RUN_DIR/StyleTTS2/monotonic_align"
    if [[ -d "$ma_dir" ]]; then
        log "building in-tree monotonic_align at $ma_dir"
        ( cd "$ma_dir" && "$PY" setup.py build_ext --inplace ) \
            || warn "in-tree monotonic_align build failed — pip-installed version should still work"
    else
        log "no in-tree monotonic_align dir; relying on pip install"
    fi
}

# ----------------------------------------------------------------------------
# Step 7: Accelerate default config (avoid first-run interactive prompt)
# ----------------------------------------------------------------------------

init_accelerate_config() {
    log "=== accelerate default config ==="
    if command -v accelerate >/dev/null 2>&1; then
        accelerate config default || warn "accelerate config default failed (may already be set)"
    else
        warn "accelerate CLI not on PATH — it should be post-install; continuing"
    fi
}

# ----------------------------------------------------------------------------
# Step 8: Launch training
# ----------------------------------------------------------------------------

run_stage_1() {
    log "=== STAGE 1: train_first.py ==="
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$RUN_DIR/StyleTTS2"
    exec accelerate launch --mixed_precision=bf16 --num_processes=1 \
        train_first.py --config_path "$CONFIG"
}

run_stage_2() {
    log "=== STAGE 2: train_second.py ==="
    # Stage 1 output check (filename varies across forks; check both common patterns).
    local s1_ckpt_dir="$RUN_DIR/StyleTTS2/logs/kokoro-marathi"
    local found=""
    for candidate in \
        "$s1_ckpt_dir/first_stage.pth" \
        "$s1_ckpt_dir/first_stage_final.pth"; do
        if [[ -f "$candidate" ]]; then
            found="$candidate"
            break
        fi
    done
    if [[ -z "$found" ]]; then
        # Also accept any first_stage_epoch_*.pth.
        found=$(ls -1t "$s1_ckpt_dir"/first_stage_epoch_*.pth 2>/dev/null | head -n1 || true)
    fi
    if [[ -z "$found" ]]; then
        die "Stage 1 checkpoint not found in $s1_ckpt_dir — run STAGE=1 first"
    fi
    log "found Stage 1 checkpoint: $found"

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$RUN_DIR/StyleTTS2"
    exec accelerate launch --mixed_precision=bf16 --num_processes=1 \
        train_second.py --config_path "$CONFIG"
}

run_stage_both() {
    log "=== STAGE both: train_first then train_second ==="
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ( cd "$RUN_DIR/StyleTTS2" && \
      accelerate launch --mixed_precision=bf16 --num_processes=1 \
          train_first.py --config_path "$CONFIG" ) \
        || die "Stage 1 training failed"
    log "Stage 1 complete, launching Stage 2"
    ( cd "$RUN_DIR/StyleTTS2" && \
      accelerate launch --mixed_precision=bf16 --num_processes=1 \
          train_second.py --config_path "$CONFIG" ) \
        || die "Stage 2 training failed"
    log "Stage 2 complete"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

main() {
    log "launch_training.sh starting (STAGE=$STAGE, RUN_DIR=$RUN_DIR)"

    preflight
    install_system_deps
    install_python_deps
    extract_bundles
    build_monotonic_align
    sanity_checks
    init_accelerate_config

    case "$STAGE" in
        setup)
            log "setup complete, not launching training. To run Stage 1: STAGE=1 ./launch_training.sh"
            exit 0
            ;;
        1)
            run_stage_1
            ;;
        2)
            run_stage_2
            ;;
        both)
            run_stage_both
            ;;
        *)
            die "unknown STAGE='$STAGE' (expected: setup, 1, 2, both)"
            ;;
    esac
}

main "$@"
