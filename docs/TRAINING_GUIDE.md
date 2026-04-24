# Training Guide

This is the practical path for fine-tuning Kokoro-82M on Marathi. For every gotcha we hit (and the signatures for when you're hitting them), see `TROUBLESHOOTING.md`. For design decisions behind the Marathi adaptation, see `ARCHITECTURE.md`.

## 1. Prerequisites

### Hardware

| GPU | Verdict |
|---|---|
| A100 SXM 80 GB | Recommended. Runs Stage 1 at bs=12 and Stage 2 at bs=8 cleanly. |
| A100 40 GB | Untested but should work with bs=4-6 for Stage 2. |
| 4090 24 GB | **Does not fit this recipe.** Stage 1's MSD/MPD/WavLM discriminators alone take ~23 GB of activations regardless of batch size. |
| CPU only | Not practical. |

### Reference run

Single A100 SXM 80 GB at $1.506/hr:

| Phase | Time | Cost |
|---|---|---|
| Setup (pip, bundle extraction, sanity checks) | ~10 min | ~$0.25 |
| Stage 1 (10 epochs, bs=12) | ~9 h | ~$14 |
| Stage 2 (10 epochs, bs=8) | ~13 h | ~$20 |
| Voicepack extraction + testing | ~15 min | ~$0.40 |
| **Total** | **~22 h** | **~$35** |

### Software

Pod should have CUDA 12.4 or newer, Python 3.11. The launch script will `pip install` the rest.

## 2. Getting code + data onto the pod

Two paths, pick whichever fits your workflow:

### Path A — package + scp (single bundle, fast)

On your Mac:

```bash
python scripts/build_bundles.py
# produces upload/bol_training_v5.tar.gz (~424 MB code bundle)
# and     upload/bol_data_v2.tar.gz    (~6.2 GB data bundle)

scp -P $POD_PORT upload/*.tar.gz $POD_HOST:/workspace/
scp -P $POD_PORT scripts/launch_training.sh $POD_HOST:/workspace/
```

### Path B — git + manual data copy

If you prefer not to tar a multi-gigabyte audio dataset, clone the repo on the pod and copy audio separately:

```bash
# on pod
cd /workspace
git clone --recurse-submodules https://github.com/<you>/bol-tts-marathi bol_run
rsync -av your.local:dataset/audio/ bol_run/dataset/audio/
```

## 3. Configure and launch

On the pod, one script drives all stages:

```bash
cd /workspace
# STAGE can be: setup | 1 | 2 | both
STAGE=setup ./launch_training.sh    # pip installs, tar extracts, sanity checks
STAGE=1     ./launch_training.sh    # Stage 1 only
STAGE=2     ./launch_training.sh    # Stage 2 only (after Stage 1 finishes)
STAGE=both  ./launch_training.sh    # Stage 1 + Stage 2 end-to-end
```

Under the hood: the script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, clears `__pycache__`, and runs `python3 train_first.py` or `python3 train_second.py` directly (no `accelerate launch` wrapper — we found it adds no value on single-GPU and complicates faulthandler plumbing).

For long-running invocations, detach:

```bash
STAGE=1 setsid nohup ./launch_training.sh > training_s1.log 2>&1 < /dev/null &
disown
```

## 4. Stage 1 (acoustic + alignment)

`StyleTTS2/train_first.py` trains the decoder, text encoder, PLBERT, pitch extractor, text aligner, and (partially) the prosody predictor. This is the expensive part — every downstream step depends on a good Stage 1 checkpoint.

### Config summary

`configs/config_marathi_ft.yml` top-level keys (`train_first.py` reads from top level, nested `training:` block is ignored):

```yaml
log_dir: logs/kokoro-marathi
save_freq: 1
pretrained_model: "../training/kokoro_base.pth"
load_only_params: true
batch_size: 12
num_workers: 8          # 24 works in Stage 1, but 8 is safer for Stage 2; we use one value
epochs_1st: 10
joint_epoch: 3          # Stage 2 adversarial kicks in at this epoch
TMA_epoch: 0            # Force TMA to learn from scratch
```

### Expected Stage 1 trajectory

On the reference run:

| Epoch | Train Mel Loss | Val Loss |
|---|---|---|
| 1 | 0.88 → 0.35 | 0.313 |
| 5 | ~0.26 | 0.252 |
| 8 | ~0.24 | 0.237 |
| 10 (final) | ~0.24 | 0.233 |

The `kokoro_tb_utils_mr.py` hook logs Marathi TensorBoard audio previews per epoch (if misaki imports cleanly on the pod — see `TROUBLESHOOTING.md#misaki-espeak-api-mismatch`).

### Stage 1 handoff

When Stage 1 writes `first_stage.pth` (plus the per-epoch `epoch_1st_00000.pth ... epoch_1st_00009.pth`), run the handoff script from your Mac to download the final checkpoint and run a Mac-side ear test:

```bash
bash scripts/post_stage1_handoff.sh
```

This script verifies Stage 1 is done, downloads the final ckpt, converts it to Kokoro inference format, extracts a mean-style voicepack from Rasa female clips, synthesizes the 7-sentence test set, and kicks off Stage 2 on the pod.

## 5. Stage 2 (prosody + adversarial)

`StyleTTS2/train_second.py` adds the MSD + MPD + WavLM-SLM discriminators, the prosody predictor training, and adversarial losses. This is the step that makes the output *sound right* rather than just have correct phonemes.

### Critical Stage 2 patches

Two patches are non-negotiable for semidark's fork on single A100 80 GB:

1. **`torch.autograd.set_detect_anomaly(False)`** — upstream hardcodes `True`, which hangs the first forward pass indefinitely on Kokoro-scale graphs. Our patched `StyleTTS2/train_second.py` env-gates this via `STYLETTS2_DETECT_ANOMALY` (default off). See `TROUBLESHOOTING.md#stage-2-hangs-after-optimizer-setup`.
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — without this, Stage 2 OOMs at `joint_epoch=3` when adversarial backward pass + discriminator activations fragment the 14+ GB of "reserved but unallocated" memory. `launch_training.sh` sets this by default.

### Stage 2 expected trajectory

| Epoch | Train Loss | Notes |
|---|---|---|
| 1, step 10 | ~0.44 | Fresh load from Stage 1 first_stage.pth. If you see ~7+ here, Stage 1 didn't load. |
| 1 end | ~0.33 | Pre-joint (no adversarial yet) |
| 3 (joint_epoch) | Gen/Disc losses become non-zero | Biggest perceptual jump happens here |
| 10 (final) | ~0.25 | Rarely below — Stage 2 plateaus |

Stage 2 checkpoint file naming is `epoch_2nd_0000N.pth` and the final `second_stage.pth`.

### Stage 2 auto-monitor (optional)

If you want to walk away and get notified when Stage 2 finishes, run this on your Mac:

```bash
nohup caffeinate -dim bash scripts/stage2_auto_monitor.sh \
    > $BOL_CHECKPOINTS/stage2_monitor.out 2>&1 &
disown
```

What it does:

- Polls the pod every 15 min
- When Stage 2 ends (final epoch ckpt exists + no training process), downloads the ckpt, extracts both Rasa voicepacks, synthesizes a test set, and triggers a macOS notification
- Optional RunPod API auto-stop path is in the script — uncomment if you set `RUNPOD_POD_ID` + `RUNPOD_API_KEY`

## 6. After Stage 2

Move on to `VOICEPACKS.md` to extract the four named voicepacks, and `INFERENCE.md` to test them on Mac.
