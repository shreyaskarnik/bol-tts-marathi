# v0.4 full Stage 2.5 — runbook with epoch-1 eval gate

The v0.4 full run is ~25h × ~$1.51/h ≈ $38. To bound the downside if epoch-1
quality is broken (the "radio tuning" failure mode), this runbook bakes in
an automated eval loop that synthesizes a WAV after every epoch save. You
listen after epoch 1 (~2.5h, ~$3.75 spent) and decide whether to keep going
or abort.

## Layout — `nohup` + background (RunPod has no tmux)

Both training and eval run as `nohup`'d background processes. They survive
SSH disconnect, log to files, and can be tailed any time you re-SSH in.
PIDs persisted at `logs/kokoro-marathi-v0_4/launch_pids.txt`.

## Step-by-step

### 1. Spin up pod (RunPod, A100 SXM 80GB, $1.51/h)

Use the existing pod with the persistent `/workspace` volume — all the v0.4
prep is already there.

### 2. Persistent env (~10s if Task #20's setup_pod_env.sh ran in a prior session)

```bash
ssh -p $PORT root@$IP

export PYTHONUSERBASE=/workspace/.local
export PATH=/workspace/.local/bin:$PATH

# Idempotent — if already done, exits in <1s
bash /workspace/bol_run/experiments/v0_4_lang_conditioning/scripts/setup_pod_env.sh
```

### 3. Build full v0.4 manifests (~5 min)

```bash
cd /workspace/bol_run

# Train manifest — v0.2 train_list.txt → v0.4 with lang_ids
python experiments/v0_4_lang_conditioning/scripts/build_v0_4_manifest.py \
    --dict   experiments/v0_4_lang_conditioning/data/loanword_dict_dev_to_latin.tsv \
    --input  training/train_list.txt \
    --output training/train_list_v0_4.txt \
    --min-freq 2

# Val manifest
python experiments/v0_4_lang_conditioning/scripts/build_v0_4_manifest.py \
    --dict   experiments/v0_4_lang_conditioning/data/loanword_dict_dev_to_latin.tsv \
    --input  training/val_list.txt \
    --output training/val_list_v0_4.txt \
    --min-freq 2

# Verify the int-speaker format is preserved (v0.2 train_list.txt already has int speakers)
head -1 training/train_list_v0_4.txt | awk -F'|' '{print "speaker col 3:", $3}'
```

Expected flip rate: ~10% en tokens (in design-doc sweet spot). If the script
prints <5% or >40%, stop and investigate before training.

### 4. Update config to point at v0.4 manifests

```bash
sed -i 's|train_list_v0_4_smoke\.txt|train_list_v0_4.txt|g; s|val_list_v0_4_smoke\.txt|val_list_v0_4.txt|g' \
    experiments/v0_4_lang_conditioning/configs/config_marathi_v0_4_langcond.yml
# Also point train_data path at /workspace/bol_run/training/ since the file is there now
sed -i 's|../experiments/v0_4_lang_conditioning/data/train_list_v0_4\.txt|../training/train_list_v0_4.txt|; s|../experiments/v0_4_lang_conditioning/data/val_list_v0_4\.txt|../training/val_list_v0_4.txt|' \
    experiments/v0_4_lang_conditioning/configs/config_marathi_v0_4_langcond.yml

grep -E 'train_data|val_data' experiments/v0_4_lang_conditioning/configs/config_marathi_v0_4_langcond.yml
```

### 5. Upload a voicepack for the eval loop

```bash
# From your laptop:
scp -P $PORT \
    ~/work/rnd/bol-tts/webgpu-demo/public/voices/mf_mukta.bin \
    root@$IP:/workspace/bol_run/
```

### 6. Launch (one command — backgrounds both processes)

```bash
bash /workspace/bol_run/experiments/v0_4_lang_conditioning/scripts/launch_full_run.sh
```

This script:
- Verifies config + voicepack + manifests + persistent env are all in place
- Starts `accelerate launch train_second.py` in nohup background → `training.log`
- Starts `auto_eval_loop.sh` in nohup background → `eval_loop.log`
- Saves both PIDs to `logs/kokoro-marathi-v0_4/launch_pids.txt`
- Disowns both so they survive SSH disconnect

### 7. Monitor (any time you re-SSH in)

```bash
tail -f /workspace/bol_run/StyleTTS2/logs/kokoro-marathi-v0_4/training.log
# Ctrl+C to stop tailing (training keeps running)

tail -f /workspace/bol_run/StyleTTS2/logs/kokoro-marathi-v0_4/eval_loop.log

ls /workspace/bol_run/StyleTTS2/logs/kokoro-marathi-v0_4/audio_samples/
```

### 8. THE EPOCH-1 GATE (~2.5h after launch)

After ~2.5h of training, the first epoch ckpt lands. The eval loop synthesizes
WAV automatically. Then on your laptop:

```bash
scp -P $PORT root@$IP:/workspace/bol_run/StyleTTS2/logs/kokoro-marathi-v0_4/audio_samples/epoch_00000.wav \
    ~/Downloads/v0_4_epoch_00000.wav
open ~/Downloads/v0_4_epoch_00000.wav
```

**Pass criteria — if epoch 1 sounds like:**
- ✓ Clean Marathi speech (similar to v0.2 baseline `/tmp/v0_2_baseline_fixed.wav`) → continue
- ✗ Radio tuning / hiss / structural garble → **abort training immediately**, debug

**Abort procedure (if needed):**
```bash
bash /workspace/bol_run/experiments/v0_4_lang_conditioning/scripts/stop_full_run.sh
```

This kills both the training process and the eval loop using the PIDs
in `launch_pids.txt`, plus a fallback grep-and-kill for any escaped child
processes.

Cost so far at epoch 1: ~2.5h × $1.51 ≈ **$3.75** (well below the $38 full).

### 9. Continue to completion (~22.5h more)

If epoch 1 sounds good, training continues. Watch:
- Disc/Gen losses become non-zero starting epoch 4 (joint_epoch:3 warmup ends)
- Validation loss trends down across epochs
- Predictor_encoder norm stays in a healthy 1-3 range (extract a voicepack each epoch via the eval loop's audio sample)

Final ckpt: `logs/kokoro-marathi-v0_4/epoch_2nd_00009.pth` (1.8 GB).

### 10. Post-training

- Extract voicepacks from the final ckpt (`scripts/extract_voicepacks_mr.py` —
  unchanged from v0.2 since predictor_encoder structure is identical)
- Convert to Kokoro inference format
- ONNX export (Task #18) — uses the kokoro fork's patched `KModelForONNX.forward`
  with `lang_ids` input

### 11. Pod shutdown

After ckpt download, stop the pod. Persistent `/workspace` retains everything
for the next session; container disk gets wiped (the dep installs are safe
on `/workspace/.local` per Task #20).

## Cost summary

| Phase | Time | Cost |
|---|---|---|
| Setup + manifest build + voicepack upload | ~10 min | <$0.30 |
| Epoch 1 (gate point) | ~2.5h | $3.75 |
| Epochs 2-10 (if epoch 1 passes) | ~22.5h | $34 |
| Post-train (extract + export + scp) | ~30 min | $0.75 |
| **Total worst case (all epochs run)** | **~25.5h** | **~$39** |
| **Total if abort at epoch 1** | **~3h** | **~$4.50** |

The eval gate effectively bounds the downside at ~12% of full cost.
