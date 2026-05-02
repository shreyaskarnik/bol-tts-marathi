# v0.4 architecture smoke test — A100 runbook

**Goal**: validate v0.4 PLBERT lang-conditioning patches end-to-end in <10 min
of A100 compute (well within a $1.51/hour budget).

**Validates**: integration (no NaN, gradient flow into lang_embedding,
predictor_encoder collapse trap NOT firing).
**Does NOT validate**: whether v0.4 actually improves Minglish quality (smoke
manifest's lang_ids are deterministic-pseudo-random, NOT semantically aligned).

## Pre-flight checklist (do this before paying for the pod)

| Item | Where it lives locally | Size |
|---|---|---|
| StyleTTS2 patches | `kokoro-deutsch/StyleTTS2` branch `v0_4_lang_conditioning` | ~10 KB |
| v0.4 smoke config | `experiments/v0_4_lang_conditioning/configs/config_marathi_v0_4_smoke.yml` | <5 KB |
| Smoke manifests (500 + 50 rows) | `experiments/v0_4_lang_conditioning/data/{train,val}_list_smoke.txt` | ~210 KB |
| **v0.2 final ckpt** | `checkpoints/epoch_2nd_v0_2_FINAL.pth` | **1.8 GB ← biggest upload** |
| SPRINGLab audio (500 wavs, smoke manifest references these) | `dataset/audio/springlab_mr/springlab_female_000000.wav` … 0499 | ~250 MB |

Upload total: ~2 GB. At 50 Mbps upload that's ~5-6 min. Plan accordingly.

## A100 session sequence (~1h total, ~10 min real compute)

### 0. Spawn pod (~2 min)
- A100 SXM 80GB with PyTorch 2.x + CUDA 12.x base image
- Network volume `/workspace` mounted (we'll re-use the v0.2 setup there)

### 1. rsync everything to the pod (~10 min depending on network)
```bash
# From your laptop — adjust POD_IP and PORT
POD_IP=<pod-ip>
PORT=<ssh-port>
RSYNC="rsync -avz --progress -e \"ssh -p $PORT\""

# 1a. Code patches (small)
$RSYNC kokoro-deutsch/StyleTTS2/ \
    root@$POD_IP:/workspace/bol_run/StyleTTS2/

# 1b. Smoke manifests (small)
$RSYNC bol-tts-marathi/experiments/v0_4_lang_conditioning/ \
    root@$POD_IP:/workspace/bol_run/bol-tts-marathi/experiments/v0_4_lang_conditioning/

# 1c. v0.2 final ckpt (BIG — 1.8GB)
$RSYNC checkpoints/epoch_2nd_v0_2_FINAL.pth \
    root@$POD_IP:/workspace/bol_run/StyleTTS2/logs/kokoro-marathi/

# 1d. Subset audio that the smoke manifest references
# NB: only the 500 wavs in train_list_smoke.txt + 50 in val_list_smoke.txt
cut -d'|' -f1 bol-tts-marathi/experiments/v0_4_lang_conditioning/data/train_list_smoke.txt \
              bol-tts-marathi/experiments/v0_4_lang_conditioning/data/val_list_smoke.txt \
    > /tmp/smoke_wavs.txt
$RSYNC --files-from=/tmp/smoke_wavs.txt bol-tts-marathi/dataset/audio/ \
    root@$POD_IP:/workspace/bol_run/bol-tts-marathi/dataset/audio/
```

### 2. SSH in + verify (~2 min)
```bash
ssh -p $PORT root@$POD_IP

cd /workspace/bol_run/StyleTTS2
git rev-parse --abbrev-ref HEAD          # expect: v0_4_lang_conditioning (or main if you applied the patch instead)

ls logs/kokoro-marathi/epoch_2nd_v0_2_FINAL.pth   # expect: 1.8 GB
ls ../bol-tts-marathi/experiments/v0_4_lang_conditioning/data/  # train_list_smoke.txt + val_list_smoke.txt

# Sanity check: dataloader can parse the smoke manifest
python -c "
import sys; sys.path.insert(0, '.')
from meldataset import FilePathDataset
with open('../bol-tts-marathi/experiments/v0_4_lang_conditioning/data/train_list_smoke.txt') as f:
    lines = f.readlines()
ds = FilePathDataset(lines, root_path='../bol-tts-marathi/dataset/audio',
                     OOD_data='../training/OOD_texts.txt')
print(f'parsed {len(ds)} rows; first row arity {len(ds.data_list[0])}')
print(f'lang_ids string of first row: \"{ds.data_list[0][3][:60]}...\"')
"
```

If this prints `parsed 500 rows; first row arity 4` you're good.

### 3. Launch the smoke test (~5-10 min)
```bash
cd /workspace/bol_run/StyleTTS2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STYLETTS2_DETECT_ANOMALY=0  # speeds up training; the gate is in the v0.2 patch

STAGE=2 \
  CONFIG=../bol-tts-marathi/experiments/v0_4_lang_conditioning/configs/config_marathi_v0_4_smoke.yml \
  ./launch_training.sh \
  2>&1 | tee logs/kokoro-marathi-v0_4-smoke/training.log
```

### 4. Validate (~5 min — the real test)

Watch the log scroll. Pass criteria — ALL must hold:

| ✓ Criterion | What you're looking for |
|---|---|
| **No NaN crashes** | Zero "NaN detected in ..." prints across the epoch |
| **Loss decreases** | First step total loss compared to last step — should trend down |
| **lang_embedding gets gradient** | After training stops, run the gradient check below |
| **predictor_encoder norm > 0.5** | Extract a voicepack and check its acoustic_norm + prosodic_norm |

#### Gradient flow check
```bash
python -c "
import torch
ck = torch.load('logs/kokoro-marathi-v0_4-smoke/epoch_2nd_00000.pth', map_location='cpu')
le = ck['net']['bert']['module.lang_embedding.weight']
print('lang_embedding shape:', tuple(le.shape))
print('mr (row 0) norm:', le[0].norm().item())
print('en (row 1) norm:', le[1].norm().item())
print('row diff (should be > 0.0001 if gradient flowed):', (le[0] - le[1]).abs().mean().item())
"
```

Expected output (post 1 epoch):
```
lang_embedding shape: (2, 128)
mr (row 0) norm:  ~0.3 (started near 0.225 at random init, drifted from training)
en (row 1) norm:  ~0.3
row diff (should be > 0.0001 if gradient flowed): >> 0.0001
```

If row diff is `~0.0` → gradient is NOT flowing → patches are broken → **STOP and debug locally before any further A100 spend**.

#### predictor_encoder norm check (the v0.2 trap)
```bash
# Extract a voicepack to read the predictor_encoder norm
python ../bol-tts-marathi/scripts/extract_voicepacks_mr.py \
    --ckpt logs/kokoro-marathi-v0_4-smoke/epoch_2nd_00000.pth \
    --voice mf_mukta \
    --out /tmp/smoke_voicepack.pt
# Look for "Acoustic style norm" and "Prosodic style norm" in the output
# Both should be 1-3. If prosodic_norm < 0.5 → PREDICTOR_ENCODER COLLAPSE → bail.
```

### 5. Spin pod down

Once smoke test passes, **stop the pod** before continuing — don't burn idle time
on follow-up tasks (manifest rebuild + IndicCMix dict can happen locally).

## What "smoke test passes" unlocks

- Task #10: re-derive IndicCMix Devanagari→Latin loanword dict
- Task #11 v2: build_v0_4_manifest.py with real word-by-word re-phonemization
- Task #12: flip-rate sanity check on full corpus
- Task #14: kokoro inference patches (for ONNX export)
- Task #17: full v0.4 Stage 2.5 (~$38)

## What "smoke test FAILS" unlocks

- Local debugging on the patches before any further A100 spend
- Most likely failure modes (in order of probability):
  1. CustomAlbert + AlbertModel forward signature mismatch (transformers 5.x vs <5)
  2. Manifest length-invariant violation we missed
  3. Memory mismatch from the additional lang_embedding param breaking some shape
  4. predictor_encoder collapse despite joint_epoch:3 (would be the design-doc-feared
     case where adding new lang signal alone destabilizes the predictor; would
     justify v0.5 architecture revisit)
