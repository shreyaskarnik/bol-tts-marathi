# Troubleshooting

Every failure we hit during Marathi Kokoro training, its signature, and the fix. Organized roughly in the order you'd hit them.

---

## Dataset preparation

### Speaker IDs as strings → `int(speaker_id)` crash

**Symptom:** `ValueError: invalid literal for int() with base 10: 'marathi_female'` inside `StyleTTS2/meldataset.py:134`.

**Cause:** StyleTTS2 expects speaker IDs to be numeric.

**Fix:** Run `scripts/data_prep/fix_speaker_ids.py` after you've built the Rasa + IV-R manifests. It converts names to ints and writes `speaker_map.json` for inverse lookup.

---

### PLBERT token length overflow

**Symptom:** `assert tokens.shape[1] <= 510` or silent model crashes on long utterances.

**Cause:** PLBERT's positional embedding caps at 510 tokens. IPA can be longer than Devanagari text (one char per phoneme, not per syllable).

**Fix:** Run `scripts/diagnostics/check_ipa_lengths.py` before training. Anything over 510 has to be filtered or re-segmented. For this dataset, the 6-second duration filter (`filter_by_duration.py --max-seconds 6`) keeps IPA below 510 for every utterance.

---

### `kokoro_symbols.py` drops unknown chars silently

**Symptom:** `NaN` training losses from step ~10, or `Gen Loss: nan`. No explicit error.

**Cause:** `TextCleaner` silently drops any IPA character not in the symbol list. The model sees shortened sequences that don't align with the audio, training diverges.

**Fix:** `scripts/diagnostics/diagnose_nan.py` scans the full train manifest and reports any char not in `kokoro_symbols.py`. For Marathi, every espeak-ng IPA char is covered EXCEPT `ɭ` (retroflex lateral). We patched `ɭ` into slot 144 of our fork — see `training/kokoro_symbols.py` and `ARCHITECTURE.md`.

---

## Config and launcher

### Top-level vs nested `batch_size`

**Symptom:** You edit `batch_size` in `configs/config_marathi_ft.yml` but training keeps using the old value.

**Cause:** The config has both a top-level `batch_size:` and a nested `training: batch_size:`. `train_first.py` reads the **top-level** value; the nested block is silently ignored.

**Fix:** Always edit the top-level key. `scripts/fix_config_and_relaunch.py` uses PyYAML to set the top-level batch size programmatically if you don't want to handle the sed edge cases.

---

### CHECKSUMS mismatch after launcher edit

**Symptom:** `launch_training.sh` aborts with a checksum verification failure.

**Cause:** The launcher verifies bundles via `CHECKSUMS.txt`. Any edit to the launcher invalidates it.

**Fix:** Regenerate with `sha256sum upload/*.tar.gz upload/launch_training.sh > upload/CHECKSUMS.txt` before re-upload.

---

## Pod / SSH / environment

### `monotonic_align` CUDA illegal memory access

**Symptom:** Random CUDA illegal memory access errors inside monotonic align on pod, even though the model was trained with the same code on a different machine.

**Cause:** `monotonic_align` ships pre-compiled for a specific torch version. Our bundle upgrades torch to 2.6 after install, but an older `monotonic_align` wheel from pypi cache gets used.

**Fix:** After any torch upgrade on pod:

```bash
pip install --force-reinstall --no-cache-dir \
    git+https://github.com/resemble-ai/monotonic_align.git
```

`launch_training.sh` does this automatically during setup.

---

### bf16 NaN losses with Kokoro

**Symptom:** Training losses go to `nan` at step 10 when `mixed_precision: bf16`.

**Cause:** Kokoro's ISTFTNet decoder has spectral operations that bf16 underflows. We diagnosed this extensively before pivoting to fp32.

**Fix:** Use `mixed_precision: no` (fp32). A100 80 GB has enough memory for fp32 at bs=12 (Stage 1) and bs=8 (Stage 2). If you absolutely need bf16, you'd need gradient clipping on the spectral norm branches — we haven't tested this.

---

### misaki / espeak API mismatch on pod

**Symptom:** At Stage 2 startup, stdout shows:
```
Could not load German G2P for TensorBoard inference:
  type object 'EspeakWrapper' has no attribute 'set_data_path'
```

**Cause:** `misaki 0.9.4`'s `espeak.py` calls `EspeakWrapper.set_data_path(...)` at module import. `phonemizer >= 3.0` removed that method — `data_path` is now an attribute (`EspeakWrapper.data_path = ...`).

**Impact:** Non-fatal. The TB preview function catches the import error. Training continues normally.

**Fix (optional, for TB audio previews):** Patch `misaki/espeak.py` line 10:

```bash
# on pod
sed -i 's|EspeakWrapper.set_data_path(espeakng_loader.get_data_path())|EspeakWrapper.data_path = espeakng_loader.get_data_path()|' \
    /usr/local/lib/python3.11/dist-packages/misaki/espeak.py
```

Note: `espeakng_loader.get_data_path()` returns a CI build path that doesn't exist at runtime on most pods, so even patching this won't make TB previews work. We've disabled that feature. Training is unaffected.

---

## Stage 1 issues

### Stage 1 "extracting voicepack for TensorBoard inference" with no progress for 10+ min

**Cause:** Usually just misaki's espeak import failing (see above). Check the next log line for `acoustic_norm=...` — if it arrives within a minute, you're fine.

### Stage 1 acoustic_norm jumps from 0.4 to 3.75 at epoch 1

**Cause:** Expected. The style encoder overshoots early, then settles around 1.7-2.0 by epoch 6. It is not runaway.

**Action:** Watch only for continued climb past epoch 3-4. If it's still above 5 at epoch 6, you have a style encoder divergence and need gradient clipping on `style_encoder`.

---

## Stage 2 issues

### Stage 2 hangs after optimizer setup

**Symptom:** Stage 2 log stops after the last `decoder AdamW(...)` optimizer dump. VRAM climbs to 70 GB, GPU util stays at 0%, power stays at idle. Minutes pass with no progress. No error.

**Cause (root):** `StyleTTS2/train_second.py` line 10 has `torch.autograd.set_detect_anomaly(True)` hardcoded upstream. Anomaly mode records a Python traceback for every autograd op, which is ~100× slower on Kokoro-scale graphs and indistinguishable from a hang.

**Fix:** Our fork env-gates it. Ensure your copy of `StyleTTS2/train_second.py` has:

```python
import os
def _env_flag(name, default=False):
    v = os.getenv(name); return default if v is None else v.strip().lower() in {"1","true","yes","on"}

DETECT_ANOMALY = _env_flag("STYLETTS2_DETECT_ANOMALY", default=False)
torch.autograd.set_detect_anomaly(DETECT_ANOMALY)
```

Only set `STYLETTS2_DETECT_ANOMALY=1` if you are actively debugging a NaN, and only for a handful of steps.

The `post_stage1_handoff.sh` scp's the patched `train_second.py` to the pod as part of Stage 2 launch — so once you use the handoff path, pod state is always correct.

---

### Stage 2 OOM at joint_epoch (epoch 3+)

**Symptom:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 378.00 MiB.
GPU 0 has a total capacity of 79.25 GiB of which 164.75 MiB is free.
Process ... has 79.08 GiB memory in use. Of the allocated memory 63.71 GiB
is allocated by PyTorch, and 14.84 GiB is reserved by PyTorch but unallocated.
```

**Cause:** At `joint_epoch: 3`, MSD / MPD / WavLM-SLM discriminators engage in the backward pass. Peak VRAM jumps ~25% and fragmentation pushes total allocation past 80 GB. `bs=12` works for epochs 0-2 and OOMs around step 440 of epoch 4.

**Fix:**
1. `batch_size: 8` (down from 12). Costs ~30% more wall time per epoch (3084 vs 2056 steps) but is solidly stable.
2. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reclaims ~10% of fragmented memory.

If bs=8 still OOMs (unlikely on 80 GB but possible on 40 GB):
- `slmadv_params.batch_percentage: 0.25` (half of default 0.5)
- `slmadv_params.max_len: 300` (from 500)
- `batch_size: 6`

---

### Stage 2 resume is broken

**Symptom:** Setting `first_stage_path` to a Stage 2 partial checkpoint (e.g. `epoch_2nd_00002.pth`) and `load_only_params: false` restarts training from epoch 0, Dur Loss spikes to ~9 (fresh-init territory).

**Cause (two things at once):**
1. `train_second.py` unconditionally runs `model.predictor_encoder = copy.deepcopy(model.style_encoder)` after `load_checkpoint`. This wipes the trained predictor_encoder weights.
2. The fork's `load_checkpoint` doesn't cleanly restore Stage 2 optimizer state when source has adversarial optimizer groups.

**Workaround:** Don't resume Stage 2 mid-training. If it dies, the pragmatic choice is:
- The partial Stage 2 ckpt is already usable for production (extract voicepacks from it, confirmed `ɭ` articulates correctly as early as epoch 2 of Stage 2).
- If you really need a full 10-epoch Stage 2, restart from `first_stage.pth` with bs=8 + expandable_segments. Losing 3 epochs ≈ $4.50 and 3 h is cheaper than multi-hour resume debugging.

---

## SSH / process management

### `pkill -f "train_second"` kills my SSH session

**Cause:** `pkill -f PATTERN` matches any process whose cmdline contains PATTERN — including the SSH remote shell running your `pkill` (because your command string contains "train_second").

**Fix:** Always kill by specific PID, not pattern, when running over SSH:

```bash
# find PIDs in one SSH call
ssh ... "pgrep -f '^python3 train_second'"
# kill them in a second SSH call
ssh ... "kill 12345 12346"
```

Or use `pkill -f "^python3 train_second"` with a `^` anchor (won't match `bash -c 'pkill -f ...'`). Never just `pkill -f "train_second"` in an inline SSH command.

---

### `nvidia-smi Processes: (empty)` while VRAM is occupied

**Cause:** Usually a training process died without releasing its CUDA context (rare but happens with OOMs). VRAM shows used, nothing visible in `nvidia-smi`.

**Fix:** `fuser -v /dev/nvidia*` will show the real holder. If nothing found, the GPU needs a full reset via `sudo fuser -k /dev/nvidia*` or pod restart.

---

## Inference

### Marathi text synthesizes with missing phonemes

**Symptom:** The `ɭ` in words like केळी sounds like `ɖ` (kedi instead of keli), or Marathi-specific retroflexes are inconsistent.

**Cause:** Usually one of two things:
1. **Config vocab mismatch:** `configs/config_mr.json` must have `"ɭ": 144`. Upstream Kokoro doesn't. Our file does — use ours, not the upstream Kokoro `config.json`.
2. **Stage 1 only model:** Stage 1 models have good phonemes but weak articulation. `ɭ` specifically was the last phoneme to solidify in our runs — it becomes clearly distinguishable from `ɖ` only after Stage 2 epoch 2+.

**Fix:** Use `configs/config_mr.json` with KModel, and train to at least Stage 2 epoch 3 (adversarial engaged).

---

### नमस्कार sounds like "nmskr" (vowels crushed)

**Cause:** Duration predictor hasn't converged. Happens at Stage 1 models and very early Stage 2 models.

**Temporary fix:** Lower `--speed` in `inference_mac_mr.py` to 0.7 — gives vowels more frames. Not a permanent solution; a properly-trained Stage 2 Final model hits this at `--speed 1.0`.

**Real fix:** Train to Stage 2 completion. By epoch 10 the duration predictor is calibrated.

---

### Per-voice optimal speed

Different voicepacks have different "natural" playback speeds. Rasa voices (seen at high frequency during training) want `speed ≈ 0.85-0.95`; IV-R voices (sparse per-speaker exposure) want `speed ≈ 1.1-1.2` because the duration predictor over-pads for rare speakers.

Tune with `scripts/inference_mac_mr.py --speed X --text "..."` across [0.75, 0.85, 1.0, 1.15, 1.25] and save into `configs/voice_speeds.json`.

---

## General advice

- **Trust the val loss curve more than ear tests early on.** Ear tests are misleading until Stage 2 epoch 3+ (joint training engages). A val loss of 0.23-0.25 on Stage 1 is already a "good enough" model; further Stage 1 epochs don't materially change quality.
- **Don't resume across stages.** Stage 1 → Stage 2 is a one-way handoff via `first_stage.pth`; if Stage 2 dies, restart Stage 2 from scratch rather than resume a partial Stage 2.
- **Keep the Stage 1 checkpoint.** `extract_voicepack.py`'s `--style-encoder-model` flag wants the Stage 1 checkpoint because Stage 2 can degrade the style encoder via adversarial drift. Using Stage 1's style encoder for voicepack extraction gives consistently better timbre.
