# Stage 2.5 — v0.2 expanded-data Stage 2 restart

After v0.1 (Stage 1 + Stage 2, ~34 h on a single A100 SXM 80 GB) we have a working Marathi Kokoro-82M with four named voices. v0.2 adds:

- **Cleaner Rasa source** (trailing-silence trimmed; fixes the "Asha eats word-final phonemes" bug).
- **New SPRINGLab/IndicTTS_Marathi data** mixed in (~5 h studio Marathi, 2 speakers → adds `mf_priya`, `mm_arjun`).
- **Tighter prosody** from a fresh Stage 2 run on the expanded corpus.

> **⚠️ This doc was rewritten after v0.2 attempts 1+2 failed.** The original recipe used `second_stage_load_pretrained: true` (continuation init from v0.1 final) + `joint_epoch: 0` (no adversarial warmup). That combination collapses predictor_encoder into a degenerate small-magnitude regime in epoch 1 and never recovers — voicepacks come out shaky/elderly. See `feedback_predictor_encoder_lr_collapse.md`. **The correct approach is a true Stage-2-from-Stage-1 restart on the expanded data**, documented below.

## Why true Stage-2-from-Stage-1 restart, not continuation

Stage 1 teaches alignment + acoustics. SPRINGLab is well-aligned Marathi from a *cleaner* recording environment — Stage 1's knowledge transfers as-is. Stage 2 then adapts prosody + applies adversarial polish.

The natural-seeming optimization — "skip Stage 1, init Stage 2 from v0.1 final" — *fails* when the new data introduces new speakers. The pretrained predictor_encoder learned its style space from Rasa+IV-R only; loading it then immediately applying adversarial pressure (no warmup) on a distribution including SPRINGLab speakers pushes it into a degenerate solution. The discriminator essentially wins permanently.

A true Stage-2-from-Stage-1 restart with `joint_epoch: 3` warmup avoids this trap: 3 epochs of teacher-forced + duration losses *before* adversarial pressure lets the predictor_encoder find a sensible style space on the new data. Then adversarial polish operates on a stable foundation.

Cost: we lose v0.1's Stage 2 compute (~$25 baked into the prior final ckpt) and have to redo Stage 2 from scratch (~22 h, ~$33). Worth it — the alternative is voicepacks that don't ship.

## Data recipe

| Source | Utterances | Speakers | New for v0.2? |
|---|---|---|---|
| Rasa Marathi (trimmed) | ~13,900 | 2 | trim_silence applied; same content as v0.1 |
| IndicVoices-R Marathi (filtered) | ~10,776 | 329 | unchanged |
| SPRINGLab/IndicTTS_Marathi | ~10,939 | 2 | **new** |
| **Combined** | **~35.6k** | **333** | ~50% bigger than v0.1 |

## Files

- [`configs/config_marathi_v0_2.yml`](../configs/config_marathi_v0_2.yml) — **the canonical v0.2 config (validated-good)**. `second_stage_load_pretrained: false`, `first_stage_path: "first_stage.pth"`, `joint_epoch: 3`, `lr: 1e-4`, `bert_lr: 1e-5`, `ft_lr: 1e-4`.
- [`configs/config_marathi_v0_2.broken.yml.do-not-use`](../configs/config_marathi_v0_2.broken.yml.do-not-use) — the original (broken) continuation-mode config, kept as a paper trail. Do not use.
- [`scripts/data_prep/prep_v0_2.sh`](../scripts/data_prep/prep_v0_2.sh) — single-command orchestrator: rasa → trim_silence → ivr → springlab → merge.
- [`scripts/data_prep/prepare_springlab_mr.py`](../scripts/data_prep/prepare_springlab_mr.py) — streams `SPRINGLab/IndicTTS_Marathi` and writes a manifest fragment + 24 kHz wavs.
- [`scripts/data_prep/merge_manifests.py`](../scripts/data_prep/merge_manifests.py) — extended to read `springlab_mr.txt`.
- [`scripts/launch_training.sh`](../scripts/launch_training.sh) — `run_stage_2` sniffs the config; **now also refuses to launch if `second_stage_load_pretrained: true` + `joint_epoch: 0`** (the trap combination).

## Launch protocol

```bash
# On the pod, after extracting the data + code bundles:

# 1. Make sure the Stage 1 final ckpt is in place. If you ran v0.1 here, this
#    is already at logs/kokoro-marathi/first_stage.pth. Otherwise, run STAGE=1
#    first (~9 h) — yes, this is the cost of doing v0.x right.
ls bol_run/StyleTTS2/logs/kokoro-marathi/first_stage.pth || \
    STAGE=1 CONFIG=../configs/config_marathi_v0_2.yml ./launch_training.sh

# 2. Assemble the expanded corpus (idempotent, resume-safe).
cd bol_run && bash scripts/data_prep/prep_v0_2.sh && cd ..

# 3. Launch Stage 2 from the Stage 1 ckpt with v0.2 data.
STAGE=2 CONFIG=../configs/config_marathi_v0_2.yml ./launch_training.sh \
    2>&1 | tee training_v0_2.log
```

Expected runtime: ~22 h on a single A100 SXM 80 GB at `bs=8` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Final ckpt lands at `bol_run/StyleTTS2/logs/kokoro-marathi/epoch_2nd_00009.pth`.

## What to watch in the training log

A healthy run shows:

| Window | Signature |
|---|---|
| Setup | `found Stage 1 checkpoint: ...first_stage.pth` followed by `Loading the first stage model at ...`. (NOT `test-loading kokoro_base.pth` — that's a *sanity check*, not the init source.) |
| Stage 2 step 0 | `Extracting Stage 1 baseline voicepack for TensorBoard ... acoustic_norm = prosodic_norm` (equal because predictor_encoder is initialized as a copy of style_encoder). |
| Epochs 1–3 | `Disc Loss: 0.00000` `Gen Loss: 0.00000` `DiscLM Loss: 0.00000` `GenLM Loss: 0.00000`. Adversarial losses literally zero. This is `joint_epoch: 3` warmup. |
| Epoch 4 onward | Adversarial losses non-zero. Biggest perceptual quality jump happens here. Watch `Disc Loss` oscillate in 2–4 range (healthy adversarial); divergence outside that band is a problem. |
| End of every epoch | `Validation loss: X.XXX` line. Should trend down. |
| Voicepack norms | Extract a voicepack each epoch via `scripts/extract_voicepacks_mr.py`. `Acoustic style norm` should stay 1–2; `Prosodic style norm` should *diverge* from the acoustic norm over epochs (predictor_encoder specializing). If prosodic norm collapses below 0.5 and stays there for 2+ epochs, abort — same trap. |

## Post-training

Same as v0.1:
1. Extract voicepacks (`scripts/extract_voicepacks_mr.py`) — now including `mf_priya` + `mm_arjun`.
2. Convert to Kokoro format (`scripts/convert_kokoro_weights.py` … plus the inverse extraction we ran inline: extract `bert/bert_encoder/predictor/decoder/text_encoder` from the StyleTTS2 ckpt and save as a flat dict).
3. ONNX export.
4. Push model + voicepacks + Space.

## Known caveats

- **Don't use continuation mode for new-speaker runs.** `second_stage_load_pretrained: true` is fine if your training data is identical to the previous Stage 2 corpus. Add new speakers and you'll hit the predictor_encoder collapse.
- **Always have adversarial warmup with `joint_epoch >= 3` for fresh Stage 2 runs.** `joint_epoch: 0` only made sense in the original (broken) "we're resuming from an already-Stage-2-trained model" framing — which we now know doesn't work for adding new speakers.
- **Default lr to 1e-4** (the v0.1 baseline). Halve only after observing actual instability (NaN, divergence, validation MOS regression).
- **Tokenizer/vocab changes need a Stage 1 redo.** Stage 2.5 can't fix vocab.
- **Disk: training writes ~17 GB of intermediate ckpts** across Stage 2's 10 epochs. `/workspace` (RunPod network volume) has plenty of headroom; archive old runs to a sibling dir to avoid confusion when relaunching.
