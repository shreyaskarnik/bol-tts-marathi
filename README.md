# bol-tts-marathi

Training recipe and inference scaffold for fine-tuning [Kokoro-82M](https://github.com/hexgrad/kokoro) on Marathi (मराठी), built on top of [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch)'s German fine-tuning recipe.

## What This Is

- A reproducible end-to-end Marathi Kokoro fine-tune: dataset prep → Stage 1 → Stage 2 → voicepack extraction → Mac-side inference
- Original scripts for Marathi dataset preparation from AI4Bharat [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) and [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
- Marathi-specific Kokoro vocab addition: `ɭ` (U+026D, retroflex lateral for ळ) inserted at slot 144
- Documentation of every gotcha we hit during training (`docs/TROUBLESHOOTING.md`)
- Automation for handoff between Stage 1 and Stage 2 plus an auto-monitor that notifies on Stage 2 completion

## What This Is Not

- Not a runnable Kokoro replacement — use [hexgrad/kokoro](https://github.com/hexgrad/kokoro) for that
- Not a bundled training dataset (Rasa and IndicVoices-R licenses apply; you fetch them yourself)
- Not a training framework fork — we submodule semidark's kokoro + StyleTTS2 forks and apply a small Marathi overlay

## Start Here

| Goal | Doc |
|---|---|
| I want to train my own Marathi voice | `docs/TRAINING_GUIDE.md` |
| I want to install and test locally | `docs/SETUP.md` + `docs/INFERENCE.md` |
| I want to prep a Marathi dataset | `docs/DATA_PREPARATION.md` |
| I want to extract voicepacks from a trained model | `docs/VOICEPACKS.md` |
| Training broke and I need to know why | `docs/TROUBLESHOOTING.md` |
| I want architecture / compatibility details | `docs/ARCHITECTURE.md` |

## Status

The end-to-end pipeline is working:

`Dataset preparation → Weight conversion → Stage 1 → Stage 2 → Voicepack extraction → KModel inference`

Reference run:
- **24,676 train utterances / 1,134 val** (Rasa marathi_female + marathi_male + IV-R 329 speakers)
- **Stage 1: 10 epochs, final val_loss ≈ 0.23**
- **Stage 2: 10 epochs, bs=8, ~13 h on A100 SXM 80GB**
- 4 named voicepacks: Asha (मf), Vivek (mम), Mukta (fमु, IV-R pick), Dnyanesh (mज्ञ, IV-R pick)

## Quick Setup

### Prerequisites

```bash
# macOS
brew install espeak-ng libsndfile

# Ubuntu/Debian (training pods)
sudo apt-get install espeak-ng libsndfile1
```

`espeak-ng` is the Marathi G2P backend (via `misaki`); `libsndfile` is required by `soundfile` for WAV I/O.

### Clone and sync

```bash
git clone --recurse-submodules https://github.com/<you>/bol-tts-marathi
cd bol-tts-marathi
uv sync
```

The `kokoro/` and `StyleTTS2/` submodules point at semidark's forks (which carry the Kokoro-82M compatibility patches).

### One environment variable

Most scripts respect `BOL_REPO` (path to this clone). Default is the repo root; export it if you run from a different cwd:

```bash
export BOL_REPO="$(pwd)"
```

## Repository Layout

```text
kokoro/          # submodule → semidark/kokoro (Kokoro inference package)
StyleTTS2/       # submodule → semidark/StyleTTS2 (training code)
configs/
  config_marathi_ft.yml   # Stage 1 + Stage 2 training config
  config_mr.json          # Kokoro inference config with ɭ → 144 patched into vocab
  voice_speeds.json       # per-voice optimal playback speed
training/
  kokoro_symbols.py       # Marathi-forked symbol table (ɭ at slot 144); copy into StyleTTS2/ before training
  kokoro_tb_utils_mr.py   # Marathi test sentences for TensorBoard inference previews
  OOD_texts.txt           # Marathi out-of-domain text set used during Stage 2
scripts/
  data_prep/              # Rasa + IV-R prep, speaker ID fixup, duration filter, OOD generation, manifest merge
  diagnostics/            # Symbol table / audio / IPA length sanity checks
  launch_training.sh      # STAGE=setup|1|2|both — pod driver
  convert_kokoro_weights.py, fix_config_and_relaunch.py
  inference_mac_mr.py     # Mac CPU inference with monkey-patched Marathi lang_code
  extract_voicepacks_mr.py, pick_ivr_voices.py
  post_stage1_handoff.sh, stage2_auto_monitor.sh  # automation between stages and after Stage 2
docs/            # SETUP, TRAINING_GUIDE, DATA_PREPARATION, VOICEPACKS, INFERENCE, TROUBLESHOOTING, ARCHITECTURE
```

## Voices

We derive four named voicepacks from the fine-tuned model. Voice names were picked to be Marathi words with positive meanings:

| ID | Display | Source | Meaning |
|---|---|---|---|
| `mf_asha` | Asha (आशा) | Rasa marathi_female | hope |
| `mm_vivek` | Vivek (विवेक) | Rasa marathi_male | wisdom |
| `mf_mukta` | Mukta (मुक्ता) | IV-R top female speaker | pearl |
| `mm_dnyanesh` | Dnyanesh (ज्ञानेश) | IV-R top male speaker | knowledge |

IV-R picks are not fixed — run `scripts/pick_ivr_voices.py` to rank IV-R speakers by utterance count, listen to samples, and choose.

## Contributing

Contributions welcome, especially:

- Reproducible runs on a different GPU (1×4090, 2×A100) — we trained on single A100 SXM 80GB
- Other Indian languages (Hindi, Gujarati, Punjabi) — Hindi needs no new vocab slot; Marathi's only novel phoneme was `ɭ`
- Minglish (Marathi + English code-switch) improvements — English phoneme slots are barely trained in the fine-tune; a second-pass adaptation is a promising direction

## Attribution

See `NOTICE` for upstream attribution and license details. This project is deeply indebted to [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch) for the reference Kokoro fine-tuning recipe and to [hexgrad](https://github.com/hexgrad) for Kokoro + misaki.

## License

Apache License 2.0 — see `LICENSE`.
