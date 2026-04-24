# Setup

This guide takes you from a fresh clone to a working local inference environment. Training pod setup is a subset of this (see `TRAINING_GUIDE.md` for the pod-specific bits).

## 1. System dependencies

```bash
# macOS
brew install espeak-ng libsndfile

# Ubuntu / Debian (training pods)
sudo apt-get install -y espeak-ng libsndfile1
```

Both are required:
- `espeak-ng` backs the Marathi G2P in `misaki.espeak.EspeakG2P(language="mr")`
- `libsndfile` is required by the `soundfile` Python package for WAV I/O

## 2. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<you>/bol-tts-marathi
cd bol-tts-marathi
```

The submodules are:

| Path | Upstream | Purpose |
|---|---|---|
| `kokoro/` | [semidark/kokoro](https://github.com/semidark/kokoro) | Kokoro inference package (semidark's fork with compat patches) |
| `StyleTTS2/` | [semidark/StyleTTS2](https://github.com/semidark/StyleTTS2) | Training code (Kokoro-82M compatible) |

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## 3. Python environment via uv

[uv](https://github.com/astral-sh/uv) is required (faster, deterministic, and matches semidark's tooling).

```bash
# one-time install
curl -LsSf https://astral.sh/uv/install.sh | sh

# create + sync .venv
uv sync
source .venv/bin/activate
```

Default dependencies are enough for local inference + data prep. For training on a pod, also install the `training` extra:

```bash
uv sync --extra training
```

## 4. Environment variables

Most scripts respect these. They default to sensible values if unset.

| Variable | Default | Purpose |
|---|---|---|
| `BOL_REPO` | `$(pwd)` | Root of this clone |
| `BOL_CHECKPOINTS` | `$BOL_REPO/checkpoints` | Where trained `.pth` files and voicepacks land |
| `POD_HOST`, `POD_PORT` | — | SSH endpoint of the training pod (for handoff + monitor scripts) |

Set them for your shell session:

```bash
export BOL_REPO="$(pwd)"
export BOL_CHECKPOINTS="$BOL_REPO/checkpoints"
# pod-only:
export POD_HOST=a.b.c.d
export POD_PORT=22
```

## 5. Verify

Quickly smoke-test the Python env:

```bash
python -c "
import torch, soundfile, misaki
from misaki import espeak
g = espeak.EspeakG2P(language='mr')
p, _ = g('नमस्कार')
print('torch', torch.__version__, ' marathi IPA:', p)
"
```

Expected: `torch 2.6.x  marathi IPA: nˌəmskˈaːɾ`

If you see `AttributeError: type object 'EspeakWrapper' has no attribute 'set_data_path'`, the pod/Mac has a phonemizer version mismatch. See `TROUBLESHOOTING.md#misaki-espeak-api-mismatch`.

## 6. Where to go next

- To prep your dataset: `DATA_PREPARATION.md`
- To run inference with a pre-trained checkpoint (skip dataset prep): `INFERENCE.md`
- To understand the Marathi-specific design: `ARCHITECTURE.md`
