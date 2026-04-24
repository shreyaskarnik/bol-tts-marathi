# Publishing to HuggingFace Hub

This repo ships scaffolding to publish three things:

1. **Model repo** (`<user>/bol-tts-marathi`) — PyTorch weights + voicepacks + config, usable via the `kokoro` Python package.
2. **ONNX model repo** (`<user>/bol-tts-marathi-onnx`) — same model exported for WebGPU / transformers.js. Requires an ONNX export step.
3. **Gradio Space** (`<user>/bol-tts-marathi`) — live web demo.

## Prerequisites

```bash
pip install -U huggingface_hub
huggingface-cli login
```

Set env vars:

```bash
export BOL_REPO="$(pwd)"
export BOL_CHECKPOINTS="$BOL_REPO/checkpoints"
export HF_USER="<your-hf-username>"
# optional: override repo names
export HF_MODEL_REPO="$HF_USER/bol-tts-marathi"
export HF_SPACE_REPO="$HF_USER/bol-tts-marathi"
```

## 1. Publish PyTorch model + Gradio Space

One script handles both:

```bash
bash scripts/push_to_hf.sh
```

What it does:

- Stages `hf_model/` with `README.md` (model card), `config.json`, `kokoro-mr-v1_0.pth`, four `voices/*.pt` files, and `voice_speeds.json`
- Substitutes `<your-user>` placeholders with `$HF_USER`
- Uploads to `$HF_MODEL_REPO` (model) and `$HF_SPACE_REPO` (space)

Verify:

- Model card: `https://huggingface.co/<user>/bol-tts-marathi`
- Live demo: `https://huggingface.co/spaces/<user>/bol-tts-marathi` (boots in ~60 s on cold start, then cached)

## 2. Export to ONNX

ONNX is needed for the WebGPU transformers.js deployment. The key constraint is that Kokoro's `TorchSTFT` uses complex tensors, which ONNX doesn't support. Setting `disable_complex=True` on `KModel` swaps in `CustomSTFT` (real-arithmetic) and exports cleanly.

```bash
python scripts/export_onnx.py \
  --model     $BOL_CHECKPOINTS/kokoro_mr_final.pth \
  --config    configs/config_mr.json \
  --output    $BOL_CHECKPOINTS/kokoro-mr-v1_0.onnx \
  --verify
```

`--verify` reloads the ONNX via `onnxruntime` and checks that outputs match the PyTorch forward to ~1e-3. Takes ~30 s on CPU.

Expected output:

```
exporting to .../kokoro-mr-v1_0.onnx
  exported .../kokoro-mr-v1_0.onnx (310.2 MB)
verifying with onnxruntime…
  max|pt_audio - ort_audio|: 5.34e-05
  max|pt_dur   - ort_dur|:   0
  OK
```

### Quantization (optional but strongly recommended for WebGPU)

Browsers load 310 MB reluctantly. Use `optimum` to quantize:

```bash
pip install optimum[onnxruntime]
python - <<'PY'
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
ORTQuantizer.from_pretrained("checkpoints/kokoro-mr-v1_0.onnx").quantize(
    save_dir="checkpoints/",
    quantization_config=qconfig,
)
PY
```

Produces `kokoro-mr-v1_0_quantized.onnx` (~80 MB int8). Quality loss is usually inaudible for Kokoro-scale TTS.

### Publish the ONNX repo

```bash
ONNX_REPO="$HF_USER/bol-tts-marathi-onnx"
STAGE="$BOL_CHECKPOINTS/onnx-stage"
mkdir -p "$STAGE/onnx" "$STAGE/voices"
cp $BOL_CHECKPOINTS/kokoro-mr-v1_0.onnx           "$STAGE/onnx/model.onnx"
cp $BOL_CHECKPOINTS/kokoro-mr-v1_0_quantized.onnx "$STAGE/onnx/model_quantized.onnx"
cp configs/config_mr.json                          "$STAGE/config.json"
cp $BOL_CHECKPOINTS/voices/*.pt                    "$STAGE/voices/"
# Use the same model card; could also write an ONNX-specific one
cp hf_model/README.md                              "$STAGE/README.md"
sed -i '' "s|<your-user>|$HF_USER|g" "$STAGE/README.md"

huggingface-cli upload "$ONNX_REPO" "$STAGE" . --repo-type model \
  --commit-message "ONNX export of bol-tts-marathi (fp32 + int8)"
```

## 3. WebGPU / transformers.js Space (future work)

Once the ONNX repo exists, the WebGPU Space follows the pattern of [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX): a static HTML + JS app using `@xenova/transformers` to load the quantized ONNX in-browser and run on WebGPU.

Key work items for a full WebGPU deployment (not yet done in this repo):

1. Fork or adapt [kokoro.js](https://github.com/hexgrad/kokoro.js) to add `'m'` as a Marathi lang code (mirror our Python monkey-patch).
2. Vendor `espeak-ng.wasm` (WebAssembly build of espeak-ng with Marathi voice data) so in-browser G2P works.
3. Host at `<user>/bol-tts-marathi-web` as a static HTML Space (`sdk: static`).

In the meantime, the Gradio Space in `hf_space/` provides the same UX on Hub-hosted CPU.
