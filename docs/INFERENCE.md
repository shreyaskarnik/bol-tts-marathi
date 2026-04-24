# Inference

Mac-friendly CPU inference for the fine-tuned Marathi Kokoro model. GPU inference works identically — just point Torch at the right device.

## Prerequisite artifacts

To synthesize speech you need three files:

1. **Converted model** — a Kokoro-format `.pth` produced from the fine-tune's Stage 2 checkpoint. `scripts/post_stage1_handoff.sh` and `stage2_auto_monitor.sh` both produce this automatically; if you need to do it manually:

   ```python
   import sys; sys.path.insert(0, "./scripts/upstream")  # vendored semidark script
   from test_inference import convert_checkpoint
   convert_checkpoint(
       "checkpoints/epoch_2nd_00009.pth",
       "checkpoints/kokoro_mr_final.pth",
   )
   ```

2. **Voicepack** — `voices/mf_asha.pt` or similar (see `VOICEPACKS.md`).

3. **Marathi-patched config** — `configs/config_mr.json`. This is the Kokoro inference config with `"ɭ": 144` added to the `vocab` dict. The upstream Kokoro `config.json` does NOT have this mapping and will silently drop `ɭ` at tokenization time.

## The inference script

`scripts/inference_mac_mr.py` wraps all the wiring:

```bash
# Default: synthesize a built-in 7-sentence Marathi test set
python scripts/inference_mac_mr.py \
  --model     checkpoints/kokoro_mr_final.pth \
  --config    configs/config_mr.json \
  --voicepack voices/mf_asha.pt \
  --output-dir checkpoints/test_output

# Single custom sentence
python scripts/inference_mac_mr.py \
  --voicepack voices/mf_asha.pt \
  --text "माझे नाव अमित आहे."

# Multiple sentences from a file
python scripts/inference_mac_mr.py \
  --voicepack voices/mf_asha.pt \
  --text-file my_tests.txt

# Speed tuning — lower = slower, more breathing room per phoneme
python scripts/inference_mac_mr.py \
  --voicepack voices/mf_asha.pt \
  --speed 0.85 \
  --text "नमस्कार."
```

Output: 24 kHz WAV files at `<output-dir>/mr_NN.wav`.

## How the Marathi lang_code works

Upstream Kokoro's `KPipeline` ships G2P support for:

```text
a = American English    j = Japanese
b = British English      z = Mandarin Chinese
d = German               e = Spanish   f = French
h = Hindi                i = Italian   p = Brazilian Portuguese
```

No Marathi. Rather than forking the `kokoro` package, `inference_mac_mr.py` monkey-patches `LANG_CODES` at import time:

```python
import kokoro.pipeline as _kp
_kp.LANG_CODES["m"] = "mr"       # Marathi → espeak-ng 'mr'
from kokoro import KModel, KPipeline
pipeline = KPipeline(lang_code="m", model=model)
```

The `espeak-ng` binary has had Marathi support since version 1.50. `misaki.espeak.EspeakG2P(language="mr")` is the layer that invokes it.

## End-to-end example

```bash
source .venv/bin/activate

# one call per voice
for voice in mf_asha mm_vivek mf_mukta mm_dnyanesh; do
  python scripts/inference_mac_mr.py \
    --voicepack voices/${voice}.pt \
    --text "नमस्कार, माझे नाव अमित आहे. मला केळी आणि आंबा आवडतो." \
    --output-dir test_output/${voice}
done

open test_output/
```

## What to expect at each training stage

| Source | Phoneme accuracy | Prosody / duration | Overall |
|---|---|---|---|
| Pretrained Kokoro (no fine-tune) | Wrong for Marathi | English-biased | Not usable |
| Stage 1 epoch 1 | Mostly right, `ɭ` confused with `ɖ` | Flat, vowels crushed | Recognizably Marathi but "robotic" |
| Stage 1 epoch 10 | Right, `ɭ` improving | Still flat | Better |
| Stage 2 epoch 2 (pre-joint) | Right, `ɭ` distinguishable | Improving | Noticeably natural |
| Stage 2 epoch 3+ (joint training) | Right | Major jump — adversarial losses engaged | Production-usable |
| Stage 2 epoch 10 | Right | Fully learned per-voice prosody | Highest quality |

The "kedi → keli" (ɭ correctly articulated) transition typically happens around Stage 2 epoch 2. The biggest perceptual jump is at `joint_epoch` (configured 3), when the adversarial MSD/MPD/WavLM discriminators start training.

## Timestamps

`KModel.forward_with_tokens(input_ids, ref_s, speed)` returns `(audio, pred_dur)`. `pred_dur` is per-phoneme duration in **predictor frames**, where one predictor frame = 600 audio samples at 24 kHz (NOT 300 as you might guess from the mel hop — the prosody predictor runs at half the mel-frame rate, and the decoder upsamples 2× internally before iSTFT).

```python
audio, pred_dur = kmodel.forward_with_tokens(input_ids, ref_s, speed=1.0)
durations_sec = pred_dur.squeeze().cpu().numpy() * 600 / 24000
starts = durations_sec.cumsum() - durations_sec
# (starts[i], starts[i] + durations_sec[i]) is the time span of phoneme[i]
```

For word-level timestamps, phonemize word-by-word (so you remember which phoneme indices belong to which word), then aggregate the min/max of each word's phoneme range. See `scripts/with_timestamps.py`.

## Per-voice speed

The duration predictor is trained per-speaker-distribution. Voices with lots of training data (Rasa female/male) fit the "natural" speed well and sometimes want a slight slowdown (0.85-0.95). IV-R speakers have sparse exposure and the predictor pads their durations conservatively — `speed=1.15` brings them in line.

See `configs/voice_speeds.json` for the per-voice defaults and `VOICEPACKS.md#speed-tuning-per-voice` for how to tune.

## Minglish (Marathi + English)

The espeak-ng Marathi backend detects Latin-script tokens and phonemizes them with English phonemes (e.g. "Google" → `ɡˈuːɡəl`, not Devanagari-transliterated nonsense). That part works.

The model side is weaker: English-specific phoneme slots (`æ`, `ʤ`, `ʧ`, `ɒ`, etc.) are all **in our vocab** but their embeddings inherit from Kokoro's English pre-training and saw almost no gradient during our Marathi fine-tune. Result: English words in a Marathi sentence sound like an **Indian-English accent** — usable for code-switch sentences but not polished.

If you want better Minglish, you'd need a second round of fine-tuning with bilingual code-switch data. Not yet attempted in this repo.

## Going to WebGPU / ONNX

Kokoro has a [JavaScript port](https://github.com/hexgrad/kokoro.js) that runs ONNX-quantized Kokoro models on WebGPU. To deploy your Marathi fine-tune there:

1. Export to ONNX (semidark has references in their kokoro-deutsch README)
2. Quantize to int8 or q4
3. Adapt the JS `LANG_CODES` map similarly to our Python monkey-patch (or fork kokoro.js to add `'m'`)

This is out of scope for the training repo; see Kokoro/kokoro.js repos for the deployment patterns.
