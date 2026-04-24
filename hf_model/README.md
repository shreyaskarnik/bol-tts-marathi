---
license: apache-2.0
language:
- mr
library_name: kokoro
pipeline_tag: text-to-speech
base_model: hexgrad/Kokoro-82M
base_model_relation: finetune
datasets:
- ai4bharat/Rasa
- ai4bharat/indicvoices_r
tags:
- text-to-speech
- tts
- kokoro
- marathi
- indic
- styletts2
- bol-tts
---

# bol-tts-marathi v0.1-preview — Kokoro-82M fine-tuned for Marathi

> ⚠️ **Preview release.** This checkpoint is from Stage 2 epoch 2 (pre-joint-training) of an ongoing 10-epoch Stage 2 run. Phoneme accuracy is good (ɭ for ळ is correctly distinguished from ɖ), but prosody is still being refined. A full v0.1 release will follow once adversarial training completes.

Marathi (मराठी) fine-tune of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), built with the [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch) training recipe.

- **Model type:** StyleTTS2 acoustic model + ISTFTNet decoder (Kokoro-82M architecture, unchanged)
- **Parameters:** 81.76 M
- **Sample rate:** 24 kHz
- **Voices in this preview:** 2 (Asha, Vivek). IV-R voices (Mukta, Dnyanesh) to follow.

Recipe repo: [github.com/shreyaskarnik/bol-tts-marathi](https://github.com/shreyaskarnik/bol-tts-marathi).

## Voices

| Voice ID | Display | Source | Meaning |
|---|---|---|---|
| `mf_asha` | Asha (आशा) | Rasa `marathi_female` | hope |
| `mm_vivek` | Vivek (विवेक) | Rasa `marathi_male` | wisdom |
| `mf_mukta` | Mukta (मुक्ता) | IV-R top female speaker | pearl *(pending)* |
| `mm_dnyanesh` | Dnyanesh (ज्ञानेश) | IV-R top male speaker | knowledge *(pending)* |

## Usage

```python
import torch, soundfile as sf
from kokoro import KModel, KPipeline
import kokoro.pipeline as _kp

_kp.LANG_CODES["m"] = "mr"  # monkey-patch Marathi lang code

kmodel = KModel(
    repo_id="shreyask/bol-tts-marathi",
    config="config.json",
    model="kokoro-mr-v1_0.pth",
)
kmodel.train(False)

pipeline = KPipeline(lang_code="m", repo_id="shreyask/bol-tts-marathi", model=kmodel)
voice = torch.load("voices/mf_asha.pt", map_location="cpu", weights_only=True)

text = "नमस्कार, मी मराठी बोलतो."
chunks = []
for _gs, _ps, audio in pipeline(text, voice=voice, speed=0.85):
    chunks.append(audio)

sf.write("out.wav", chunks[0].numpy() if len(chunks) == 1 else torch.cat(chunks).numpy(), 24000)
```

### Per-voice speed

Rasa voices prefer a slight slowdown; IV-R voices prefer a slight speedup. Defaults in `voice_speeds.json`:

```json
{"mf_asha": 0.85, "mm_vivek": 0.90, "mf_mukta": 1.15, "mm_dnyanesh": 1.15}
```

### Timestamps

Kokoro predicts per-phoneme durations. `KModel.forward_with_tokens` returns `(audio, pred_dur)`. `pred_dur` is in **predictor frames** where 1 frame = 600 audio samples at 24 kHz (the prosody predictor runs at half the mel-frame rate; the decoder upsamples 2× before iSTFT):

```python
audio, pred_dur = kmodel.forward_with_tokens(input_ids, ref_s, speed=1.0)
durations_sec = pred_dur.squeeze().cpu().numpy() * 600 / 24000
starts = durations_sec.cumsum() - durations_sec
# (starts[i], starts[i]+durations_sec[i]) is the time span of phoneme[i]
```

## Training

| Phase | Details |
|---|---|
| Base | `hexgrad/Kokoro-82M` |
| Stage 1 | 10 epochs, bs=12, fp32, ~9h on A100 SXM 80GB. Final val_loss ≈ 0.23 |
| Stage 2 (this preview) | 2 of 10 epochs, bs=8, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Train utts | 24,676 (95/5 split) |
| Speakers | 331 (2 Rasa + 329 IndicVoices-R) |
| Vocab change | `ɭ` (U+026D, retroflex lateral) at Kokoro slot 144 |

Full methodology: [TRAINING_GUIDE.md](https://github.com/shreyaskarnik/bol-tts-marathi/blob/main/docs/TRAINING_GUIDE.md).

## Datasets

- **[AI4Bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa)** (CC-BY-4.0) — Marathi, 13,900 studio-quality utts, 2 speakers.
- **[AI4Bharat/IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)** (CC-BY-4.0, gated) — Marathi, ~11,910 utts, 329 speakers after filtering.

## Limitations

- Preview: Stage 2 adversarial training not yet complete. Expect prosody improvement in the final release.
- Minglish (Marathi + English code-switch) sounds like Indian-English accent — usable but not polished.
- IV-R-derived voices (Mukta, Dnyanesh) pending speaker selection.
- Single language (Marathi only).

## License

Apache 2.0. Training data under CC-BY-4.0.

## Citation

```bibtex
@software{bol_tts_marathi_2026, title={bol-tts-marathi: Kokoro-82M fine-tuned for Marathi}, author={Karnik, Shreyas}, year={2026}, url={https://github.com/shreyaskarnik/bol-tts-marathi}, license={Apache-2.0}}
@software{kokoro_2025, title={Kokoro-82M}, author={hexgrad}, year={2025}, url={https://github.com/hexgrad/kokoro}}
@software{kokoro_deutsch_2026, title={kokoro-deutsch}, author={semidark}, year={2026}, url={https://github.com/semidark/kokoro-deutsch}}
@inproceedings{li2024styletts2, title={StyleTTS 2}, author={Li, Yinghao Aaron and others}, booktitle={NeurIPS}, year={2024}}
```
