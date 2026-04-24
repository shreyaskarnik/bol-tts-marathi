---
title: bol-tts-marathi
emoji: 🪔
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
license: apache-2.0
models:
- <your-user>/bol-tts-marathi
tags:
- text-to-speech
- marathi
- kokoro
---

# bol-tts-marathi — Marathi TTS

Live demo of [bol-tts-marathi](https://huggingface.co/<your-user>/bol-tts-marathi), a Kokoro-82M fine-tune for Marathi (मराठी).

Four voices: **Asha** (आशा, female), **Vivek** (विवेक, male), **Mukta** (मुक्ता, female, IV-R), **Dnyanesh** (ज्ञानेश, male, IV-R).

Enter Marathi text (Devanagari script), pick a voice and speed, get 24 kHz audio back. Recommended speeds:

| Voice | Speed |
|---|---|
| Asha, Vivek | 0.85-0.95 |
| Mukta, Dnyanesh | 1.10-1.20 |

### WebGPU version

This Space is a CPU Gradio demo. A WebGPU (ONNX / transformers.js) version is planned — see the [bol-tts-marathi repo](https://github.com/<your-user>/bol-tts-marathi) for progress.

### Technical details

See the [model card](https://huggingface.co/<your-user>/bol-tts-marathi) for training details, limitations, and citation.
