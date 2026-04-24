"""Gradio Space for bol-tts-marathi Kokoro fine-tune.

Downloads the model + voicepacks from HF Hub on cold-start, then serves a
per-voice / per-speed synthesis UI.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Kokoro's upstream KPipeline doesn't know Marathi — monkey-patch before import
import kokoro.pipeline as _kp

_kp.LANG_CODES["m"] = "mr"

from kokoro import KModel, KPipeline  # noqa: E402

REPO_ID = os.environ.get("BOL_MODEL_REPO", "<your-user>/bol-tts-marathi")
VOICES: dict[str, dict] = {
    "mf_asha":      {"display": "Asha (आशा) — female",      "speed": 0.85},
    "mm_vivek":     {"display": "Vivek (विवेक) — male",     "speed": 0.90},
    "mf_mukta":     {"display": "Mukta (मुक्ता) — female",  "speed": 1.15},
    "mm_dnyanesh":  {"display": "Dnyanesh (ज्ञानेश) — male", "speed": 1.15},
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_model() -> KModel:
    model_path  = hf_hub_download(repo_id=REPO_ID, filename="kokoro-mr-v1_0.pth")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    kmodel = KModel(repo_id=REPO_ID, config=config_path, model=model_path, disable_complex=True)
    kmodel = kmodel.to(DEVICE)
    kmodel.train(False)
    return kmodel


@lru_cache(maxsize=4)
def _load_voice(voice_id: str) -> torch.Tensor:
    vp_path = hf_hub_download(repo_id=REPO_ID, filename=f"voices/{voice_id}.pt")
    return torch.load(vp_path, map_location="cpu", weights_only=True)


@lru_cache(maxsize=1)
def _pipeline() -> KPipeline:
    return KPipeline(lang_code="m", repo_id=REPO_ID, model=_load_model())


def synth(text: str, voice_id: str, speed: float) -> tuple[int, np.ndarray]:
    if not text.strip():
        raise gr.Error("Please enter some Marathi text.")
    voice = _load_voice(voice_id)
    pipe  = _pipeline()
    chunks = []
    phoneme_out = ""
    for _gs, ps, audio in pipe(text, voice=voice, speed=float(speed)):
        chunks.append(audio)
        phoneme_out += ps + " "
    if not chunks:
        raise gr.Error("Model produced no audio — try a shorter input.")
    wav = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    return 24000, wav.numpy() if hasattr(wav, "numpy") else wav


DEFAULT_EXAMPLES = [
    ["नमस्कार, माझे नाव अमित आहे. मला केळी आणि आंबा आवडतो.", "mf_asha",     0.85],
    ["आज हवामान खूप छान आहे.",                                 "mm_vivek",    0.90],
    ["महाराष्ट्र हे भारतातील एक महत्त्वाचे राज्य आहे.",           "mf_mukta",    1.15],
    ["एक दोन तीन चार पाच सहा सात आठ.",                         "mm_dnyanesh", 1.15],
]


def _voice_change(voice_id: str) -> float:
    return VOICES[voice_id]["speed"]


with gr.Blocks(title="bol-tts-marathi") as demo:
    gr.Markdown(
        "# 🪔 bol-tts-marathi\n"
        "Marathi TTS via [Kokoro-82M fine-tune](https://huggingface.co/" + REPO_ID + "). "
        "Enter Marathi text in Devanagari, pick a voice, listen."
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_in = gr.Textbox(
                label="Marathi text",
                placeholder="नमस्कार…",
                lines=3,
            )
        with gr.Column(scale=1):
            voice_in = gr.Dropdown(
                label="Voice",
                choices=[(v["display"], k) for k, v in VOICES.items()],
                value="mf_asha",
            )
            speed_in = gr.Slider(
                label="Speed",
                minimum=0.5,
                maximum=1.5,
                step=0.05,
                value=0.85,
            )

    go = gr.Button("Synthesize", variant="primary")
    audio_out = gr.Audio(label="Output", type="numpy")

    voice_in.change(fn=_voice_change, inputs=voice_in, outputs=speed_in)
    go.click(fn=synth, inputs=[text_in, voice_in, speed_in], outputs=audio_out)

    gr.Examples(
        examples=DEFAULT_EXAMPLES,
        inputs=[text_in, voice_in, speed_in],
        outputs=audio_out,
        fn=synth,
        cache_examples=False,
    )

    gr.Markdown(
        "---\n"
        "Recommended speeds: Rasa-trained voices (Asha, Vivek) → 0.85-0.95; "
        "IV-R voices (Mukta, Dnyanesh) → 1.10-1.20. "
        "Model and training details on the [model card](https://huggingface.co/" + REPO_ID + ")."
    )


if __name__ == "__main__":
    demo.launch()
