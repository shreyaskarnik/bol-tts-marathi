# Voicepacks

A voicepack is a small file (~512 KB) that conditions the TTS model to produce a specific speaker's voice. Format: `torch.Tensor` of shape `[510, 1, 256]` (`float32`), where the first 128 dims are the acoustic/timbre conditioning and the last 128 are the prosody conditioning.

## The four named voicepacks

| ID | Display | Source | Meaning |
|---|---|---|---|
| `mf_asha` | Asha (आशा) | Rasa `marathi_female` (5,500 clips) | hope |
| `mm_vivek` | Vivek (विवेक) | Rasa `marathi_male` (8,400 clips) | wisdom |
| `mf_mukta` | Mukta (मुक्ता) | IV-R top female speaker you pick | pearl |
| `mm_dnyanesh` | Dnyanesh (ज्ञानेश) | IV-R top male speaker you pick | knowledge |

## Picking IV-R speakers

IV-R speakers are hashed IDs (`mr_sXXXX`); gender isn't stored in the manifest. List top speakers and listen to samples:

```bash
python scripts/pick_ivr_voices.py \
    --manifest training/indicvoices_r_mr.txt \
    --audio-dir dataset/audio/indicvoices_r \
    --top 20
```

Output: ranked table (speaker_id, n_utts, 3 sample wav paths). Play samples:

```bash
# macOS
afplay dataset/audio/indicvoices_r/mr_s1418_000000.wav

# Linux (with pulse / alsa)
aplay dataset/audio/indicvoices_r/mr_s1418_000000.wav
```

Pick one clearly-female voice → `mf_mukta`. One clearly-male voice → `mm_dnyanesh`. Prefer speakers with higher utt count (70+) for cleaner style extraction.

**Important**: silence at either end of source clips contaminates the style vector. The decoder learns to reproduce that silence as voice identity, which manifests as:

- **Leading silence** → first word gets eaten + audible hiss before speech (seen with raw SPRINGLab/IndicTTS, 0.3-0.4 s pre-speech pad).
- **Trailing silence** → word-final semi-vowels and aspirated stops get clipped (seen with raw Rasa voices Asha/Vivek; final /j/ in सांगतोय fades early).

Always trim both ends before extracting (`trim_silence.py` keeps a 50 ms pad each side):

```bash
python scripts/trim_silence.py \
  --src-dir dataset/audio/rasa_marathi_female \
  --dst-dir dataset/audio/rasa_marathi_female_trimmed

# then extract from the *_trimmed dir
python scripts/upstream/extract_voicepack.py \
  --audio-dir dataset/audio/rasa_marathi_female_trimmed \
  ...
```

If you don't trim, the style encoder averages that silence into the voicepack and synthesized output "eats" the first word (onset is replaced by hiss/silence that matches the baked-in signature).

Typical picks in our reference run: Mukta from the top-5 list, Dnyanesh similar. Speakers with 70-100 utts give usable voicepacks; <50 utts can be too noisy.

## Extracting a voicepack

### Single voicepack

```bash
python scripts/extract_voicepacks_mr.py \
    --stage2-checkpoint    checkpoints/epoch_2nd_00009.pth \
    --stage1-checkpoint    checkpoints/epoch_1st_00009.pth \
    --audio-dir            dataset/audio/rasa \
    --audio-filter         "marathi_female_*.wav" \
    --output               voices/mf_asha.pt \
    --num-samples          200
```

What it does (per `scripts/extract_voicepacks_mr.py`, which wraps semidark's upstream `extract_voicepack.py`):

- Loads the `style_encoder` (acoustic/timbre) weights from the **Stage 1** checkpoint — Stage 2's adversarial training can drift the style encoder, so Stage 1 gives cleaner timbre
- Loads the `predictor_encoder` (prosody) weights from the **Stage 2** checkpoint — predictor_encoder is trained in Stage 2 only
- Samples 200 audio clips matching the filter, computes log-mel spectrograms with the exact training params (n_fft=2048, hop=300, n_mels=80)
- Averages the 200 style vectors from each encoder
- Writes a `[510, 1, 256]` tensor: first 128 dims acoustic, last 128 prosodic

### Producing all four in one shot

```bash
bash scripts/extract_all_voicepacks.sh  # if present
# or loop manually:

for src in "marathi_female" "marathi_male"; do
  case "$src" in
    marathi_female) vp="mf_asha" ;;
    marathi_male)   vp="mm_vivek" ;;
  esac
  python scripts/extract_voicepacks_mr.py \
    --stage2-checkpoint checkpoints/epoch_2nd_00009.pth \
    --stage1-checkpoint checkpoints/epoch_1st_00009.pth \
    --audio-dir dataset/audio/rasa \
    --audio-filter "${src}_*.wav" \
    --output voices/${vp}.pt
done

# IV-R: pick_ivr_voices.py gave you e.g. mr_s1418 (F) and mr_sec31 (M)
python scripts/extract_voicepacks_mr.py \
  --stage2-checkpoint checkpoints/epoch_2nd_00009.pth \
  --stage1-checkpoint checkpoints/epoch_1st_00009.pth \
  --audio-dir dataset/audio/indicvoices_r \
  --audio-filter "mr_s1418_*.wav" \
  --output voices/mf_mukta.pt

python scripts/extract_voicepacks_mr.py \
  --stage2-checkpoint checkpoints/epoch_2nd_00009.pth \
  --stage1-checkpoint checkpoints/epoch_1st_00009.pth \
  --audio-dir dataset/audio/indicvoices_r \
  --audio-filter "mr_sec31_*.wav" \
  --output voices/mm_dnyanesh.pt
```

## Sanity-checking a voicepack

The extraction script prints two numbers:

```
Acoustic style norm: 2.02
Prosodic style norm: 1.71
```

Healthy ranges:

- **Acoustic norm:** 1.5–3.0 after full training. Below 0.5 suggests style encoder collapse (read `TROUBLESHOOTING.md#stage-2-static-noise`).
- **Prosodic norm:** 1.5–2.5 after Stage 2. If it's < 0.5, either Stage 2 didn't finish or you're pointing at a Stage 1 checkpoint (predictor_encoder not trained).
- **Balanced ratio:** acoustic / prosodic should be between 0.7 and 1.5. Very asymmetric norms indicate one of the encoders is undertrained or broken.

## Speed tuning per voice

Different voicepacks have different optimal playback speeds. Rasa voices (high training exposure) typically want `speed ≈ 0.85-0.95`; IV-R voices (sparse per-speaker data) typically want `speed ≈ 1.1-1.2` because the duration predictor pads conservatively for rare speakers.

Tune by sweeping:

```bash
for spd in 0.75 0.85 1.0 1.15 1.25; do
  python scripts/inference_mac_mr.py \
    --voicepack voices/mf_asha.pt \
    --speed $spd \
    --text "नमस्कार, माझे नाव अमित आहे. मला केळी आणि आंबा आवडतो." \
    --output-dir tmp_sweep/speed_$spd
done
```

Listen, pick the best speed per voice, save into `configs/voice_speeds.json`:

```json
{
  "mf_asha":     { "speed": 0.85 },
  "mm_vivek":    { "speed": 0.90 },
  "mf_mukta":    { "speed": 1.15 },
  "mm_dnyanesh": { "speed": 1.15 }
}
```

Your inference application reads this file and passes the right `speed=` to KPipeline per voice.

## Voicepack portability

The resulting `.pt` files are 512 KB each and are fully portable — same format as upstream Kokoro voicepacks. You can use them with any Kokoro inference stack (Python, WebGPU, ONNX — once exported) as long as the model served matches the one they were extracted from.

They are NOT compatible with the stock hexgrad/Kokoro-82M English voicepacks. Use only with the fine-tuned Marathi model, or you'll get garbled output.

## Publishing voicepacks

If you want to publish the Marathi voicepacks to HuggingFace Hub:

```bash
# one-time
pip install huggingface_hub
huggingface-cli login

# upload
huggingface-cli upload <your-user>/bol-tts-marathi-voices voices/ --repo-type model
```

Recommended repo name convention: `<user>/bol-tts-marathi-<voice-id>` (one repo per voice) or `<user>/bol-tts-marathi-voices` (all four in one repo). One-repo-per-voice is friendlier for downstream integrations that mirror the upstream Kokoro voicepack layout.
