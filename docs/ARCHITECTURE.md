# Architecture and Design Notes

Marathi-specific design decisions and compatibility requirements.

For how-to training steps, see `TRAINING_GUIDE.md`.
For troubleshooting specific errors, see `TROUBLESHOOTING.md`.

## Kokoro-82M component layout

Reference component sizes (unchanged from upstream):

| Component | Parameters |
|---|---|
| bert (PLBERT) | 6.29 M |
| bert_encoder | 0.39 M |
| predictor | 16.19 M |
| text_encoder | 5.61 M |
| decoder (ISTFTNet) | 53.28 M |
| **Total** | **81.76 M** |

Voicepack target shape: `[510, 1, 256]` (`float32`). 510 = PLBERT max token length; 256 = `style_dim × 2` (128 acoustic + 128 prosodic).

The decoder is ISTFTNet (22-channel STFT spectrogram → iSTFT → waveform). This is NOT HiFi-GAN — don't try to swap in HiFi-GAN vocoder weights. The distinction is visible in shape match counts during checkpoint loading (ISTFTNet has 22-channel output; HiFi-GAN has 1).

## Vocab: the ɭ at slot 144

Kokoro's 178-token vocab is English-centric with ~65 unused Private Use Area (PUA) placeholder slots (U+E000–U+F8FF) reserved for future multilingual use. Our only Marathi addition was replacing one of those placeholders with `ɭ` (U+026D, voiced retroflex lateral approximant — the phoneme for Marathi ळ).

Specifically:

- **Slot 144** in `training/kokoro_symbols.py` was `''` (PUA placeholder).
- We changed it to `'ɭ'` and regenerated `dicts` (char → idx mapping).
- The inference-time vocab in `configs/config_mr.json` has `"ɭ": 144` mirroring this.

### Why just `ɭ`?

Every other Marathi IPA character (including `ʈ`, `ɖ`, `ɳ`, `ɾ`, `ʰ`, `ã`, `ã̃ː`, etc.) is already in the Kokoro vocab, because they appear in one or more of the languages Kokoro was pretrained on (Hindi, Japanese, Chinese). The retroflex lateral `ɭ` is distinctively Dravidian + Marathi and is the only phoneme espeak-ng's Marathi backend produces that wasn't already slotted.

Hindi does NOT use `ɭ` — Hindi ळ is rare and typically transcribed as `l`. So this patch is specifically what Marathi needs beyond Hindi.

### Why slot 144 and not, say, 177?

Any unused PUA slot would work. 144 was arbitrary — it's near the "middle" of the PUA range and was what we committed to during the first training run. Changing the slot index now would invalidate every trained checkpoint (the embedding table layout is baked in), so it stays at 144 forever for this project.

## Symbol table = single source of truth

There are TWO places that refer to the Kokoro vocab:

1. **`training/kokoro_symbols.py`** — a Python list (`symbols`) and derived `dicts: char → int`. Used by StyleTTS2's `TextCleaner` **at training time**.
2. **`configs/config_mr.json` → `vocab` key** — a JSON `{char: int}` dict. Used by Kokoro's `KModel.forward_with_tokens` **at inference time**.

These must agree on EVERY slot, not just `ɭ`. The inference path silently drops any phoneme whose char isn't in `config_mr.json["vocab"]`. If you re-index the training symbols but forget to update `config_mr.json`, you'll get mysteriously garbled output with no errors.

The `scripts/diagnostics/diagnose_nan.py` script cross-checks these two sources and prints disagreements.

## Phonemizer: espeak-ng via misaki

- **G2P backend:** `espeak-ng` (C binary, must be installed at system level)
- **Python wrapper:** `misaki.espeak.EspeakG2P(language="mr")` — a thin layer that shells out to espeak-ng
- **Language code:** we use `"m"` at the Kokoro `KPipeline` layer (monkey-patched into `LANG_CODES`) which maps to espeak-ng's `"mr"`.

espeak-ng has had Marathi support since version 1.50 (released 2019). The Marathi G2P rules are reasonably good for read speech and handle common Devanagari characters, conjuncts, and schwa deletion. Edge cases (loan words, English-in-Marathi) are handled by espeak-ng falling back to English-like phoneme mappings.

## Multi-speaker style conditioning

StyleTTS2 is multi-speaker via two style encoders:

1. **`style_encoder`** (trained in Stage 1, refined in Stage 2): takes a reference mel spectrogram → 128-dim acoustic/timbre vector.
2. **`predictor_encoder`** (trained in Stage 2 only, copy-initialized from `style_encoder`): takes a reference mel → 128-dim prosody vector.

A voicepack is the mean of both encoders' outputs over 200 reference samples from one speaker, concatenated and broadcast into the `[510, 1, 256]` shape KModel expects.

For the Marathi fine-tune:

- Rasa has 2 distinct speakers (marathi_female, marathi_male) with thousands of clips each. The style encoders converge well on these identities.
- IV-R has 329 speakers with 30-105 clips each. Per-speaker style vectors are noisier due to sparser training exposure. This is why IV-R voicepacks benefit from `speed ≈ 1.15` (see `VOICEPACKS.md`).

We use the **Stage 1 `style_encoder`** and the **Stage 2 `predictor_encoder`** when extracting voicepacks. Stage 2's adversarial training can drift `style_encoder` (spectral-norm buffer drift), so taking it from Stage 1 gives more consistent timbre. `predictor_encoder` must come from Stage 2 because Stage 1 doesn't train it.

## Why fp32 instead of bf16

Kokoro's ISTFTNet decoder contains spectral operations (iSTFT on 22-channel spectrograms) that accumulate numerical error in bf16 and produce NaN gradients within the first few steps. We diagnosed this extensively early in the project before pivoting to fp32.

fp32 training on A100 80 GB comfortably fits `batch_size: 12` (Stage 1) and `batch_size: 8` (Stage 2). The wall-time cost of fp32 vs bf16 is ~30%, which is worth paying for stability on this specific model.

## `joint_epoch` and the adversarial phase

`configs/config_marathi_ft.yml` has `joint_epoch: 3`. What happens at that boundary:

| Phase | Losses active |
|---|---|
| Stage 1 (all 10 epochs) | Mel, Gen, Disc (small MSD only), Mono (monotonic alignment), S2S (seq2seq), SLM |
| Stage 2 epochs 1-2 (pre-joint) | Same as Stage 1 + Dur, F0, Norm, LM (predictor training) |
| Stage 2 epochs 3-10 (joint) | All pre-joint losses + full MSD/MPD adversarial + GenLM/DiscLM + Diff |

The "joint" part means the generator (decoder + predictor) and the full discriminator stack train together with adversarial pressure. This is where the Kokoro model goes from "correct phonemes, flat delivery" to "natural-sounding speech." It's also the VRAM spike that's notorious for OOMing on tight-budget GPUs (see `TROUBLESHOOTING.md#stage-2-oom-at-joint_epoch`).

## What we deliberately did NOT change

- Model architecture (no layer count / dimension tweaks)
- Loss weights (semidark's defaults work)
- Optimizer choice (AdamW stays)
- Training stages (Stage 1 → Stage 2 structure is unchanged)
- Kokoro's core vocab slots 0-177 (we only added `ɭ` at a previously-unused slot)
- Voicepack format (full [510, 1, 256] compatibility with upstream Kokoro inference paths)

This keeps the Marathi fine-tune compatible with standard Kokoro inference tooling — a Marathi voicepack drops into any Kokoro-compatible inference stack as long as the served model is the fine-tune we trained.

## Untrained English phoneme slots

Our Marathi training data contains essentially no English words. The Kokoro vocab slots for English-specific phonemes (`æ`, `ʤ`, `ʧ`, `ɒ`, `ɑː`, etc.) are present but receive almost no gradient during our fine-tune. Their weights remain at whatever Kokoro's base checkpoint had them at.

Consequence: pure English text synthesizes poorly (as expected — not our use case), but code-switch Marathi+English synthesizes surprisingly well because:

1. espeak-ng's Marathi backend detects Latin script and produces real English IPA (not transliteration)
2. The Kokoro base's English phoneme training carries through (those slots weren't nuked, just not refined)
3. The Marathi voicepack's style vector applies Marathi timbre on top → "Indian-English accent" output

See `INFERENCE.md#minglish-marathi--english` for examples and limitations.
