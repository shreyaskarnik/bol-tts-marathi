# Data Preparation

This guide describes how to produce a StyleTTS2-compatible Marathi training manifest from two AI4Bharat datasets. Target layout:

```text
dataset/
  audio/
    rasa/marathi_female_00000.wav ... marathi_male_08399.wav  # 13,900 files
    indicvoices_r/mr_sXXXX_NNNNNN.wav                          # ~12,000 files, 329 speakers
training/
  rasa_mr.txt                 # per-utt: relpath|ipa|speaker
  indicvoices_r_mr.txt        # per-utt: relpath|ipa|speaker
  indicvoices_r_mr_stats.json
  train_list.txt              # 24,676 utts (merged + 95/5 split)
  val_list.txt                # 1,134 utts
  speaker_map.json            # speaker-name → int id (required by StyleTTS2)
  OOD_texts.txt               # Marathi OOD text set for Stage 2 (in this repo)
```

## Datasets and licenses

| Dataset | Purpose | License |
|---|---|---|
| [AI4Bharat Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) (Marathi split) | 13.9k utterances, 2 speakers (female + male), studio-quality | CC-BY-4.0 |
| [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) (Marathi) | ~12k utterances, 329 speakers, noisier | CC-BY-4.0, gated |

Neither dataset is redistributed here. IV-R requires HuggingFace login + gated access request. Rasa is open but you still need to accept its terms.

### Strongly recommended for next training: SPRINGLab/IndicTTS_Marathi

[SPRINGLab/IndicTTS_Marathi](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Marathi) (10,939 utts, 2 speakers F+M, studio, CC-BY-4.0) — used for the `mf_mukta` and `mm_dnyanesh` voicepacks in this project. The audio quality is consistently higher than both Rasa and IV-R on listening tests: cleaner studio recording, less background noise, denser per-speaker data (~5,400 clips per speaker vs Rasa's ~5,500 + IV-R's 70-105).

For the next training pass, mix this into the Stage 1 + Stage 2 manifests alongside Rasa + IV-R. Expected improvements:

- Better per-voice consistency (more clips of the same two speakers means the style encoder can overfit less and generalize better)
- Cleaner acoustic model (less noise in training → crisper output)
- Potentially drop IV-R entirely and rely on SPRINGLab + Rasa for a smaller, cleaner training set (~25k utts)

See `scripts/data_prep/prepare_indictts_marathi.py` for the dataset loader pattern. Extending it to write a full training manifest (like `rasa_mr.txt`) is a small edit — reuse the misaki phonemization + speaker ID assignment from `prepare_rasa_mr.py`.

## 1. Rasa Marathi

```bash
# Download Rasa Marathi split via HF CLI (see its repo for auth)
python scripts/data_prep/prepare_rasa_mr.py \
    --hf-dataset ai4bharat/Rasa \
    --language marathi \
    --out-audio dataset/audio/rasa \
    --out-manifest training/rasa_mr.txt
```

What it does:

- Streams the Marathi split, resamples to 24 kHz, saves as `rasa/marathi_{female,male}_NNNNN.wav`
- Phonemizes the Devanagari text through `misaki.espeak.EspeakG2P(language="mr")` with NBSP cleanup
- Writes manifest rows as `rasa/marathi_female_NNNNN.wav|<IPA>|marathi_female`

Expected output: ~5,500 female + ~8,400 male utterances = **13,900 total**.

## 2. IndicVoices-R Marathi

IV-R streaming from HuggingFace requires an `HF_TOKEN` with gated access to `ai4bharat/indicvoices_r`.

```bash
python scripts/data_prep/prepare_indicvoices_r_mr.py \
    --out-audio dataset/audio/indicvoices_r \
    --out-manifest training/indicvoices_r_mr.txt \
    --stats-out training/indicvoices_r_mr_stats.json
```

What it does:

- Streams the Marathi subset
- Filters on duration (2-15 s), SNR, CER, and scenario (read-speech preferred)
- Hashes original speaker IDs to stable short form (`mr_sXXXX`) so speaker IDs fit the Kokoro vocab budget
- Resamples to 24 kHz, phonemizes the same way Rasa does

Expected output: **~11,910 utts across 329 speakers** after filtering.

## 3. Speaker ID integerization

StyleTTS2's `meldataset.py` does `int(speaker_id)` on every manifest row. Speaker names (`marathi_female`, `mr_s1418`) will crash the dataloader. `fix_speaker_ids.py` converts names to integers and writes a reversible mapping:

```bash
python scripts/data_prep/fix_speaker_ids.py \
    --manifests training/rasa_mr.txt training/indicvoices_r_mr.txt \
    --out-map training/speaker_map.json
```

Output: `speaker_map.json` like `{"marathi_female": 0, "marathi_male": 1, "mr_s1418": 2, ...}`. The manifests are rewritten in place with the integer IDs.

## 4. Duration filter

Very long utterances are memory hazards during training. We cap at **6 s**:

```bash
python scripts/data_prep/filter_by_duration.py \
    --manifest training/rasa_mr.txt \
    --max-seconds 6 \
    --audio-root dataset/audio
# also on IV-R
python scripts/data_prep/filter_by_duration.py \
    --manifest training/indicvoices_r_mr.txt \
    --max-seconds 6 \
    --audio-root dataset/audio
```

Each run snapshots the source manifest to `*.full.txt` on first invocation, so you can re-filter with different thresholds later without losing data.

## 5. Merge + train/val split

Combines Rasa and IV-R into a single manifest with a stratified 95/5 split per speaker:

```bash
python scripts/data_prep/merge_manifests.py \
    --inputs training/rasa_mr.txt training/indicvoices_r_mr.txt \
    --train-out training/train_list.txt \
    --val-out training/val_list.txt \
    --val-frac 0.05
```

Reference counts after filter + merge: **24,676 train / 1,134 val**.

## 6. Out-of-domain text set

Stage 2 needs a Marathi OOD text set for adversarial prosody training. One is included in this repo:

```bash
ls training/OOD_texts.txt
```

If you want to regenerate (e.g. with different length distribution), use:

```bash
python scripts/data_prep/generate_ood_marathi.py \
    --out training/OOD_texts.txt \
    --num 2000
```

## 7. Sanity checks

Before training, run the three diagnostic scripts:

```bash
# 1. Symbol table alignment and tokenizer coverage
python scripts/diagnostics/diagnose_nan.py

# 2. Audio + mel integrity on random samples
python scripts/diagnostics/diagnose_data.py

# 3. IPA length distribution (PLBERT overflows at > 510 chars)
python scripts/diagnostics/check_ipa_lengths.py
```

`diagnose_nan.py` MUST show `✓ every char in train_list is covered by symbol table`. A single unknown char silently drops tokens at training time and will cause mysterious NaN losses once it enters a batch.

## 8. Convert Kokoro base weights

The fine-tune starts from the published Kokoro-82M checkpoint. We strip `module.` prefixes and wrap in `{'net': ...}` to match StyleTTS2's expected format:

```bash
python scripts/convert_kokoro_weights.py \
    --in  path/to/kokoro-v1_0.pth \
    --out training/kokoro_base.pth
```

Output: `training/kokoro_base.pth` (≈ 312 MB), which the training config's `pretrained_model` field points at.

## Ready to train

At this point you have:

```text
dataset/audio/rasa/            — 13,900 24 kHz wavs
dataset/audio/indicvoices_r/   — ~12,000 24 kHz wavs
training/train_list.txt        — 24,676 utts
training/val_list.txt          — 1,134 utts
training/speaker_map.json      — speaker name → int
training/kokoro_base.pth       — converted Kokoro-82M base
training/OOD_texts.txt         — Marathi OOD set for Stage 2
```

See `TRAINING_GUIDE.md` for the pod-side flow.
