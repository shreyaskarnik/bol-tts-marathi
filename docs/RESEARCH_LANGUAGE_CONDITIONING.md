# Language Conditioning for Code-Switched Indic TTS

Design doc for adding per-token language conditioning to bol-tts-marathi to
improve Marathi-English code-switching ("Minglish") synthesis. Written after
v0.1 final and the SPRINGLab/Svara evaluation revealed that *all* current
Indic TTS systems — including SOTA multilingual ones — degrade on embedded
English content (truncation, wrong stress, vowel substitution).

Status: **proposed for v0.4** (not blocking v0.2 / v0.3 lexicon work).

## 1. Problem statement

Marathi (and most Indic languages) is heavily code-switched in real-world use.
A typical urban Marathi sentence contains 1–3 English content words:

> _"मी Google मध्ये काम करतो, project खूप interesting आहे."_

Current handling in bol-tts-marathi (v0.1):

- espeak-ng phonemizes Marathi via `mr` voice, English via `en` voice, emits
  `(en)`/`(mr)` language-switch tags around code-switched runs.
- We parse those tags in `phonemizer.ts::parseTaggedIpa` and apply per-segment
  remap rules: `MARATHI_FIXES` for mr segments, plus `ENGLISH_INDIC_EXTRAS`
  for en segments (Indic-English vowel/fricative shifts).
- The model itself sees a flat phoneme stream with no language information.
  PLBERT, the prosody predictor, and the decoder all treat phonemes as
  language-agnostic.

Failure modes observed (in bol-tts AND Svara TTS, a strong baseline):

| Input | Expected | Actual | Cause |
|---|---|---|---|
| `morning` (in Marathi context) | "MOR-ning" | "mon" | English phoneme tail truncated |
| `tickets` (in Marathi context) | "TI-kets" | "tika" | Wrong consonant cluster handling |
| `ɐmˈeɪzɪŋ` (with Marathi voice) | "amazing" | varies | Sparse training signal for English-only phonemes in Marathi-voice context |

**Root cause**: the model has no way to learn that the prosodic distribution
for English-tagged phonemes should differ from Marathi-tagged phonemes. With
~99% Marathi training data, English phonemes get treated as "weird Marathi"
and produce out-of-distribution durations and F0 contours — and **durations
are decided in the PLBERT/predictor path**, not in the phoneme TextEncoder.

## 1.1 Empirical evidence (v0.1 inference test)

Before committing to architecture changes, ran a controlled test to
disconfound two possible failure paths:

  (a) acoustic decoder degraded for English (would manifest as garbled words)
  (b) duration / prosody predictor stuck on Marathi rhythm (would manifest
      as intelligible-but-wrongly-paced English)

**Method**: 10 pure-English sentences (no Devanagari) — 5 derived from the
webgpu-demo Minglish chips, 5 extended for stress and length coverage —
phonemized through espeak `en-us` with our existing `ENGLISH_INDIC_EXTRAS`
rules, then synthesized through the v0.1 checkpoint
(`kokoro-mr-v1_0.pth`) with the `mm_mukta` voicepack on CPU. Test script
at `experiments/v0_4_lang_conditioning/scripts/bol_english_only_test.py`.

**Findings** (subjective listening, multiple samples):

| Sample | Observation | Diagnostic |
|---|---|---|
| 01 "I work at Google" | Intelligible but "1st-grader" cadence | Acoustic OK, prosody broken |
| 02 "...meeting today" | "meeting" sounds **American** (audible base-Kokoro English) | Acoustic decoder retains base-model English |
| 03 "Shall we have some coffee?" | Same 1st-grader pattern, words clear | Confirms (b), rules out (a) |
| 04 "Artificial intelligence is amazing technology" | "amaʤing" — `z→ʤ` rule audible (intentional) | Frontend rules reaching the model as expected |
| 04 "weekend → weekenda" | Schwa appended after word-final /d/ | **Marathi phonotactic interference** — model applies vowel-epenthesis pattern from Marathi-final stop clusters to English. Diagnostic for (b) at the prosodic level, not phoneme level. |

**Conclusion**: the failure mode is firmly **(b) "intelligible + Marathi
pace"**, not (a). The acoustic decoder retains its base-Kokoro English
phoneme→audio mapping; the broken layer is duration / F0 prediction
(`predictor.text_encoder` + `duration_proj` + F0/N predictor at
`kokoro/model.py:105–115`), which was Marathi-fine-tuned and now applies
Marathi prosodic patterns to all phoneme runs regardless of language.

**Implications for v0.4 scope**:

1. **Drop the optional phoneme TextEncoder injection** from §3.1. The
   acoustic side isn't degraded — adding language conditioning there
   would solve a problem that doesn't exist.
2. **Narrow v0.4 to PLBERT-only conditioning**. The duration/prosody path
   is where the language signal must reach.
3. **v0.3 lexicon expansion alone is insufficient.** Word-level pronunciation
   overrides cannot fix system-level prosodic patterns like
   stop-release schwa epenthesis.

## 1.2 A simpler data path: Rasa already contains code-switched English

While inspecting trimmed Rasa samples (during v0.2 prep), spot-checks revealed
that the Rasa Marathi corpus contains substantial English loanword content —
just transliterated to Devanagari script in the source text:

| File | Source text | mr-G2P IPA | English content |
|---|---|---|---|
| `marathi_female_03178.wav` | "हल्लीच आम्ही एक **सॅनिटरी व्हेंडिंग मशीन** बसवलंय..." | `... sɛnɪʈɾi vẽːɖĩɡ məʃiːn ...` | sanitary / vending / machine |
| `marathi_female_01405.wav` | "तुम्ही माझा **टीव्ही** दुरुस्त..." | `... ʈiːvhi ...` | TV |

**Implications**: Rasa already contains thousands of (text, audio) pairs for
real human Marathi-accented English loanwords. We don't need synthetic CS
data (Svara, splice pipeline, IndicCMix audio) — the gold-standard data is
hidden inside our existing Rasa corpus, just labeled with the wrong-script
text. Three properties make this attractive:

1. **Real human audio** — better than any synthesis path (F5-TTS, Parler-Indic,
   splice). The Marathi-accented English is exactly what users actually
   produce when code-switching.
2. **Already aligned** — text and IPA already paired with audio in v0.1's
   training manifest. No forced alignment, no segmentation, no synth.
3. **Distribution match** — Rasa speakers' Marathi-accented English is the
   target distribution for inference. Training on it teaches lang_id=en =
   "Indian English as Marathi speakers actually pronounce it," which is
   what users want when they type Latin "meeting" in a Marathi sentence.

**The IPA-routing catch**: today, Rasa's transliterated English goes through
mr-G2P at training time (since the source text is Devanagari). At
inference, when a user types Latin "machine", the demo routes through
en-G2P with `ENGLISH_INDIC_EXTRAS` rules. The two pipelines produce
*different* IPA for the same intended word:

```
"मशीन"    → mr-G2P → məʃiːn         (training sees this)
"machine" → en-G2P → məʃˈiːn         (inference produces this)
```

The model has never seen the en-routed IPA paired with Marathi-accented
English audio. That's likely a meaningful contributor to the §1.1 "1st-grader
cadence" findings.

**Two ways to leverage this**:

| Approach | Mechanic | Train ↔ inference IPA match? |
|---|---|---|
| **A. Re-phonemize via en-G2P after romanization** | Identify English loanwords → romanize Devanagari→Latin → re-phonemize whole row via espeak with `(en)/(mr)` tags → audio stays the same. | ✅ Yes — identical pipeline at train and inference. |
| **B. Keep mr-IPA, romanize text only** | Text becomes mixed-script + lang_ids derived from script, but the IPA in the manifest stays as mr-routed. | ❌ No — same train/inference mismatch as today. |

**Recommendation: A.** The IPA-pipeline-matching is the higher-order fix.
With approach A, v0.4 trains on real Marathi-accented English audio with
en-routed IPA matching what inference produces. lang_id=en correctly tells
PLBERT "this is English content that needs English-shaped durations" — and
the durations the model learns ARE the Marathi-accented-English durations
because that's what the training audio contains. Win-win.

**The loanword identification step**: how do we tell which Devanagari words
are English? IndicCMix's parallel `native_script_codemixed` ↔
`full_native_script` columns are explicitly aligned word-for-word per their
dataset card, with English in Latin in one column and the same word
transliterated to Devanagari in the other. Word-pair alignment across their
~9,500 Marathi rows yields a free Devanagari→Latin loanword dictionary
(estimated ~3-5K entries). No model, no classifier needed.

**Net strategic shift for v0.4 data prep**:

| Was planned | Now |
|---|---|
| IndicCMix audio splice pipeline (Parler-Indic-IE for English chunks + Marathi recordings) | Skipped — Rasa already has the English |
| Parler-Indic-IE re-synth at 24 kHz | Skipped |
| F5-TTS English chunks from IndicCMix | Skipped (already known unusable: 16 kHz + American accent) |
| **IndicCMix audio** | **Skipped — text-only role: source the loanword dictionary from `native_script_codemixed` ↔ `full_native_script` alignment** |
| **Rasa training data** | **Re-prepped with romanized English loanwords + lang_ids per script** |

This is dramatically cheaper than the prior synthetic-CS pipelines and uses
a higher-quality data source. Compute cost goes from ~$10-20 of synth + ~$48
of v0.4 training to ~$0 of data prep + ~$43 of v0.4 training.

## 2. Prior art

Surveyed in late v0.1; closest neighbors:

| Paper | Architecture | Key idea | What we'd do differently |
|---|---|---|---|
| [Indonesian-English STEN-TTS](https://arxiv.org/abs/2412.19043) (Dec 2024) | STEN-TTS | Per-word BERT-LID + multilingual base | Not StyleTTS2; uses BERT classifier (heavy) instead of espeak tags (free) |
| [Hindi+Indian-English Parler-TTS](https://arxiv.org/abs/2506.16310) (Jun 2025) | Parler-TTS | **Sentence-level** natural-language style prompts via Flan-T5 + RVQ-based discrete audio + 3-stage curriculum (accent → language → emotion) — 23.7% WER reduction | Not StyleTTS2; their conditioning is *coarser* (per-utterance style description, not per-token). Architectural pieces (Flan-T5, RVQ, decoder-only) are non-transferable. **Recipe ideas (3-stage curriculum, datasaspeech auto-feature-tagging) are transferable.** |
| [CS-LLM](https://arxiv.org/abs/2409.10969) (Sep 2024) | LLM-based TTS | Synthetic CS data via splice-and-concat from monolingual | Different architecture; relevant for v0.4 synthetic-data plan |
| [F5-TTS-RO](https://arxiv.org/abs/2512.12297) (Dec 2025) | F5-TTS | Lightweight input-adapter, freeze base | Architectural pattern (frozen base + adapter) we could borrow |

**What's still open for bol**:

1. **StyleTTS2/Kokoro-family with language conditioning** — none of the
   surveyed papers use this architecture. The transfer of the technique is
   not a foregone conclusion (different prosody predictor, different
   discriminator, different style conditioning).
2. **Marathi specifically** — the Hindi paper is closest, but Marathi has its
   own conventions (retroflex `ɭ`, schwa-deletion patterns, distinct vowel
   inventory).
3. **Free training labels from espeak-ng tags** — Indonesian paper uses BERT
   for LID; using existing espeak `(en)`/`(mr)` tags is "free" and avoids a
   classification dependency.

## 3. Proposed approach

### 3.1 Where to inject lang_ids: the PLBERT/predictor pathway

The first-class target is the **duration/prosody path**, not the phoneme
TextEncoder. Walking `forward_with_tokens` in
`kokoro-deutsch/kokoro/kokoro/model.py:87`:

```python
# Duration path — drives all prosody and pred_dur:
bert_dur = self.bert(input_ids, attention_mask=...)   # line 102: PLBERT
d_en     = self.bert_encoder(bert_dur).transpose(-1, -2)  # line 103
d        = self.predictor.text_encoder(d_en, s, ...)  # line 105
x, _     = self.predictor.lstm(d)                     # line 106
duration = self.predictor.duration_proj(x)            # line 107: pred_dur

# Acoustic path — feeds the decoder, no influence on durations:
t_en = self.text_encoder(input_ids, ...)              # line 116
asr  = t_en @ pred_aln_trg
audio = self.decoder(asr, F0_pred, N_pred, ...)        # line 118
```

Because the failures we are trying to fix (English truncation, wrong
durations, drifted F0) live in `pred_dur` and the F0/N predictor, the
conditioning **must reach `bert` and `predictor.text_encoder`**. Patching
only the phoneme `text_encoder` (line 116) influences only the decoder's
acoustic features after durations are already locked in.

**Primary injection point: PLBERT input.** Add a `lang_embedding` table
(small: `~3 entries × hidden_dim`) and add its lookup to PLBERT's input
embeddings before the transformer stack:

```python
# in models.py / wherever PLBERT is wrapped
def forward(self, input_ids, lang_ids, attention_mask=...):
    tok_emb  = self.bert.embeddings.word_embeddings(input_ids)
    lang_emb = self.lang_embedding(lang_ids)          # [B, T, D]
    h        = self.bert.embeddings.LayerNorm(tok_emb + lang_emb + pos_emb)
    return self.bert.encoder(h, attention_mask=...)
```

This way every downstream consumer of `bert_dur` (the bert_encoder, the
predictor's text_encoder, the duration projection, F0/N predictor) sees the
language signal. No additional changes are required in the predictor; PLBERT
carries the conditioning through.

**Phoneme TextEncoder injection — not pursued.** Originally proposed as an
optional secondary injection point. Removed from v0.4 scope after the §1.1
empirical test confirmed the acoustic decoder retains base-Kokoro English
phoneme→audio mapping and is not the broken layer. Adding language
conditioning to the phoneme TextEncoder would solve a problem that doesn't
exist. See §3.4.

**Why additive over concat**: keeps the input dimension to PLBERT
unchanged, no other module touches change. Concat would force projection
layers everywhere downstream.

### 3.2 Lang_ids label space

`lang_ids` is a `[B, T]` int tensor with values in `{0=mr, 1=en, 2=shared}`,
parallel to `input_ids`. Punctuation, stress marks, and language-neutral
tokens are `shared`. Labels are computed once during phonemization from
espeak-ng's `(en)`/`(mr)` tags — the same tags we already parse in
`phonemizer.ts::parseTaggedIpa` for inference. No classifier, no extra
dependency.

### 3.3 Borrowed from Parler-Hindi (auxiliary, not architecture)

After reading [Optimizing Multilingual TTS with Accents & Emotions
(arXiv:2506.16310)](https://arxiv.org/abs/2506.16310) in detail, two of their
techniques are worth adopting at the **data-prep layer** (not model
architecture, since their decoder-only Parler-TTS is incompatible with
StyleTTS2):

- **Auto-feature-tagging via [`datasaspeech`](https://github.com/ylacombe/dataspeech)**:
  annotate each training utterance with SNR, speaking rate, reverberation,
  monotony, and inferred emotion tags. Adds rich conditioning signal at zero
  manual labelling cost. Recommended for v0.4 manifest extension. ~50 LOC
  in data prep.
- **Three-stage training curriculum**: instead of (Stage 1) → (Stage 2),
  consider (Stage 1: monolingual Marathi base) → (Stage 2a: Marathi + Indian
  English mix) → (Stage 2b: emotion-tagged subset). Doesn't require
  architecture changes — purely a manifest-and-config curriculum.
  Speculative whether it helps; worth a short pilot (~5 epochs each) before
  committing.

What we **don't** borrow from Parler-Hindi:

- Their natural-language style prompts via Flan-T5 — incompatible with
  StyleTTS2's audio-extracted `ref_s` style vector. Adopting it would
  require building a parallel text-style encoder, which is a much larger
  architectural change than the language conditioning we're already proposing.
- RVQ discrete audio tokens via DAC — StyleTTS2 uses continuous mel features
  + ISTFTNet, fundamentally different audio pipeline.

### 3.4 Out of scope for v0.4

- **Phoneme TextEncoder injection.** Originally a v0.4 secondary arm; removed
  per §1.1 empirical test. The acoustic decoder retained base-Kokoro
  English phoneme→audio mapping through Marathi fine-tuning, so per-token
  language conditioning at the acoustic-side TextEncoder is solving a
  problem that doesn't exist. Could be revisited in v0.5+ if PLBERT-only
  conditioning underdelivers, but the test gave no signal that it's
  needed.
- **Token-weighted adversarial loss in Stage 2.** Originally proposed here as
  a parallel arm. Removed because the discriminators in
  `kokoro-deutsch/StyleTTS2/losses.py:106` (MPD) and `losses.py:177` (MSD)
  produce **audio-domain critic outputs** — multi-period and multi-scale
  scores over the *waveform*, not over input tokens. Multiplying their
  outputs by a `[B, T_token]` language mask requires first inventing a
  token-to-audio-frame-to-critic-window alignment. That alignment is
  non-trivial (durations from `pred_dur` map tokens to mel frames; the
  discriminators then operate on samples or mel frames at a different rate
  again). Without a concrete alignment design, this is hand-waving. **Defer
  to v0.5** once duration alignment is verifiable from a trained v0.4
  checkpoint.
- Per-language style projections (`style = base_speaker + lang_delta`).
  Higher cost, breaks voicepack format compatibility. Worth revisiting only
  if 3.1 underdelivers.
- Cross-lingual PLBERT pre-training. Most expensive change; uncertain ROI.
- Per-language phoneme partitioning. Loses cross-lingual sharing; probably
  wrong direction.
- Natural-language style prompt conditioning (Parler-style). Architecturally
  incompatible with StyleTTS2; would be a v0.6+ project of building a
  different model.

## 4. Implementation plan

### 4.0 Code organization — separate experimental directory

This is **research code, not production-track**. To avoid contaminating
v0.2/v0.3 main-line work and to preserve a clean A/B baseline, all
language-conditioning experiments live under their own directory tree:

```
bol-tts-marathi/
├── experiments/
│   └── v0_4_lang_conditioning/
│       ├── README.md              # this design doc, scoped to the experiment
│       ├── configs/
│       │   └── config_marathi_v0_4_langcond.yml
│       ├── scripts/
│       │   ├── build_v0_4_manifest.py    # rebuild manifests from raw text + lang_ids
│       │   └── train_v0_4.sh              # launch wrapper
│       ├── scripts/
│       │   └── bol_english_only_test.py    # §1.1 empirical test reproducer
│       ├── styletts2_patches/     # diffs to apply to the StyleTTS2 fork
│       │   ├── 01_lang_embedding_plbert.patch
│       │   └── README.md
│       ├── kokoro_patches/        # diffs to apply to the kokoro inference fork
│       │   └── 01_forward_with_tokens_lang_ids.patch
│       └── notes/                  # experiment logs, ablation results
│           └── pilot_runs.md
```

**Rationale**: v0.2 / v0.3 / v0.4 are independent training axes (data
recipe, frontend lexicon, model architecture respectively). They should
ship orthogonally. If v0.4 underperforms, we revert by ignoring the
`experiments/v0_4_lang_conditioning/` directory; the rest of the repo is
untouched. If it works, we promote selected pieces back into the main
`scripts/` and `configs/` directories under a clear v0.4 release.

The fork patches live as patch files (not direct submodule edits) so the
parent submodule pins don't drift. To apply:

```bash
cd kokoro-deutsch/StyleTTS2 && git apply ../../bol-tts-marathi/experiments/v0_4_lang_conditioning/styletts2_patches/01_lang_embedding_plbert.patch
cd kokoro-deutsch/kokoro && git apply ../../bol-tts-marathi/experiments/v0_4_lang_conditioning/kokoro_patches/01_forward_with_tokens_lang_ids.patch
```

### 4.1 Files to touch

**Training (semidark/StyleTTS2 fork):**

| File | Change | LOC est |
|---|---|---|
| `models.py` (PLBERT wrapper) | Add `lang_embedding` table; sum into PLBERT input embeddings before encoder stack | ~30 |
| `meldataset.py` | Read `lang_ids` column from manifest, batch alongside `input_ids` with matching pad/length handling | ~50 |
| `train_first.py` / `train_second.py` | Pass `lang_ids` through model forward calls | ~30 |
| **Total in StyleTTS2** | | **~110 LOC** |

**Inference (kokoro-deutsch/kokoro fork):**

| File | Change | LOC est |
|---|---|---|
| `kokoro/model.py:87` (`forward_with_tokens`) | Add `lang_ids: torch.LongTensor` param; pass into `self.bert(...)` and (optionally) `self.text_encoder(...)` | ~10 |
| `kokoro/model.py` (`forward`, public API) | Accept lang_ids alongside phonemes; thread through to `forward_with_tokens` | ~10 |
| `kokoro/pipeline.py` (if calling `model.forward`) | Accept tagged input from phonemizer, derive lang_ids | ~15 |
| **Total in kokoro inference** | | **~35 LOC** |

**bol-tts-marathi (data + export + browser):**

| File | Change | LOC est |
|---|---|---|
| `experiments/v0_4_lang_conditioning/scripts/build_v0_4_manifest.py` | Rebuild manifests **from raw text** (not retrofit v0.2 phoneme strings): re-phonemize with `(en)`/`(mr)` tags retained, derive lang_ids alongside | ~80 |
| `configs/config_marathi_v0_4.yml` | New v0.4 config with `use_language_conditioning: true` | ~5 |
| `scripts/export_onnx.py` | Add `lang_ids` (int64, dynamic axis `[1, T]`) to ONNX input signature; verify the static graph still traces | ~15 |
| `webgpu-demo/src/tokenize.ts` | Mirror the lang_id labeling at inference: pass espeak `(en)`/`(mr)` tags through, emit parallel lang_ids array next to input_ids | ~30 |
| `webgpu-demo/src/model.ts:73` (`synthesize`) | Add `langIds: number[]` arg; build `int64 [1, T]` tensor; pass `lang_ids` alongside `input_ids` and `ref_s` to `this.model({...})` | ~10 |
| Caller chain: `main.ts` → `phonemizer.ts` → `model.ts` | Plumb the parallel lang_ids array end-to-end | ~20 |
| **Total in bol-tts** | | **~160 LOC** |

**Total: ~305 LOC across three repos.**

### 4.2 Manifest format — clean rebuild from raw text

v0.4 manifests are **rebuilt from raw text**, not retrofitted from v0.1/v0.2
phoneme strings. The retrofit path is brittle: existing manifests already
collapsed the `(en)`/`(mr)` tags during phonemization, so we can't recover
ground-truth lang_ids per phoneme without re-running the phonemizer anyway.
Cleaner to do it once, properly, from the source text.

`build_v0_4_manifest.py` reads each dataset's raw text + speaker labels,
calls the same `(en)`/`(mr)`-aware espeak-ng wrapper used at inference, and
emits a v0.4 manifest line:

```
rasa/marathi_female_00001.wav|kˈəʈ iŋɡ tʃˈai|0 0 0 0 1 1 1 0 0 0 0 0|marathi_female
```

The third pipe-separated column is space-separated lang_ids per phoneme
(0=mr, 1=en, 2=shared). v0.4 training reads only v0.4 manifests; v0.1/v0.2
manifests are not consumed. This is a clean break — no
backward-compatibility shim, no fallback "all-mr" inference of lang_ids
from old manifests.

### 4.3 Open design decisions

The user (or whoever picks this up) should weigh in on:

1. **`shared` class — needed?** Punctuation, stress marks, and silence-like
   tokens could be `shared` (lang_id=2) or just default to `mr`. Adding
   `shared` is more principled but adds a degree of freedom the model may
   not benefit from. Recommendation: start with binary {mr, en} and revisit
   only if needed.
2. **Inference fallback at the demo**: when ONNX export consumes the
   lang_ids input, what should the demo do for users who don't tag their
   input? Default all to mr=0, or run a mini language-classifier (regex on
   Latin chars) at the JS layer? Recommend the latter — auto-detect Latin
   runs as English at inference time, mirroring espeak's training-time
   behavior. The phonemizer already has the tag info; we just need to
   surface it as lang_ids.

## 5. Training plan

### 5.1 Approach A: train from scratch (clean baseline)

Cold start: Stage 1 + Stage 2 from `kokoro_base.pth` with v0.4 config.

- Pros: clean comparison, no init biases from v0.1's already-learned
  language-agnostic representations.
- Cons: ~$50 + ~34h on A100. Loses the SPRINGLab improvements from v0.2/2.5.

### 5.2 Approach B: continue from v0.2/2.5 final

Init from `epoch_2nd_00009.pth` of the v0.2 run (assuming it lands first).
Stage 2.5-style continuation: add the new `lang_embedding` table (random
init), set `second_stage_load_pretrained: true`, run 5–10 more Stage 2 epochs.

- Pros: ~$15 + ~25h on A100. Builds on existing improvements.
- Cons: the random-init `lang_embedding` has to find its niche in an already-
  converged model; might produce noisy gradients early.

**Recommendation**: Approach B. Cheaper, builds on v0.2 quality. Mitigate
init-noise risk by warmup-freezing the lang_embedding for the first 200
steps (linearly increase its gradient scale from 0 to 1).

### 5.3 Compute budget

| Phase | Resource | Cost (RunPod A100) |
|---|---|---|
| Pilot: PLBERT-only, 3 epochs sanity (verify lang_embedding learns something, no NaN) | ~3.5h × $1.51 | ~$5 |
| Full Stage 2.5 (PLBERT-only) | ~25h × $1.51 | ~$38 |
| **Total** | | **~$43** |

(Down from ~$48 in the prior version because §1.1 collapsed the two-variant
ablation to a single-variant sanity check.)

## 6. Evaluation plan

Three axes, all blinded A/B vs v0.1 / v0.2 baselines:

### 6.1 Objective metrics

- **Phoneme-level WER on Indic-English** — use Whisper-large or a finetuned
  ASR to transcribe Minglish synthesis, compare to ground-truth English
  spellings. Lower WER = better English intelligibility.
- **Phoneme-level CER on monolingual Marathi** — should NOT regress vs v0.1.
  Verify the language conditioning didn't hurt the dominant case.
- **Duration prediction MAE** — does the prosody predictor produce closer-to-
  ground-truth durations on a held-out code-switch test set? This is the
  metric most directly aligned with the PLBERT-pathway hypothesis.

### 6.2 Subjective evaluation

- **MOS test (n≥10 listeners)** on three slices:
  - Pure Marathi (control — should match v0.1 quality)
  - Pure English embedded in Marathi context (target improvement)
  - Heavy code-switch (3+ English content words per sentence)
- **AB preference test**: v0.1 vs v0.4 on the same Minglish utterances. Listeners
  pick which sounds more natural.

### 6.3 Test set

Build a held-out Minglish eval set (~100 utterances) BEFORE training. Sources:

- Curated from real Marathi tweets / WhatsApp / news with code-switch
- Each utterance manually tagged with ground-truth pronunciation guidance
- Mix of: tech jargon, brand names, time/date expressions, common loanwords,
  dense code-switch (>50% English)

This eval set should be **frozen** before training and not used during
hyperparameter tuning. Otherwise the metrics are train-test contaminated.

## 7. Risk & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Lang embedding doesn't help (no measurable improvement) | Medium | Pilot with 3-epoch ablation first; bail before full run if no signal on duration MAE |
| Marathi quality regresses (interference) | Low–Medium | Per-segment metrics; revert if >5% MOS regression on monolingual |
| ONNX export breaks (new int64 input not exportable cleanly) | Low | Test ONNX round-trip on epoch 1 before full training; lang_ids has the same shape and dtype semantics as input_ids, so torch.onnx should handle it identically |
| Demo regression (wrong lang_ids inference) | Low | Phonemizer already has `(en)`/`(mr)` tags from espeak-ng; lang_ids extraction is deterministic from those tags |

## 8. Publishing pathway (optional)

If results land at >10% MOS preference for v0.4 over v0.1 on the Minglish
slice, the work is plausibly publishable as an applied/systems paper:

> **Language-Conditioned StyleTTS2 for Indic Code-Switched TTS: A
> Reproducible Recipe**
>
> Applied per-token language conditioning at the PLBERT/predictor pathway
> (free labels from espeak-ng) to the StyleTTS2 family. Marathi + Indian
> English. Open dataset, open code. AB-tested vs Parler-TTS Hindi baseline.

Workshop venue: Interspeech / ICASSP code-switch / multilingual TTS
workshops. Not main-track novel ML — this is engineering with measurement,
not new theory.

The bigger value is shipping the better demo. Publishing is the cherry on top
if the engineering pans out.

## 9. References

- [Indonesian-English STEN-TTS, arXiv:2412.19043](https://arxiv.org/abs/2412.19043)
- [Hindi+Indian-English Parler-TTS, arXiv:2506.16310](https://arxiv.org/abs/2506.16310)
- [CS-LLM monolingual-only, arXiv:2409.10969](https://arxiv.org/abs/2409.10969)
- [F5-TTS-RO Romanian adapter, arXiv:2512.12297](https://arxiv.org/abs/2512.12297)
- [bol-tts-marathi v0.1 release notes](../README.md)
- [bol-tts-marathi v0.2 plan: Stage 2.5 with SPRINGLab](STAGE_2_5.md)
