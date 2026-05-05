# v0.5 — language conditioning via FiLM at the predictor

## TL;DR

v0.4's design (add lang_embedding to PLBERT input embeddings) is architecturally
blocked. We confirmed this with a hand-injection test: scaling the trained
lang_embedding row 1 (en) up to norm 0.5 — 29× its trained value of 0.0172 —
produced **audibly identical output** vs the un-scaled baseline. Even at norm
5.0 the post-add direction is preserved but the audio doesn't change, because
PLBERT's embedding LayerNorm renormalizes magnitude per token, and the predictor
appears insensitive to the resulting direction tilt.

v0.5 fixes this by injecting lang_id **at the predictor**, after PLBERT, via
**FiLM** (Feature-wise Linear Modulation): learned per-language gain γ and bias
β applied to the predictor's text_encoder output. There is no LayerNorm
between FiLM and the LSTM/duration head; the gain/bias survives end-to-end.

## Why FiLM specifically

| Approach | Pro | Con |
|---|---|---|
| ~~Add lang_emb at PLBERT input~~ (v0.4) | minimal change | LayerNorm washes magnitude; squashed by 12 transformer layers |
| Multi-layer add inside PLBERT | direction survives more | every layer has LayerNorm; signal still attenuated |
| Concat lang_emb to style vector | survives, no LN at predictor | doubles style dim; couples lang info into style space |
| **FiLM at `predictor.text_encoder` output** | **multiplicative + additive; no LN downstream; tiny param count; well-known to work for conditioning** | needs forward override on AdaINResBlk / dur encoder path |
| FiLM at decoder input | even more direct | decoder doesn't naturally receive non-style scalars; bigger surgery |

FiLM is canonical for "tell this network to do mode A vs mode B". 256 params per
language for `[γ; β]` over 128 channels — trivial. And critically: the
predictor's text_encoder output goes directly into LSTM → duration_proj →
F0Ntrain. No LayerNorm in that path. **What we apply at FiLM is exactly what
the duration / F0 / energy heads see.**

## What we measured (v0.4 evidence trail)

| Datapoint | Value |
|---|---|
| ep4 lang_embedding row 0 (mr) norm | 0.0117 |
| ep4 lang_embedding row 1 (en) norm | 0.0159 |
| ep4 cosine(mr, en) | -0.12 (rows directionally diverging) |
| ep4 ratio vs typical word_embedding | 2.44% |
| ep5 lang_embedding row 0 norm | 0.0137 |
| ep5 lang_embedding row 1 norm | 0.0172 |
| ep5 ratio vs word_embedding | 2.72% |
| Per-epoch growth rate | 1.12× |
| Linear extrapolation at ep9 | 4.25% |
| WITH vs WITHOUT lang_ids (ep4 + ep5) | audibly identical |
| Hand-inject lang_emb[1] @ norm 0.5 (29×) | audibly identical |

The hand-inject test rules out "we just didn't train the embedding enough" —
the predictor downstream of PLBERT genuinely cannot act on the lang signal at
the BERT input. The architecture is the bottleneck.

## Architecture changes (concrete)

### 1. New FiLM module — `Modules/film.py`

```python
class LangFiLM(nn.Module):
    """Feature-wise linear modulation conditioned on language id.

    output = γ[lang_id] * x + β[lang_id]

    γ initialized to 1, β to 0  →  identity at init (no-op).
    Both γ and β are learnable per-language vectors of size `channels`.
    """
    def __init__(self, num_languages: int, channels: int):
        super().__init__()
        self.gamma = nn.Embedding(num_languages, channels)
        self.beta  = nn.Embedding(num_languages, channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]   lang_ids: [B, T]   →  out: [B, T, C]
        g = self.gamma(lang_ids)   # [B, T, C]
        b = self.beta(lang_ids)    # [B, T, C]
        return g * x + b
```

### 2. Wire into the predictor — `models.py` `ProsodyPredictor`

```python
class ProsodyPredictor(nn.Module):
    def __init__(self, ...):
        ...
        self.lang_film = LangFiLM(NUM_LANGUAGES, channels=...)  # match text_encoder out dim

    def forward(self, ..., lang_ids):
        d = self.text_encoder(d_en, s, input_lengths, text_mask)
        d = self.lang_film(d, lang_ids)        # ← inject HERE, before LSTM
        x, _ = self.lstm(d)
        duration = self.duration_proj(x)
        ...
```

`F0Ntrain` and the alignment-applied features downstream all see the FiLM-modified
representation.

### 3. Plumb `lang_ids` through call sites — `train_second.py`, `synth_v0_4.py`

`predictor(...)` calls already get `text_mask`/`input_lengths`; we add `lang_ids`.
Already plumbed through PLBERT in v0.4, so the dataloader emits it. Just thread
it one level deeper.

### 4. Keep v0.4's `bert.lang_embedding` — but only as auxiliary signal

Don't remove it; it provides a minor extra cue at the BERT input. Just stop
relying on it as the *primary* lang signal. (Optional: zero-init it again and
let it train alongside FiLM — costs nothing.)

## Why FiLM should escape v0.4's failure modes

1. **No LayerNorm downstream.** The predictor LSTM consumes FiLM output directly.
2. **Multiplicative gating** — even small γ deviations from 1.0 produce nonlinear
   downstream effects; can't be normalized away by anything.
3. **Per-channel control.** γ and β are 128-d each; FiLM can independently gate
   different feature channels (some channels for /z/, others for diphthongs, etc.).
4. **Identity at init** (γ=1, β=0) — initial behavior is exactly v0.4 ep5, so
   we can't regress. Gradient signal grows γ/β where it helps.
5. **Direct gradient path** — FiLM γ/β receive gradient through the duration,
   F0, energy, mel losses. No 12 transformer layers of attenuation upstream.

## Smoke design — distinguishes "FiLM works" from "still architecturally blocked"

~$2, ~1h on the existing pod (already warm).

- **Init**: v0.4 ep5 (the polish + lang_embedding[1] at norm 0.0172, FiLM γ=1 β=0)
- **Manifest**: train_list_smoke_intspk.txt (500 rows, 9.98% flip)
- **Epochs**: 2 (1 frozen warmup + 1 unfrozen) — same recipe as smoke #6
- **freeze_backbone_epochs: 1** but UNFREEZE FiLM from epoch 0 (FiLM is a new
  param like lang_embedding; we want gradients flowing into it from step 1)

### Smoke gates (decision points)

| Signal | Pass | Action |
|---|---|---|
| FiLM γ row 1 (en) — distance from 1.0 vector | > 0.05 mean abs | Hypothesis "FiLM picks up signal" confirmed |
| FiLM β row 1 (en) — norm | > 0.02 | Same |
| WITH vs WITHOUT lang_ids audio | detectably different | **FiLM works → run full v0.5** |
| Predictor_encoder norm | > 50 (no collapse) | Architecture stable |
| Validation loss | < 0.30 | No regression vs v0.4 |

If WITH/WITHOUT differ but γ/β haven't moved much, that's still a Pass — means
the lang_embedding from v0.4 + minor FiLM perturbation is enough.

If WITH/WITHOUT identical AND γ/β still ≈ 1/0 → FiLM also bottlenecked, escalate
to FiLM at decoder + Multilingual PL-BERT init.

## Full run (if smoke passes)

- **Init**: v0.4 ep5 (or smoke's final ckpt)
- **Manifest**: train_list_v0_4.txt (freq2, 7.21% — same as v0.4 full run)
- **Epochs**: 5 (we've already polished 5 epochs in v0.4; 5 more is enough)
- **Cost**: ~$8

## Success criteria (no more "sounds bit cleaner")

1. **WITH vs WITHOUT lang_ids: audibly different** on en-tagged tokens for
   at least 7 of 10 fixed test sentences (blind A/B)
2. **/z/ rendering**: synth "amazing", "zoo", "easy", "Brazil" — ≥ 3/4 sound
   like /z/ not /dʒ/
3. **No regression on Marathi**: existing 4 retroflex test sentences from
   `kokoro_tb_utils.py` sound at least as good as v0.4 ep5
4. **FiLM γ for row 1 (en)**: at least one channel with |γ−1| > 0.3 by full-run
   epoch 5 — quantitative confirmation that FiLM is doing real work

## Cost ladder

| Stage | Cost | Time | Outcome |
|---|---|---|---|
| FiLM smoke (2 epochs, smoke manifest) | ~$2 | 1h | Go/no-go on FiLM |
| Full v0.5 (5 epochs, freq2 manifest) | ~$8 | 5h | Final ckpt → ONNX → webgpu |

Total: ~$10, ~6h.

## v0.4 artefacts kept

- `epoch_2nd_00005.pth` on pod — best v0.4 ckpt, init source for v0.5
- All v0.4 logs — empirical evidence of v0.4 limit
- `models.py` load_checkpoint fix — keeps for v0.5
- `freeze_backbone_epochs` recipe — works, keeps

## Open questions before launching smoke

1. What's the exact channel dim of `predictor.text_encoder` output? (Should be
   `hidden_dim` from config = 512, but verify by reading models.py)
2. Should FiLM be applied per-token (as designed above) or per-utterance
   (single γ, β per sequence)? Per-token is what we want for code-switching;
   confirm `lang_ids` shape matches.
3. Should we also add FiLM at `predictor.F0Ntrain`? It receives the encoded
   features after duration alignment; another natural conditioning point.
4. NUM_LANGUAGES = 2 (mr, en) for now. Future: bump to N for hi/te/ta if we
   ever expand.
