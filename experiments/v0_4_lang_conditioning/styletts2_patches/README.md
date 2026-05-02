# StyleTTS2 fork patches for v0.4 language conditioning

Single combined patch (`01_lang_conditioning.patch`) covering:

| File | Change | LOC |
|---|---|---|
| `Utils/PLBERT/util.py` | `CustomAlbert` now has `lang_embedding` table; forward accepts optional `lang_ids` | ~30 |
| `meldataset.py` | Manifest schema extended to 4-col (`wav\|text\|speaker\|lang_ids`); `Collater` pads + emits a 9th batch element | ~50 |
| `train_second.py` | (1) Two `model.bert(...)` call sites pass `lang_ids` (train + eval); two batch unpacks accept the 9th element. (2) **predictor_encoder reset gate** (#19): in the `load_pretrained` continuation branch, gate the `predictor_encoder = copy.deepcopy(style_encoder)` reset on a new `reset_predictor_encoder_on_load` config flag (default `False`), so v0.2's trained prosody specialization survives v0.4 continuation. | ~24 |
| `train_first.py` | Two batch unpacks accept the 9th element (Stage 1 ignores it via `_`) | ~4 |

Total: ~108 LOC, within the design-doc budget of ~110.

## How to apply

```bash
cd /path/to/StyleTTS2  # the kokoro-deutsch/StyleTTS2 fork
git checkout main      # or wherever you want to base
git apply /path/to/bol-tts-marathi/experiments/v0_4_lang_conditioning/styletts2_patches/01_lang_conditioning.patch
```

Or in this local development setup the patch is already applied on the
`v0_4_lang_conditioning` branch of `kokoro-deutsch/StyleTTS2`:

```bash
cd kokoro-deutsch/StyleTTS2
git checkout v0_4_lang_conditioning
```

## Backwards compatibility

- Old 3-col manifests still load (lang_ids defaults to all-zeros = mr).
- Old checkpoints without `lang_embedding` load fine (`strict=False`); the new param stays at random init.
- Calling `model.bert(input_ids, attention_mask=...)` (no `lang_ids`) still works — it falls through CustomAlbert's else branch and the lang_embedding contribution is bypassed entirely. This is the v0.1/0.2/0.3 baseline path.

## What was deliberately NOT changed

- `train_finetune.py` / `train_finetune_accelerate.py` — these are alternate Stage 2 entry points, unused for v0.4 (we launch via `train_second.py`). Skipping their unpacks is internally consistent for our launch flow but means **don't run those scripts on this branch** without first patching them similarly (one-line each).
- `slmadv` (`Modules/slmadv.py`) — its `model.bert(ref_text, ...)` call is left unchanged. Ref-text OOD passages are treated as language-neutral (no lang_emb contribution). Defensible since OOD_texts.txt is not lang-tagged and the SLM critic operates downstream of the prosody path we're conditioning.
- `models.py` — `build_model` doesn't need changes; `bert` is passed in as the `CustomAlbert` instance and stored as `model.bert`.

## Smoke test (CPU, no real data)

```python
import yaml, torch
from transformers import AlbertConfig
from Utils.PLBERT.util import CustomAlbert

cfg = yaml.safe_load(open('Utils/PLBERT/config.yml'))['model_params']
m = CustomAlbert(AlbertConfig(**cfg))
B, T = 2, 8
input_ids = torch.randint(0, cfg['vocab_size'], (B, T))
lang_ids  = torch.tensor([[0,0,0,1,1,1,0,0],[0,1,1,1,1,1,1,0]])
attn      = torch.ones(B, T, dtype=torch.int32)

# Verify lang_ids changes output
out_a = m(input_ids=input_ids, lang_ids=torch.zeros_like(lang_ids), attention_mask=attn)
out_b = m(input_ids=input_ids, lang_ids=torch.ones_like(lang_ids),  attention_mask=attn)
print('mr vs en mean abs diff:', (out_a - out_b).abs().mean().item())

# Verify gradient flow into lang_embedding
m.train()
out = m(input_ids=input_ids, lang_ids=lang_ids, attention_mask=attn)
out.sum().backward()
print('lang_emb grad nonzero:', (m.lang_embedding.weight.grad.abs() > 0).any().item())
```

Expected output (validated 2026-04-30):

```
mr vs en mean abs diff: 0.754094
lang_emb grad nonzero: True
```
