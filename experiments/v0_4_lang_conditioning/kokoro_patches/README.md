# Kokoro inference fork patches for v0.4 language conditioning

Single combined patch (`01_lang_ids_forward.patch`) covering:

| File | Change | LOC |
|---|---|---|
| `kokoro/modules.py` | `CustomAlbert` now has `lang_embedding` table; forward accepts optional `lang_ids` (mirrors the StyleTTS2 fork patch so ckpts are loadable here) | ~25 |
| `kokoro/model.py` | `forward_with_tokens`, `forward`, and `KModelForONNX.forward` all accept optional `lang_ids` parameter, threaded into `self.bert(...)` | ~25 |

Total: ~50 LOC, against the design-doc estimate of ~35.

## How to apply

```bash
cd /path/to/kokoro
git checkout main
git apply /path/to/bol-tts-marathi/experiments/v0_4_lang_conditioning/kokoro_patches/01_lang_ids_forward.patch
```

Or in this local development setup the patch is already applied on the
`v0_4_lang_conditioning` branch of `kokoro-deutsch/kokoro`:

```bash
cd kokoro-deutsch/kokoro
git checkout v0_4_lang_conditioning
```

## Backwards compatibility

- Calling `kmodel(phonemes, ref_s)` without lang_ids → falls through CustomAlbert's else branch → no lang_emb contribution → identical output to v0.1/0.2/0.3 baseline path.
- Old Kokoro ckpts (no `lang_embedding` param) load fine via `strict=False` (the existing kokoro load path uses non-strict; lang_embedding stays at random init when missing).

## ONNX export implications

`KModelForONNX.forward` now accepts an optional `lang_ids` int64 tensor. To export with lang conditioning:

```python
torch.onnx.export(
    KModelForONNX(kmodel),
    args=(input_ids, ref_s, torch.tensor([1.0]), lang_ids),
    f='kokoro-mr-v0_4.onnx',
    input_names=['input_ids', 'ref_s', 'speed', 'lang_ids'],
    output_names=['waveform', 'duration'],
    dynamic_axes={
        'input_ids': {0: 'B', 1: 'T'},
        'lang_ids':  {0: 'B', 1: 'T'},
        ...
    },
    opset_version=17,
)
```

For backward-compat ONNX exports (no lang conditioning), pass `lang_ids=None` — but note that `torch.onnx.export` typically requires concrete tensors for all args. Easier to always pass a zero-tensor for lang_ids and let CustomAlbert's "all-zeros adds lang_embedding[0]" path apply.

## Inference usage from Python

```python
from kokoro import KModel
import torch

kmodel = KModel(repo_id='shreyask/bol-tts-marathi-v0_4', model='kokoro-mr-v0_4.pth')

# Build lang_ids parallel to the phoneme string (1 per char, 0/1 ints)
phonemes = "..."  # IPA string
lang_ids_inner = [...]  # length matches phonemes (no boundary 0s yet)
lang_ids = torch.LongTensor([[0, *lang_ids_inner, 0]])

audio = kmodel(phonemes, ref_s, lang_ids=lang_ids)
```

For a self-contained inference scaffold without the kokoro lib (uses StyleTTS2 directly), see `experiments/v0_4_lang_conditioning/scripts/synth_v0_4.py`.
