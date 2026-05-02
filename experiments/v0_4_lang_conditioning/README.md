# v0.4 Language Conditioning Experiment

Per-token language conditioning at the PLBERT/predictor pathway. Goal: fix
"intelligible-but-Marathi-paced" English output (the v0.1.1 finding from
[../docs/RESEARCH_LANGUAGE_CONDITIONING.md §1.1](../../docs/RESEARCH_LANGUAGE_CONDITIONING.md#11-empirical-evidence-v01-inference-test)).

> **Full design doc:** [docs/RESEARCH_LANGUAGE_CONDITIONING.md](../../docs/RESEARCH_LANGUAGE_CONDITIONING.md).
> Read it first; this README is just the experiment-local map.

## Layout

```
v0_4_lang_conditioning/
├── README.md              # this file
├── configs/               # config_marathi_v0_4_langcond.yml lives here
├── scripts/
│   ├── build_loanword_dict.py     # IndicCMix word-aligned Devanagari→Latin dict
│   ├── build_v0_4_manifest.py     # re-phonemize Rasa/IV-R/SPRINGLab with romanized loanwords + lang_ids
│   ├── flip_rate_check.py         # sanity check: % phoneme tokens flipped mr→en
│   └── train_v0_4.sh              # launch wrapper
├── styletts2_patches/     # patches against semidark/StyleTTS2 fork (not direct edits)
│   ├── 01_lang_embedding_plbert.patch
│   ├── 02_dataloader_lang_ids.patch
│   └── 03_train_*_lang_ids.patch
├── kokoro_patches/        # patches against kokoro-deutsch/kokoro fork
│   └── 01_forward_with_tokens_lang_ids.patch
├── data/                  # generated artifacts (loanword_dict_dev_to_latin.tsv, manifests)
│   └── .gitignore         # large generated files not committed
└── notes/                 # experiment logs, ablation results
    └── pilot_runs.md
```

## Why an isolated experiment dir?

v0.2 / v0.3 / v0.4 are independent training axes (data recipe, frontend
lexicon, model architecture). They should ship orthogonally. If v0.4
underperforms, we revert by ignoring this directory; the rest of the repo is
untouched. If it works, we promote selected pieces back into top-level
`scripts/` and `configs/` under a clear v0.4 release.

The fork patches live as patch files (not direct submodule edits) so the
parent submodule pins don't drift.

## Status

| Task | Status | Artifact |
|---|---|---|
| #9  Scaffolding | ✓ done | this directory |
| #11 v0.4 manifest re-phonemizer (V1 stub) | ✓ done | `scripts/build_smoke_manifest.py`, `data/{train,val}_list_smoke.txt` |
| #13 StyleTTS2 patches | ✓ done | `styletts2_patches/01_lang_conditioning.patch` (also on `kokoro-deutsch/StyleTTS2` branch `v0_4_lang_conditioning`) |
| #15 v0.4 config | ✓ done | `configs/config_marathi_v0_4_langcond.yml` (full run), `configs/config_marathi_v0_4_smoke.yml` (1-epoch smoke) |
| **#16 Smoke test on A100 (~$1.51)** | **next — see [`notes/SMOKE_TEST_RUNBOOK.md`](notes/SMOKE_TEST_RUNBOOK.md)** | TBD |
| #10 IndicCMix loanword dict | blocked on #16 | TBD |
| #12 Flip-rate sanity check | blocked on #10 | TBD |
| #11 v2 (full re-phonemizer) | follow-up to #10 | TBD |
| #14 kokoro inference patches | blocked on #16 | TBD |
| #17 Full Stage 2.5 (~$38) | blocked on #10/#11/#12/#16 | TBD |
| #18 ONNX + demo + deploy | blocked on #14/#17 | TBD |
