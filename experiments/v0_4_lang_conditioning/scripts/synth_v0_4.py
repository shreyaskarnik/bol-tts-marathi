"""
End-to-end inference for v0.4 ckpts (StyleTTS2 fork format).

Phonemization uses espeak-ng directly (via subprocess) so the (en)/(mr)
language-switch tags survive — same approach as webgpu-demo's phonemizer.ts.
Per-token lang_ids are derived directly from those tags, not from a
word-level Latin/Devanagari heuristic. This matches the language signal
espeak emits during inference exactly.

Doesn't depend on Task #14 (kokoro inference fork patches) — uses StyleTTS2's
models directly so we can pass lang_ids without going through the kokoro lib.

Usage:
    python synth_v0_4.py \\
        --ckpt logs/kokoro-marathi-v0_4-smoke/epoch_2nd_00000.pth \\
        --voicepack /path/to/mf_mukta.bin \\
        --config configs/config_marathi_v0_4_smoke.yml \\
        --text "मी Google मध्ये काम करतो" \\
        --output out.wav
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Resolve StyleTTS2 source on import path
HERE = Path(__file__).resolve().parent
STYLETTS2_FORK = HERE.parents[3] / "kokoro-deutsch" / "StyleTTS2"
sys.path.insert(0, str(STYLETTS2_FORK))

import numpy as np
import soundfile as sf
import torch
import yaml
from munch import Munch

# StyleTTS2's models.py + Utils/* call torch.load without explicit weights_only,
# and PyTorch 2.6+ defaults to True (rejects unallowlisted module-level objects).
# The auxiliary ckpts (ASR, JDC F0) are trusted local files. Override default
# globally so we do not have to edit every load site upstream.
_torch_load_orig = torch.load
torch.load = lambda *args, **kwargs: _torch_load_orig(*args, **{"weights_only": False, **kwargs})

from kokoro_symbols import TextCleaner  # noqa: E402
from models import build_model, load_F0_models, load_ASR_models  # noqa: E402
from Utils.PLBERT.util import load_plbert  # noqa: E402


# espeak-ng wraps switched runs in (xx) tags — same regex used by webgpu-demo's
# phonemizer.ts (LANG_TAG_SPLIT). Captures the tag so we can route per-segment.
LANG_TAG_RE = re.compile(r"\(([a-z]{2,3}(?:-[a-z0-9]+)?)\)")


def espeak_phonemize_tagged(text: str, voice: str = "mr") -> str:
    """Run espeak-ng directly (preserving (xx) tags) and return raw IPA.

    The misaki Python wrapper strips these tags; we want them, hence subprocess.
    """
    proc = subprocess.run(
        ["espeak-ng", "-v", voice, "--ipa", "-q", text],
        capture_output=True,
        text=True,
        check=True,
    )
    # espeak emits leading whitespace + newlines; collapse to single line
    return " ".join(proc.stdout.split())


def parse_tagged_ipa(raw: str, default_lang: str = "mr") -> tuple[str, list[int]]:
    """Strip (xx) tags, return (clean_ipa, per_char_lang).

    per_char_lang: 0 = mr, 1 = en (we only distinguish the two — other langs map to mr).
    """
    parts = LANG_TAG_RE.split(raw)  # ['text0', 'lang0', 'text1', 'lang1', 'text2', ...]
    clean_chars: list[str] = []
    char_lang: list[int] = []
    cur_lang = default_lang
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text segment under the current language
            for ch in part:
                clean_chars.append(ch)
                char_lang.append(1 if cur_lang.startswith("en") else 0)
        else:
            # Tag — sets lang for the next text segment
            cur_lang = part
    return "".join(clean_chars), char_lang


def tokenize_with_lang_ids(
    ipa: str,
    char_lang: list[int],
    cleaner: TextCleaner,
) -> tuple[list[int], list[int]]:
    """Char-by-char tokenize with TextCleaner's skip-unknown semantics, keeping
    parallel lang_ids. Adds boundary 0 tokens at both ends (mirrors dataloader).
    """
    cleaner_dict = cleaner.word_index_dictionary
    inner_ids: list[int] = []
    inner_langs: list[int] = []
    for i, ch in enumerate(ipa):
        if ch in cleaner_dict:
            inner_ids.append(cleaner_dict[ch])
            inner_langs.append(char_lang[i])
    return [0] + inner_ids + [0], [0] + inner_langs + [0]


def build_and_load(config_path: Path, ckpt_path: Path, device: str) -> Munch:
    """Build StyleTTS2 model from config + load v0.4 ckpt state. Returns model
    in inference mode (Module.train(False))."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # build_model uses dotted access (args.decoder.type, etc.), needs recursive Munch
    def _munchify(d):
        if isinstance(d, dict):
            return Munch({k: _munchify(v) for k, v in d.items()})
        if isinstance(d, list):
            return [_munchify(x) for x in d]
        return d
    model_params = _munchify(config["model_params"])

    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params = state["net"]
    n_loaded_total = 0
    for key in model:
        if key not in params:
            continue
        # Always strip 'module.' prefix (saved with DataParallel). load_state_dict
        # with strict=False silently skips non-matching keys, so we MUST match
        # exactly — strip the prefix before load. (Earlier try/except logic
        # silently loaded zero matching keys → model ran with random weights.)
        sd = params[key]
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):] if k.startswith("module.") else k: v
                  for k, v in sd.items()}
        result = model[key].load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(result.missing_keys)
        n_loaded_total += n_loaded
        if result.missing_keys or result.unexpected_keys:
            # Log mismatch but proceed (lang_embedding is expected missing on
            # pre-v0.4 ckpts, and DataParallel buffers may be unexpected)
            missing_summary = result.missing_keys[:3] if result.missing_keys else []
            unexpected_summary = result.unexpected_keys[:3] if result.unexpected_keys else []
            print(f"  [{key}] loaded {n_loaded}/{len(sd)} params; "
                  f"missing={len(result.missing_keys)} (e.g. {missing_summary}); "
                  f"unexpected={len(result.unexpected_keys)} (e.g. {unexpected_summary})")
    print(f"[model] total params loaded: {n_loaded_total}")

    for key in model:
        model[key] = model[key].to(device).train(False)
    return model


@torch.no_grad()
def synthesize(
    model: Munch,
    input_ids: list[int],
    lang_ids: list[int] | None,
    ref_s: torch.Tensor,
    speed: float,
    device: str,
) -> torch.Tensor:
    """Forward through PLBERT(+lang_ids) → predictor → decoder. Returns audio tensor.

    lang_ids=None bypasses the lang_embedding entirely (PLBERT runs as if there
    were no language conditioning). Use for pre-v0.4 ckpts where the lang_embed
    is at random init and could otherwise perturb output.
    """
    input_ids_t = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    lang_ids_t = torch.LongTensor(lang_ids).unsqueeze(0).to(device) if lang_ids is not None else None
    input_lengths = torch.LongTensor([len(input_ids)]).to(device)

    text_mask = torch.zeros(1, len(input_ids), dtype=torch.bool, device=device)

    bert_dur = model.bert(
        input_ids_t,
        lang_ids=lang_ids_t,
        attention_mask=(~text_mask).int(),
    )
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    s_acoustic = ref_s[:, :128]
    s_prosodic = ref_s[:, 128:]

    d = model.predictor.text_encoder(d_en, s_prosodic, input_lengths, text_mask)
    # v0.5: FiLM language conditioning. apply_lang_film is identity when lang_ids
    # is None (so pre-v0.5 ckpts behave unchanged) AND when γ=1, β=0 (so a
    # v0.5 ckpt with un-trained FiLM also behaves unchanged).
    if hasattr(model.predictor, "apply_lang_film"):
        d = model.predictor.apply_lang_film(d, lang_ids_t)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

    indices = torch.repeat_interleave(
        torch.arange(input_ids_t.shape[1], device=device), pred_dur
    )
    pred_aln_trg = torch.zeros(
        (input_ids_t.shape[1], indices.shape[0]), device=device
    )
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)

    en = d.transpose(-1, -2) @ pred_aln_trg
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s_prosodic)

    t_en = model.text_encoder(input_ids_t, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg

    audio = model.decoder(asr, F0_pred, N_pred, s_acoustic).squeeze()
    return audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="v0.4 StyleTTS2 ckpt (.pth)")
    ap.add_argument("--voicepack", type=Path, required=True,
                    help="ref_s style vector — .bin float32 [N,1,256] per-position, or .pt")
    ap.add_argument("--config", type=Path, required=True,
                    help="v0.4 config yaml (matches the ckpt's training config)")
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--text-file", type=Path, default=None,
                    help="newline-separated lines (overrides --text if both given)")
    ap.add_argument("--output", type=Path, default=Path("synth.wav"))
    ap.add_argument("--speed", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-lang-ids", action="store_true",
                    help="bypass lang_embedding (pass lang_ids=None to PLBERT). "
                         "Use to A/B test whether lang_embedding random init is "
                         "perturbing pre-v0.4 ckpts.")
    args = ap.parse_args()

    if args.text_file:
        lines = [l.strip() for l in args.text_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.text:
        lines = [args.text]
    else:
        # Default test set — pure mr + minglish (mirrors webgpu-demo test_phrases)
        lines = [
            "नमस्कार मी मराठी बोलतो.",
            "मी Google मध्ये काम करतो.",
            "Coffee पिऊया का?",
            "Weekend ला movie बघायचा plan आहे का?",
        ]

    cleaner = TextCleaner()
    print(f"[model] loading {args.ckpt} on {args.device}...")
    model = build_and_load(args.config, args.ckpt, args.device)

    if args.voicepack.suffix == ".bin":
        raw = np.fromfile(args.voicepack, dtype=np.float32)
        ref_s_full = torch.from_numpy(raw).reshape(-1, 1, 256).to(args.device)
    else:
        ref_s_full = torch.load(args.voicepack, map_location=args.device, weights_only=False)
        if ref_s_full.dim() == 2:
            ref_s_full = ref_s_full.unsqueeze(1)

    print(f"[model] loaded; voicepack shape {tuple(ref_s_full.shape)}")
    print(f"[synth] output → {args.output}")

    sample_rate = 24000
    audio_chunks: list[np.ndarray] = []
    for i, line in enumerate(lines):
        raw_ipa = espeak_phonemize_tagged(line, voice="mr")
        ipa, char_lang = parse_tagged_ipa(raw_ipa)
        input_ids, lang_ids = tokenize_with_lang_ids(ipa, char_lang, cleaner)
        n_en = sum(lang_ids)
        print(f"  [{i+1}/{len(lines)}] {line!r}")
        print(f"    raw_ipa: {raw_ipa[:80]}{'…' if len(raw_ipa) > 80 else ''}")
        print(f"    tokens: {len(input_ids)}  en_tokens: {n_en} ({100*n_en/len(input_ids):.1f}%)")

        # Pick voicepack slice indexed by token count (matches Kokoro convention)
        n = len(input_ids)
        if ref_s_full.shape[0] >= n:
            ref_s = ref_s_full[n - 1]
        else:
            ref_s = ref_s_full.mean(dim=0)
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        eff_lang_ids = None if args.no_lang_ids else lang_ids
        audio = synthesize(model, input_ids, eff_lang_ids, ref_s, args.speed, args.device)
        audio_chunks.append(audio.cpu().numpy())
        # 80ms trailing pad — same as webgpu-demo (avoids ISTFTNet edge truncation)
        audio_chunks.append(np.zeros(int(sample_rate * 0.08), dtype=np.float32))

    full_audio = np.concatenate(audio_chunks)
    sf.write(args.output, full_audio, sample_rate, subtype="PCM_16")
    print(f"[done] wrote {args.output} ({len(full_audio)/sample_rate:.2f}s @ {sample_rate}Hz)")


if __name__ == "__main__":
    main()
