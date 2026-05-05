"""Export a Kokoro-format fine-tune (.pth) to ONNX for WebGPU / transformers.js deployment.

Why `disable_complex=True`: upstream Kokoro uses `TorchSTFT` which relies on
`torch.stft(return_complex=True)`. ONNX doesn't support complex tensors. Setting
`disable_complex=True` on `KModel` swaps in `CustomSTFT` that works in pure real
arithmetic and exports cleanly.

Usage:

    python scripts/export_onnx.py \
        --model     checkpoints/kokoro_mr_final.pth \
        --config    configs/config_mr.json \
        --output    checkpoints/kokoro-mr-v1_0.onnx \
        --opset     17 \
        --dummy-phonemes 40

The exported ONNX takes THREE inputs:
    input_ids: int64   [1, n_phonemes]  — phoneme token IDs (per configs/config_mr.json vocab)
    ref_s:     float32 [1, 256]         — per-phoneme-position style vector (slice of voicepack)
    speed:     float32 [1]              — pacing multiplier (1.0 = neutral; <1.0 slows, >1.0 fastens).
                                          Divides the predictor's per-phoneme duration BEFORE rounding,
                                          so it scales actual frame allocation — not just playback rate.

And produces:
    audio:     float32 [1, n_samples] — 24 kHz waveform
    pred_dur:  int64   [1, n_phonemes] — per-phoneme durations in predictor frames
                                         (1 frame = 600 audio samples at 24 kHz)

pred_dur is exposed so downstream apps can build word/phoneme timestamps.
See scripts/with_timestamps.py for the timestamp-extraction recipe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_submodule_to_path() -> None:
    """Prefer the kokoro/ submodule over a pip-installed kokoro package."""
    here = Path(__file__).resolve().parent
    sub = here.parent / "kokoro" / "kokoro"
    if sub.exists() and str(sub.parent) not in sys.path:
        sys.path.insert(0, str(sub.parent))


_add_submodule_to_path()

import torch  # noqa: E402

from kokoro import KModel  # noqa: E402


class _KokoroONNXWrapper(torch.nn.Module):
    """Thin wrapper over KModel for ONNX tracing with speed as a dynamic input.

    KModel.forward_with_tokens uses speed as `duration = ... / speed` — a plain
    division, traceable by the legacy TorchScript tracer if speed is passed as a
    0-d / 1-d float tensor instead of a Python scalar. We unbox speed[0] inside
    the wrapper so callers pass it as a [1] tensor (matches transformers.js
    convention for scalar-shaped inputs).
    """

    def __init__(self, kmodel: KModel) -> None:
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        # speed is float32[1]; pass the scalar tensor through — division by a
        # tensor traces cleanly in forward_with_tokens.
        audio, pred_dur = self.kmodel.forward_with_tokens(
            input_ids=input_ids, ref_s=ref_s, speed=speed[0]
        )
        return audio, pred_dur


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to Kokoro-format .pth")
    ap.add_argument("--config", required=True, help="Path to config_mr.json")
    ap.add_argument("--output", required=True, help="Output .onnx path")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version (default 17)")
    ap.add_argument(
        "--dummy-phonemes",
        type=int,
        default=40,
        help="Number of phonemes in the dummy trace input (any dynamic axis is exported anyway)",
    )
    ap.add_argument("--repo-id", default="hexgrad/Kokoro-82M", help="Kokoro HF repo_id (metadata only)")
    ap.add_argument("--verify", action="store_true", help="After export, load back via onnxruntime and compare outputs")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading KModel with disable_complex=True (so TorchSTFT → CustomSTFT)")
    kmodel = KModel(
        repo_id=args.repo_id,
        config=args.config,
        model=args.model,
        disable_complex=True,  # mandatory for ONNX export
    )
    # Recursively set ALL submodules (incl. InstanceNorm/BatchNorm hidden under
    # spectral_norm wrappers) to eval mode. A naive .train(False) on the top
    # module doesn't reach every nested submodule when spectral_norm is in play
    # — leads to "instance_norm set to train=True" warning + ONNX runtime
    # produces static (batch stats instead of running stats) on a Marathi voice.
    # The for-loop forces propagation. (Bug we hit on May 4 2026 — re-exported
    # v0.2 ONNX produced static; root cause was naive .train(False).)
    for m in kmodel.modules():
        m.train(False)

    wrap = _KokoroONNXWrapper(kmodel)
    for m in wrap.modules():
        m.train(False)

    # Dummy inputs for tracing. input_ids is [batch=1, n_phonemes]; batch is
    # fixed at 1, phoneme dim is dynamic via dynamic_axes. speed is float32[1] —
    # callers pass [1.0] for neutral pacing, [0.75] to slow down, etc.
    n_phones = args.dummy_phonemes
    dummy_input_ids = torch.zeros(1, n_phones, dtype=torch.long)
    dummy_ref_s = torch.zeros(1, 256, dtype=torch.float32)
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)

    print(f"exporting to {out}")
    torch.onnx.export(
        wrap,
        (dummy_input_ids, dummy_ref_s, dummy_speed),
        str(out),
        input_names=["input_ids", "ref_s", "speed"],
        output_names=["audio", "pred_dur"],
        dynamic_axes={
            "input_ids": {1: "n_phonemes"},
            "audio":     {1: "n_samples"},
            "pred_dur":  {1: "n_phonemes"},
        },
        opset_version=args.opset,
        dynamo=False,  # legacy TorchScript tracer; works cleanly on Kokoro's LSTM/InstanceNorm
    )
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  exported {out} ({size_mb:.1f} MB)")

    if args.verify:
        print("verifying with onnxruntime…")
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        # small-input sanity check — input_ids is [1, n_phonemes], ref_s is [1, 256], speed is [1]
        test_ids = torch.randint(0, 100, (1, n_phones), dtype=torch.long)
        test_ref = torch.randn(1, 256)
        test_speed = torch.tensor([1.0], dtype=torch.float32)
        ort_out = sess.run(
            None,
            {
                "input_ids": test_ids.numpy(),
                "ref_s": test_ref.numpy().astype("float32"),
                "speed": test_speed.numpy(),
            },
        )
        pt_audio, pt_dur = wrap(test_ids, test_ref, test_speed)
        diff_audio = (pt_audio.detach().numpy() - ort_out[0]).__abs__().max()
        diff_dur = (pt_dur.detach().numpy() - ort_out[1]).__abs__().max()
        print(f"  max|pt_audio - ort_audio|: {diff_audio:.2e}")
        print(f"  max|pt_dur   - ort_dur|:   {diff_dur}")
        ok = diff_audio < 1e-3 and diff_dur == 0
        print("  " + ("OK" if ok else "MISMATCH — investigate"))

    print()
    print("Next steps:")
    print(f"  1. (optional) quantize for WebGPU — e.g. `onnxruntime` or `optimum` to int8/q4")
    print(f"  2. push to Hub as `<user>/bol-tts-marathi-onnx/onnx/model.onnx`")
    print(f"  3. wire up transformers.js with the Marathi lang_code patch in scripts/onnx_client.js (TODO)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
