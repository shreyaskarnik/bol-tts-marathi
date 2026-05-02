"""
Export a v0.4 StyleTTS2 ckpt to ONNX with lang_ids input.

Two-stage process:
  1. StyleTTS2 → Kokoro inference format (.pth, ~310 MB):
     extract {bert, bert_encoder, predictor, decoder, text_encoder} from the
     1.8 GB StyleTTS2 ckpt, strip 'module.' prefix, save flat dict.
     The `bert` state dict now includes `lang_embedding.weight` from the v0.4
     CustomAlbert.

  2. Kokoro .pth → ONNX:
     load via KModel(disable_complex=True) from the v0_4_lang_conditioning
     branch of the kokoro inference fork (which has KModelForONNX.forward
     accepting lang_ids), wrap, torch.onnx.export with lang_ids as a 3rd input.

Output ONNX inputs:
    input_ids: int64   [1, n_phonemes]
    ref_s:     float32 [1, 256]
    lang_ids:  int64   [1, n_phonemes]   ← v0.4 addition

Outputs:
    audio:    float32 [1, n_samples]
    pred_dur: int64   [1, n_phonemes]

Usage:
    python export_v0_4_onnx.py \\
        --styletts2-ckpt /workspace/.../epoch_2nd_00009.pth \\
        --kokoro-config  configs/config_mr.json \\
        --output         kokoro-mr-v0_4.onnx \\
        --intermediate   kokoro-mr-v0_4.pth        # optional; defaults next to output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# StyleTTS2's models.py + Utils/* call torch.load without explicit weights_only,
# and PyTorch 2.6+ defaults to True (rejects unallowlisted module-level objects).
# Override default globally so we don't have to edit every load site upstream.
import torch
_torch_load_orig = torch.load
torch.load = lambda *args, **kwargs: _torch_load_orig(*args, **{"weights_only": False, **kwargs})


KOKORO_INFERENCE_MODULES = ["bert", "bert_encoder", "predictor", "decoder", "text_encoder"]


def styletts2_to_kokoro(styletts2_ckpt: Path, out_pth: Path) -> None:
    """Extract the 5 inference modules from a StyleTTS2 ckpt + strip 'module.' prefix."""
    print(f"[1/2] Loading StyleTTS2 ckpt {styletts2_ckpt}")
    state = torch.load(styletts2_ckpt, map_location="cpu")
    if "net" not in state:
        raise RuntimeError(f"expected 'net' top-level key, got {list(state.keys())}")
    net = state["net"]
    missing = set(KOKORO_INFERENCE_MODULES) - set(net.keys())
    if missing:
        raise RuntimeError(f"StyleTTS2 ckpt missing modules: {missing}")

    out: dict[str, dict[str, torch.Tensor]] = {}
    for mod in KOKORO_INFERENCE_MODULES:
        sd = net[mod]
        # Strip 'module.' prefix (DataParallel)
        stripped = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        out[mod] = stripped

    # Sanity-check for v0.4 lang_embedding presence
    bert_keys = list(out["bert"].keys())
    has_lang = any("lang_embedding" in k for k in bert_keys)
    if has_lang:
        for k in bert_keys:
            if "lang_embedding" in k:
                shape = tuple(out["bert"][k].shape)
                print(f"     v0.4 lang_embedding: bert.{k} shape {shape}")
    else:
        print("     WARNING: no lang_embedding in bert state — is this a pre-v0.4 ckpt?")

    print(f"     writing Kokoro-format ckpt → {out_pth}")
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pth)
    size_mb = out_pth.stat().st_size / (1024 * 1024)
    print(f"     done ({size_mb:.1f} MB)")


class _V04ONNXWrapper(torch.nn.Module):
    """Thin wrapper for ONNX trace. lang_ids is a true input; speed is baked at 1.0."""

    def __init__(self, kmodel) -> None:
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        lang_ids: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        audio, pred_dur = self.kmodel.forward_with_tokens(
            input_ids=input_ids, ref_s=ref_s, speed=1.0, lang_ids=lang_ids
        )
        return audio, pred_dur


def export_onnx(kokoro_pth: Path, kokoro_config: Path, out_onnx: Path,
                opset: int, n_phones: int, repo_id: str, verify: bool) -> None:
    print(f"\n[2/2] Loading KModel from {kokoro_pth} (disable_complex=True for ONNX-clean STFT)")

    # Add the kokoro fork to sys.path. Local layout: ../../kokoro-deutsch/kokoro
    here = Path(__file__).resolve().parent
    kokoro_src = here.parents[3] / "kokoro-deutsch" / "kokoro"
    if kokoro_src.exists() and str(kokoro_src) not in sys.path:
        sys.path.insert(0, str(kokoro_src))
    # Pod fallback: /workspace/bol_run/kokoro-deutsch/kokoro
    pod_kokoro = Path("/workspace/bol_run/kokoro-deutsch/kokoro")
    if pod_kokoro.exists() and str(pod_kokoro) not in sys.path:
        sys.path.insert(0, str(pod_kokoro))

    from kokoro import KModel

    kmodel = KModel(
        repo_id=repo_id,
        config=str(kokoro_config),
        model=str(kokoro_pth),
        disable_complex=True,
    )
    # Recursively set ALL submodules (incl. InstanceNorm/BatchNorm hidden under
    # spectral_norm wrappers) to eval mode. A naive .train(False) on the top
    # module doesn't reach every nested submodule when spectral_norm is in play
    # — leads to "instance_norm set to train=True" warning + ONNX runtime
    # produces different audio because it uses batch stats instead of running
    # stats. The for-loop forces the propagation.
    for m in kmodel.modules():
        m.train(False)

    wrap = _V04ONNXWrapper(kmodel)
    for m in wrap.modules():
        m.train(False)

    dummy_input_ids = torch.zeros(1, n_phones, dtype=torch.long)
    dummy_ref_s = torch.zeros(1, 256, dtype=torch.float32)
    dummy_lang_ids = torch.zeros(1, n_phones, dtype=torch.long)

    print(f"     exporting → {out_onnx}")
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap,
        (dummy_input_ids, dummy_ref_s, dummy_lang_ids),
        str(out_onnx),
        input_names=["input_ids", "ref_s", "lang_ids"],
        output_names=["audio", "pred_dur"],
        dynamic_axes={
            "input_ids": {1: "n_phonemes"},
            "lang_ids":  {1: "n_phonemes"},
            "audio":     {1: "n_samples"},
            "pred_dur":  {1: "n_phonemes"},
        },
        opset_version=opset,
        dynamo=False,
    )
    size_mb = out_onnx.stat().st_size / (1024 * 1024)
    print(f"     wrote {out_onnx} ({size_mb:.1f} MB)")

    if verify:
        print("\n[3/3] Verifying with onnxruntime")
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
        test_ids = torch.randint(0, 100, (1, n_phones), dtype=torch.long)
        test_ref = torch.randn(1, 256)
        # 50/50 mr/en mix in lang_ids — exercises both rows of lang_embedding
        test_lang = torch.zeros(1, n_phones, dtype=torch.long)
        test_lang[0, n_phones // 2:] = 1
        ort_out = sess.run(None, {
            "input_ids": test_ids.numpy(),
            "ref_s":     test_ref.numpy().astype("float32"),
            "lang_ids":  test_lang.numpy(),
        })
        pt_audio, pt_dur = wrap(test_ids, test_ref, test_lang)
        diff_audio = (pt_audio.detach().numpy() - ort_out[0]).__abs__().max()
        diff_dur = (pt_dur.detach().numpy() - ort_out[1]).__abs__().max()
        print(f"     max|pt_audio - ort_audio|: {diff_audio:.2e}")
        print(f"     max|pt_dur   - ort_dur|:   {diff_dur}")
        ok = diff_audio < 1e-3 and diff_dur == 0
        print("     " + ("OK — ONNX matches PyTorch" if ok else "MISMATCH — investigate"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--styletts2-ckpt", required=True, type=Path,
                    help="StyleTTS2 .pth (e.g. logs/.../epoch_2nd_00009.pth)")
    ap.add_argument("--kokoro-config", required=True, type=Path,
                    help="Kokoro config_mr.json (vocab + arch hyperparams)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output .onnx path")
    ap.add_argument("--intermediate", type=Path, default=None,
                    help="Intermediate Kokoro-format .pth path (default: <output>.pth)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dummy-phonemes", type=int, default=40)
    ap.add_argument("--repo-id", default="hexgrad/Kokoro-82M",
                    help="HF repo_id for KModel metadata only")
    ap.add_argument("--verify", action="store_true",
                    help="After export, run onnxruntime trace + compare to PyTorch")
    args = ap.parse_args()

    intermediate = args.intermediate or args.output.with_suffix(".pth")
    styletts2_to_kokoro(args.styletts2_ckpt, intermediate)
    export_onnx(intermediate, args.kokoro_config, args.output,
                args.opset, args.dummy_phonemes, args.repo_id, args.verify)

    print("\nNext steps:")
    print(f"  1. (optional) quantize to q4f16 for smaller WebGPU bundle")
    print(f"  2. push to HF: shreyask/bol-tts-marathi-onnx (overwrite onnx/model.onnx)")
    print(f"  3. update webgpu-demo to pass lang_ids alongside input_ids in the ONNX session call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
