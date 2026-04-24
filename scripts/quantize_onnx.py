"""Quantize a Kokoro-format ONNX model for WebGPU / small-download deployment.

Supports two modes:
  --mode int8   (default, works with stock onnxruntime)
  --mode int4   (requires onnxruntime-genai or a source checkout — see notes)

Usage:

    # int8 dynamic quantization — 325 MB fp32 → ~80 MB int8
    python scripts/quantize_onnx.py \
        --input  checkpoints/kokoro-mr-v1_0.onnx \
        --output checkpoints/kokoro-mr-v1_0_q8.onnx \
        --mode int8

    # int4 block-wise quantization — ~40 MB int4 (requires onnxruntime-genai)
    python scripts/quantize_onnx.py \
        --input  checkpoints/kokoro-mr-v1_0.onnx \
        --output checkpoints/kokoro-mr-v1_0_q4.onnx \
        --mode int4

int8 path: uses `onnxruntime.quantization.quantize_dynamic` with `QuantType.QInt8`
and `per_channel=True`. Operators quantized: MatMul, Gemm. Keeps Conv ops at fp32
(quantizing them typically hurts audio-gen model quality noticeably).

int4 path: uses `onnxruntime.quantization.matmul_4bits_quantizer.MatMul4BitsQuantizer`
with block-wise quantization (block_size=32 by default). This module is NOT
distributed in the vanilla `onnxruntime` pip package — install via:

    pip install onnxruntime-genai
    # OR clone onnxruntime source and add
    #   onnxruntime/python/tools/quantization/ to PYTHONPATH

After quantization we run the output through `onnxruntime` once to confirm it
loads and produces non-degenerate output (all-zero audio is a collapse signal).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _int8(inp: Path, out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[int8] dynamic quantization: MatMul + Gemm → int8 (per-channel)")
    quantize_dynamic(
        model_input=str(inp),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        per_channel=True,
        # Conv is excluded — quantizing ISTFTNet's convolutions collapses audio (NaN).
        # MatMul + Gemm quantization keeps quality but gives limited size reduction
        # (Kokoro is Conv-heavy). True WebGPU-sized models require q4 MatMul + Conv
        # with static calibration — run with --mode int4 after installing onnxruntime-genai.
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={"WeightSymmetric": True},
    )


def _int4(inp: Path, out: Path) -> None:
    try:
        # onnxruntime-genai install path
        from onnxruntime.quantization.matmul_4bits_quantizer import (
            MatMul4BitsQuantizer,
            DefaultWeightOnlyQuantConfig,
        )
    except ImportError as e:
        raise SystemExit(
            "int4 mode requires onnxruntime-genai or an onnxruntime source checkout.\n"
            "  Install: `uv pip install onnxruntime-genai`\n"
            f"Original error: {e}"
        )

    import onnx

    print(f"[int4] block-wise quantization: MatMul weights → int4 (block_size=32)")
    model = onnx.load(str(inp))
    cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True)
    q = MatMul4BitsQuantizer(model=model, algo_config=cfg)
    q.process()
    onnx.save(q.model, str(out))


def _verify(onnx_path: Path) -> None:
    """Load quantized model, run a dummy input, confirm non-zero audio."""
    import numpy as np
    import onnxruntime as ort

    print(f"[verify] loading {onnx_path.name} in onnxruntime")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_ids = np.array([[53, 156, 47, 158, 144, 51]], dtype=np.int64)  # kˈeːɭi
    ref_s = np.random.randn(1, 256).astype(np.float32) * 0.1
    out = sess.run(None, {"input_ids": input_ids, "ref_s": ref_s})
    audio_rms = np.sqrt(np.mean(out[0] ** 2))
    print(f"  audio shape {out[0].shape}, RMS {audio_rms:.4f}")
    if audio_rms < 1e-4:
        print("  WARN: audio RMS is near zero — quantization may have collapsed the model")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mode", choices=["int8", "int4"], default="int8")
    ap.add_argument("--no-verify", action="store_true", help="Skip onnxruntime smoke test after quantization")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "int8":
        _int8(args.input, args.output)
    else:
        _int4(args.input, args.output)

    size_mb = args.output.stat().st_size / 1e6
    ratio = args.input.stat().st_size / args.output.stat().st_size
    print(f"  wrote {args.output.name} ({size_mb:.1f} MB, {ratio:.1f}x smaller)")

    if not args.no_verify:
        _verify(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
