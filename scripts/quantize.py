"""INT8 dynamic quantization of the FP32 ONNX policy.

    python -m scripts.quantize [--fp32 models/policy_fp32.onnx] [--out models/policy_int8.onnx]

Dynamic quantization stores the weights as INT8 ahead of time and quantizes the activations on
the fly at every call, so it needs no calibration data (unlike static quantization).
"""

import argparse
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def quantize(fp32_path, int8_path):
    with tempfile.TemporaryDirectory() as tmp:
        prepped = Path(tmp) / "prepped.onnx"
        # Pre-processing (shape inference + graph cleanup). Without it ORT warns about shape info,
        # and quantization then fails on our export: in dynamic mode ORT rewrites each Gemm as
        # MatMul + Add, transposing the weight in place, which clashes with the weight shapes the
        # exporter recorded in the graph's value_info. Pre-processing rebuilds that shape info.
        quant_pre_process(str(fp32_path), str(prepped))
        # QInt8 = signed 8-bit weights, one scale per tensor. Activations get uint8 scales computed
        # per call (DynamicQuantizeLinear), then MatMulInteger does the integer matrix product.
        quantize_dynamic(str(prepped), str(int8_path), weight_type=QuantType.QInt8)


def main():
    parser = argparse.ArgumentParser(description="Quantize the FP32 ONNX policy to INT8 (dynamic).")
    parser.add_argument("--fp32", default="models/policy_fp32.onnx")
    parser.add_argument("--out", default="models/policy_int8.onnx")
    args = parser.parse_args()

    fp32, int8 = Path(args.fp32), Path(args.out)
    quantize(fp32, int8)
    onnx.checker.check_model(onnx.load(str(int8)))
    for path in (fp32, int8):
        ops = sorted({n.op_type for n in onnx.load(str(path)).graph.node})
        print(f"{path}: {path.stat().st_size / 1024:.1f} KB | ops {ops}")
    print(f"size ratio INT8 / FP32: {int8.stat().st_size / fp32.stat().st_size:.2f}")


if __name__ == "__main__":
    main()
