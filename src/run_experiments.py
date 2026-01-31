import os
from typing import Any
import torch
import pandas as pd
import numpy as np
import tensorflow as tf
import litert_torch
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.qtyping import QuantGranularity, TFLOperationName
from src.model import get_model
from src.utils import get_calibration_dataset, compute_metrics
import shutil
import time

# Configuration
RESULTS_CSV = "experiment_results.csv"
MODEL_FLOAT = "mobilenet_v2_float.tflite"

EXPERIMENTAL_CASES = [
    {"id": "C0", "name": "Baseline", "type": "fp32", "data": None, "samples": 0},
    {
        "id": "C1",
        "name": "Weight-Only",
        "type": "weight_only",
        "data": None,
        "samples": 0,
    },
    {"id": "C2", "name": "Dynamic", "type": "dynamic", "data": None, "samples": 0},
    {
        "id": "C3-A",
        "name": "Static-Random-100",
        "type": "static",
        "data": "random",
        "samples": 100,
    },
    {
        "id": "C3-B",
        "name": "Static-Real-10",
        "type": "static",
        "data": "random",
        "samples": 10,
    },
    {
        "id": "C3-C",
        "name": "Static-Real-100",
        "type": "static",
        "data": "random",
        "samples": 100,
    },
    {
        "id": "C3-D",
        "name": "Static-Real-500",
        "type": "static",
        "data": "random",
        "samples": 500,
    },
    {
        "id": "C4",
        "name": "Full-Integer",
        "type": "full_int",
        "data": "random",
        "samples": 100,
    },
    {
        "id": "C5",
        "name": "Per-Tensor",
        "type": "per_tensor",
        "data": "random",
        "samples": 100,
    },
]


def run_experiments():
    results = []

    # 0. Prepare FP32 Model (Common Source)
    print("[Setup] Loading PyTorch Model and converting to Float TFLite...")
    pt_model = get_model(pretrained=True)
    sample_input = (torch.randn(1, 3, 224, 224),)

    tflite_model = litert_torch.convert(pt_model, sample_input)
    tflite_model.export(MODEL_FLOAT)

    # 1. Warm-up / Prepare Baseline Verification Data (PyTorch Output)
    print("[Setup] Preparing Golden Set (PyTorch Outputs)...")
    golden_inputs = []
    golden_outputs = []

    # Use real data for verification if possible, to measure realistic loss
    # Using 100 samples for verification metrics
    verify_gen = get_calibration_dataset("random", 100)
    for inp_list in verify_gen:
        # inp_list: [numpy_array]
        inp_tensor = torch.from_numpy(inp_list[0])
        with torch.no_grad():
            out = pt_model(inp_tensor).numpy()
        golden_inputs.append(inp_list[0])
        golden_outputs.append(out)

    print(f"[Setup] Goldset ready. {len(golden_inputs)} samples.")

    # 2. Run Automation Loop
    for case in EXPERIMENTAL_CASES:
        cid = case["id"]
        cname = case["name"]
        print(f"\n>>> Running Experiment {cid}: {cname} <<<")

        output_filename = f"mobilenet_v2_{cid.lower().replace('-', '_')}.tflite"

        qt = quantizer.Quantizer(MODEL_FLOAT)

        # Apply Quantization Settings
        if case["type"] == "fp32":
            # Just copy float model
            shutil.copy(MODEL_FLOAT, output_filename)

        elif case["type"] == "weight_only":
            # Weight Only (Int8 Weight, Float Act)
            qt.add_weight_only_config(
                regex=".*",
                operation_name=TFLOperationName.ALL_SUPPORTED,
                num_bits=8,
                granularity=QuantGranularity.CHANNELWISE,
            )
            q_res = qt.quantize()
            q_res.save(
                os.path.dirname(output_filename) or ".",
                os.path.splitext(output_filename)[0],
                overwrite=True,
            )

        elif case["type"] == "dynamic":
            # Dynamic (Int8 Weight, Dynamic Act)
            qt.add_dynamic_config(
                regex=".*",
                operation_name=TFLOperationName.ALL_SUPPORTED,
                num_bits=8,
                granularity=QuantGranularity.CHANNELWISE,
            )
            q_res = qt.quantize()
            q_res.save(
                os.path.dirname(output_filename) or ".",
                os.path.splitext(output_filename)[0],
                overwrite=True,
            )

        elif case["type"] in ["static", "full_int", "per_tensor"]:
            # Static Quantization requires calibration

            # Calibration Data Setup
            calib_samples = case["samples"]
            calib_dtype = case["data"]

            # Helper for input name
            interpreter = tf.lite.Interpreter(model_path=MODEL_FLOAT)
            input_details = interpreter.get_input_details()
            sig_info = interpreter.get_signature_list()

            # Robust input name extraction
            input_arg_name = "args_0"
            sig_key = None
            if sig_info:
                sig_key = list(sig_info.keys())[0]
                sig_inputs = sig_info[sig_key]["inputs"]
                if isinstance(sig_inputs, dict):
                    input_arg_name = list(sig_inputs.keys())[0]

            def calib_gen():
                gen = get_calibration_dataset(calib_dtype, calib_samples)
                for data in gen:
                    yield {input_arg_name: data[0]}

            # Explicit type hint to solve Pyright error
            calib_data: dict[str, Any] = {str(sig_key): calib_gen()}

            # Config
            granularity = (
                QuantGranularity.TENSORWISE
                if case["type"] == "per_tensor"
                else QuantGranularity.CHANNELWISE
            )

            qt.add_static_config(
                regex=".*",
                operation_name=TFLOperationName.ALL_SUPPORTED,
                activation_num_bits=8,
                weight_num_bits=8,
                weight_granularity=granularity,
            )

            print(f"   Calibrating with {calib_samples} {calib_dtype} samples...")
            calib_result = qt.calibrate(calib_data)

            q_res = qt.quantize(calib_result)

            q_res.save(
                os.path.dirname(output_filename) or ".",
                os.path.splitext(output_filename)[0],
                overwrite=True,
            )

        # 3. Verification & Metrics
        print(f"   Verifying {output_filename}...")

        # Load TFLite
        try:
            interpreter = tf.lite.Interpreter(model_path=output_filename)
            interpreter.allocate_tensors()
        except Exception as e:
            print(f"   [Error] Failed to load {output_filename}: {e}")
            continue

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Metrics Accumulators
        mse_list = []
        snr_list = []
        cos_list = []

        for i in range(len(golden_inputs)):
            g_in = golden_inputs[i]  # (1,3,224,224) float
            g_out = golden_outputs[i]

            # Prepare Input
            input_scale, input_zp = input_details[0]["quantization"]
            if input_scale > 0:
                # Quantize input
                in_data = (g_in / input_scale + input_zp).astype(
                    input_details[0]["dtype"]
                )
            else:
                in_data = g_in

            interpreter.set_tensor(input_details[0]["index"], in_data)
            interpreter.invoke()
            out_data = interpreter.get_tensor(output_details[0]["index"])

            # Handle Quantized Output
            out_scale, out_zp = output_details[0]["quantization"]
            if out_scale > 0:
                # Dequantize
                out_data = (out_data.astype(np.float32) - out_zp) * out_scale

            # Compute Metrics
            m = compute_metrics(g_out, out_data)
            mse_list.append(m["MSE"])
            snr_list.append(m["SNR_dB"])
            cos_list.append(m["Cosine_Sim"])

        # Latency Measurement (100 runs)
        latency_list = []
        # Warmup
        for _ in range(10):
            interpreter.invoke()

        start_time = time.time()
        for _ in range(100):
            interpreter.invoke()
        end_time = time.time()
        avg_latency_ms = ((end_time - start_time) / 100) * 1000

        # Aggregated Results
        avg_mse = np.mean(mse_list)
        avg_snr = np.mean(snr_list)
        avg_cos = np.mean(cos_list)
        f_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB

        print(
            f"   Stats -> MSE: {avg_mse:.6f}, SNR: {avg_snr:.2f} dB, Size: {f_size:.2f} MB, Latency: {avg_latency_ms:.2f} ms"
        )

        results.append(
            {
                "Experiment ID": cid,
                "Name": cname,
                "Config": f"{case['type']}-{case['data']}-{case['samples']}",
                "MSE": avg_mse,
                "SNR (dB)": avg_snr,
                "Cosine Sim": avg_cos,
                "Model Size (MB)": f_size,
                "Latency (ms)": avg_latency_ms,
            }
        )

    # 4. Save Results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nAll experiments completed. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    run_experiments()
