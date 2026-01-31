import subprocess
import os
import re
import pandas as pd
import time

# --- Configuration ---
ADB_REMOTE_DIR = "/data/local/tmp/quant_study"
LOCAL_MODEL_DIR = "."  # Models are in root/src run dir basically
BENCHMARK_BIN = "./benchmark_model"  # Local path to downloaded binary
RESULTS_CSV = "android_benchmark_results.csv"

# Models to benchmark (filenames expected to exist locally)
MODELS = {
    "C0": "mobilenet_v2_c0.tflite",
    "C1": "mobilenet_v2_c1.tflite",
    "C2": "mobilenet_v2_c2.tflite",
    "C3-C": "mobilenet_v2_c3_c.tflite",
    "C4": "mobilenet_v2_c4.tflite",
    "C5": "mobilenet_v2_c5.tflite",
}

# Runtime Configurations
CONFIGS = [
    {
        "name": "CPU-1T",
        "params": ["--num_threads=1", "--use_gpu=false", "--use_nnapi=false"],
    },
    {
        "name": "CPU-4T",
        "params": ["--num_threads=4", "--use_gpu=false", "--use_nnapi=false"],
    },
    {
        "name": "GPU",
        "params": ["--use_gpu=true"],
    },  # GPU delegate often ignores thread count
]


def run_adb_command(cmd_list):
    """Runs an ADB command and returns (stdout, stderr)."""
    try:
        # print(f"Running: {' '.join(cmd_list)}")
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running ADB command: {e}")
        print(f"Stderr: {e.stderr}")
        return None


def setup_device():
    """Pushes models and benchmark binary to device."""
    print("[Setup] Preparing device...")
    run_adb_command(["adb", "shell", f"mkdir -p {ADB_REMOTE_DIR}"])

    # Push Benchmark Binary
    print(f"[Setup] Pushing {BENCHMARK_BIN}...")
    run_adb_command(["adb", "push", BENCHMARK_BIN, f"{ADB_REMOTE_DIR}/benchmark_model"])
    run_adb_command(["adb", "shell", f"chmod +x {ADB_REMOTE_DIR}/benchmark_model"])

    # Push Models
    for model_id, filename in MODELS.items():
        if os.path.exists(filename):
            print(f"[Setup] Pushing {filename}...")
            run_adb_command(["adb", "push", filename, f"{ADB_REMOTE_DIR}/{filename}"])
        else:
            print(f"[Warning] Model file {filename} not found locally! Skipping push.")


def parse_latency(output):
    """Parses average latency from benchmark_model output."""
    # Example output: "count=50 first=19205 curr=15975 min=15822 max=19205 avg=16345.2 std=597"
    # We want 'avg=16345.2' -> 16.3452 ms (Output is in microseconds usually)
    # Wait, benchmark_model output depends on version.
    # Standard output: "Inference timings in us: Init: 123, First inference: 456, Warmup (avg): 789, Inference (avg): 12345"

    # Let's try regex for "Inference (avg): <number>"
    try:
        match = re.search(r"Inference \(avg\):\s+([\d\.]+)", output)
        if match:
            latency_us = float(match.group(1))
            return latency_us / 1000.0  # Convert to ms
        else:
            # Fallback for older/different versions often showing kv pairs
            match = re.search(r"avg=([\d\.]+)", output)
            if match:
                latency_us = float(match.group(1))
                return latency_us / 1000.0
    except Exception as e:
        print(f"Parsing error: {e}")
    return -1.0


def run_benchmarks():
    results = []

    print("\n[Benchmark] Starting Grid Search...")
    for model_id, filename in MODELS.items():
        remote_path = f"{ADB_REMOTE_DIR}/{filename}"

        for config in CONFIGS:
            config_name = config["name"]
            print(f"   > Running {model_id} on {config_name}...")

            cmd = [
                "adb",
                "shell",
                f"{ADB_REMOTE_DIR}/benchmark_model",
                f"--graph={remote_path}",
                "--num_runs=50",
                "--warmup_runs=10",
            ] + config["params"]

            output = run_adb_command(cmd)

            if output:
                latency = parse_latency(output)
                print(f"     Result: {latency:.4f} ms")

                results.append(
                    {
                        "Model": model_id,
                        "Filename": filename,
                        "Backend": config_name,
                        "Latency (ms)": latency,
                    }
                )
            else:
                print("     Result: Failed")
                results.append(
                    {
                        "Model": model_id,
                        "Filename": filename,
                        "Backend": config_name,
                        "Latency (ms)": None,
                    }
                )

    return results


def main():
    if not os.path.exists(BENCHMARK_BIN):
        print(f"Error: {BENCHMARK_BIN} not found. Please download it first.")
        return

    setup_device()
    data = run_benchmarks()

    # Save CSV
    df = pd.DataFrame(data)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[Done] Results saved to {RESULTS_CSV}")
    print(df)


if __name__ == "__main__":
    main()
