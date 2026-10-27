"""Batch-1 latency benchmark: PyTorch FP32 vs ONNX Runtime FP32 vs ONNX Runtime INT8.

    python -m scripts.benchmark [--checkpoint models/best.pt] [--fp32 models/policy_fp32.onnx]
                                [--int8 models/policy_int8.onnx] [--out_dir results]
"""

import argparse
import csv
import io
import platform
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import torch

from ppo.utils import load_agent
from scripts.evaluate import evaluate_policy, onnx_policy, torch_policy
from scripts.export_onnx import DeployPolicy, collect_observations

FIELDS = [
    "variant", "p50_us", "p95_us", "p99_us", "mean_us", "inferences_per_s", "size_kb",
    "eval_return_mean", "eval_return_std", "max_abs_action_diff", "p99_pct_of_control_period",
]


def make_session(path):
    """Single-threaded ORT session: mimics one small embedded core and keeps timings stable."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # no thread pool inside an operator
    opts.inter_op_num_threads = 1  # operators run one after another
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # constant folding, fusions
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def time_calls(fn, inputs, warmup, calls):
    """Latency of each call in microseconds. Inputs are real observations, cycled through."""
    # Warm-up: the first calls pay one-off costs (lazy initialization, memory allocation, cold
    # caches) that a control loop running for hours would not see.
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    times = np.empty(calls)
    for i in range(calls):
        x = inputs[i % len(inputs)]  # prepared beforehand, so only the inference itself is timed
        start = time.perf_counter_ns()  # monotonic clock with nanosecond resolution
        fn(x)
        times[i] = time.perf_counter_ns() - start
    return times / 1e3


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch-1 policy inference latency.")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--fp32", default="models/policy_fp32.onnx")
    parser.add_argument("--int8", default="models/policy_int8.onnx")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--calls", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)  # same single-core budget as the ORT sessions

    agent, cfg = load_agent(args.checkpoint)
    policy = DeployPolicy(agent).eval()
    env = gym.make(cfg["env_id"])
    period_us = env.unwrapped.dt * 1e6  # control period: 8 ms for Hopper (125 Hz)
    env.close()
    observations = collect_observations(agent, cfg["env_id"], n=2000, seed=0)
    with torch.inference_mode():
        reference = policy(torch.from_numpy(observations)).numpy()  # PyTorch FP32 actions

    # PyTorch size = the serialized weights + normalizer buffers it needs, comparable to an .onnx file.
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)

    rows = []
    variants = [("PyTorch FP32 (eager)", None), ("ONNX Runtime FP32", Path(args.fp32)),
                ("ONNX Runtime INT8", Path(args.int8))]
    for name, path in variants:
        if path is None:
            inputs = [torch.from_numpy(o[None]) for o in observations]  # (1, obs_dim) tensors
            # inference_mode around the whole timing loop: eager forward only, no autograd bookkeeping.
            with torch.inference_mode():
                latencies = time_calls(policy, inputs, args.warmup, args.calls)
            size_kb = len(buffer.getvalue()) / 1024
            act = torch_policy(agent)
            diff = 0.0  # this is the reference
        else:
            session = make_session(path)
            inputs = [o[None] for o in observations]  # (1, obs_dim) float32 arrays
            latencies = time_calls(lambda x: session.run(None, {"obs": x}), inputs, args.warmup, args.calls)
            size_kb = path.stat().st_size / 1024
            act = onnx_policy(path)
            diff = float(np.abs(session.run(None, {"obs": observations})[0] - reference).max())
        returns, _ = evaluate_policy(act, cfg["env_id"], args.episodes, args.seed_start)
        p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
        rows.append({
            "variant": name, "p50_us": p50, "p95_us": p95, "p99_us": p99, "mean_us": latencies.mean(),
            "inferences_per_s": 1e6 / latencies.mean(), "size_kb": size_kb,
            "eval_return_mean": returns.mean(), "eval_return_std": returns.std(),
            "max_abs_action_diff": diff, "p99_pct_of_control_period": 100 * p99 / period_us,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "benchmark.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    header = ("| Variant | p50 (µs) | p95 (µs) | p99 (µs) | mean (µs) | inferences/s | size (KB) | "
              f"eval return ({args.episodes} ep) | max abs action diff vs PyTorch | p99 % of {period_us / 1000:.0f} ms |")
    lines = [header, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['p50_us']:.1f} | {r['p95_us']:.1f} | {r['p99_us']:.1f} | {r['mean_us']:.1f} | "
            f"{r['inferences_per_s']:,.0f} | {r['size_kb']:.1f} | {r['eval_return_mean']:.1f} ± {r['eval_return_std']:.1f} | "
            f"{r['max_abs_action_diff']:.1e} | {r['p99_pct_of_control_period']:.2f}% |"
        )
    table = "\n".join(lines)
    seeds = f"{args.seed_start}..{args.seed_start + args.episodes - 1}"
    report = "\n".join([
        "# Inference latency benchmark",
        "",
        f"- CPU: {cpu_name()} ({platform.machine()}) | OS: {platform.platform()}",
        f"- Python {sys.version.split()[0]} | torch {torch.__version__} | onnxruntime {ort.__version__} | "
        f"onnx {onnx.__version__} | numpy {np.__version__} | gymnasium {gym.__version__} | mujoco {mujoco.__version__}",
        f"- Setup: batch 1 (one control step), one thread everywhere, {args.warmup} warm-up + {args.calls:,} timed "
        "calls on real observations, `time.perf_counter_ns()` around each call; ORT graph optimization ORT_ENABLE_ALL",
        f"- Policy: `{args.checkpoint}` ({cfg['exp_name']}, seed {cfg['seed']}) | control period "
        f"{period_us / 1000:.0f} ms (`env.unwrapped.dt`) | eval seeds {seeds}, deterministic actions",
        "",
        table,
        "",
    ])
    (out_dir / "benchmark.md").write_text(report)
    print(report)
    print(f"wrote {out_dir / 'benchmark.md'} and {out_dir / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
