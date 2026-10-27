"""Fixed-rate control loop: the INT8 ONNX policy driving Hopper at its real control rate (125 Hz).

    python -m scripts.control_loop_demo [--onnx models/policy_int8.onnx] [--seconds 10] [--out results/control_loop.md]

Each tick: read obs -> infer -> env.step -> wait for the next tick. The simulator stands in for the
robot, so this measures whether inference plus I/O fits the real-time budget on this CPU.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from scripts.benchmark import cpu_name, make_session


def wait_until(deadline, spin_margin):
    """Block until perf_counter() reaches `deadline`; return the actual wake-up time.

    time.sleep() alone wakes up late by a variable amount (OS timer granularity), so we sleep
    through most of the gap and busy-wait on the clock for the final `spin_margin` seconds.
    """
    remaining = deadline - time.perf_counter()
    if remaining > spin_margin:
        time.sleep(remaining - spin_margin)
    while (now := time.perf_counter()) < deadline:
        pass
    return now  # if the deadline had already passed (an overrun), this returns at once, late


def pct(values_s):
    """p50 / p99 / max of a timing array, in microseconds."""
    return np.percentile(values_s, 50) * 1e6, np.percentile(values_s, 99) * 1e6, values_s.max() * 1e6


def main():
    parser = argparse.ArgumentParser(description="Run the ONNX policy in a fixed-rate real-time loop.")
    parser.add_argument("--onnx", default="models/policy_int8.onnx")
    parser.add_argument("--env_id", default="Hopper-v5")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    # Measured on macOS: sleep() woke this process ~1.4 ms late, so a 1 ms margin left every tick
    # ~0.4 ms late (jitter ~230 µs); 3 ms absorbs it (jitter ~2 µs). Spinning the whole period was
    # worse: a thread pinned at 100% CPU got preempted for several ms. Lower it on a real-time kernel.
    parser.add_argument("--spin_ms", type=float, default=3.0, help="busy-wait this long before each tick")
    parser.add_argument("--out", default="results/control_loop.md")
    args = parser.parse_args()
    spin_margin = args.spin_ms / 1000

    session = make_session(args.onnx)  # same single-threaded settings as the benchmark
    env = gym.make(args.env_id)
    period = env.unwrapped.dt  # 0.008 s: 4 MuJoCo substeps of 2 ms each
    n_ticks = round(args.seconds / period)
    obs, _ = env.reset(seed=args.seed)

    start_error, infer, work = np.empty(n_ticks), np.empty(n_ticks), np.empty(n_ticks)
    returns, ep_return = [], 0.0
    t0 = time.perf_counter() + period  # first tick one period from now, so setup cost is not counted
    for k in range(n_ticks):
        # Absolute schedule t0 + k * period, rather than sleep(period) after the work: per-tick errors
        # do not accumulate, so the loop cannot drift away from 125 Hz.
        deadline = t0 + k * period
        tick_start = wait_until(deadline, spin_margin)
        start_error[k] = tick_start - deadline

        x = obs[None].astype(np.float32)  # read the "sensors": raw observation, batch of 1
        t_infer = time.perf_counter()
        action = session.run(None, {"obs": x})[0][0]
        infer[k] = time.perf_counter() - t_infer
        obs, reward, terminated, truncated, _ = env.step(action)  # send the command, advance the robot
        work[k] = time.perf_counter() - tick_start

        ep_return += reward
        if terminated or truncated:
            # Resetting is simulator housekeeping, not part of a control step, so it is outside the
            # work timer. Any delay it causes still shows up in the next tick's start error.
            returns.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()
    wall = time.perf_counter() - t0
    env.close()

    misses = int((work > period).sum())  # the tick's work did not finish within its 8 ms slot
    jitter_us = start_error.std() * 1e6
    rows = [("inference", pct(infer)), ("work per tick (obs -> infer -> env.step)", pct(work)),
            ("tick-start error (lateness)", pct(start_error))]
    lines = [
        "# Fixed-rate control loop",
        "",
        f"- Policy: `{args.onnx}` in ONNX Runtime (1 thread) driving {args.env_id} | CPU: {cpu_name()}",
        f"- Loop: {n_ticks:,} ticks at {period * 1000:.0f} ms ({1 / period:.0f} Hz), {wall:.2f} s wall time; "
        f"absolute `perf_counter` deadlines, sleep then spin for the last {args.spin_ms:g} ms",
        "",
        "| Timing (µs) | p50 | p99 | max |",
        "|---|---|---|---|",
        *[f"| {name} | {p50:.1f} | {p99:.1f} | {mx:.1f} |" for name, (p50, p99, mx) in rows],
        "",
        f"- Deadline misses (work > {period * 1000:.0f} ms): **{misses} / {n_ticks:,}**",
        f"- Jitter (std of tick-start error): **{jitter_us:.1f} µs**",
        f"- Worst-case work per tick: {work.max() * 1e6:.1f} µs = {100 * work.max() / period:.2f}% of the period",
        f"- Episodes finished during the run: {len(returns)}"
        + (f" (mean return {np.mean(returns):.1f})" if returns else ""),
        "",
    ]
    report = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
