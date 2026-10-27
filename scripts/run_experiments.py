"""Launch the observation-normalization ablation: {baseline, no_obsnorm} x seeds {1, 2, 3}.

    python scripts/run_experiments.py [--track] [--total_timesteps N] [--max_parallel K]

Pure Python (subprocess + pathlib) rather than a shell script, so it runs unchanged on
macOS, Linux and Windows.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root, so the script works from any directory
CONFIGS = {"baseline": "configs/hopper_baseline.yaml", "no_obsnorm": "configs/hopper_no_obsnorm.yaml"}
SEEDS = [1, 2, 3]
STATUS_EVERY = 60  # seconds between status lines
CHECK_EVERY = 5  # seconds between checks for finished runs, so a queued run starts promptly
# Parses train.py's per-iteration console line, e.g. "iter  12/488 | step 24576 | return(last 10) 312.4 | SPS 2100".
ITER_LINE = re.compile(r"iter\s+(\d+)/(\d+) \| step\s+(\d+) \| return\(last 10\)\s+(\S+) \| SPS (\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 6-run obs-normalization ablation in parallel.")
    parser.add_argument("--track", action="store_true", help="log every run to W&B")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="override the step budget (applied to both arms equally)")
    # Leave one core free for the OS and this launcher, so the training processes are not preempted.
    parser.add_argument("--max_parallel", type=int, default=min(6, (os.cpu_count() or 2) - 1))
    return parser.parse_args()


def progress(log_path):
    """Short progress summary from the last iteration line in a run's log."""
    matches = ITER_LINE.findall(log_path.read_text(errors="replace"))
    if not matches:
        return "starting"
    it, n, _, ret, sps = matches[-1]
    return f"{100 * int(it) // int(n)}% R={float(ret):.0f} SPS={sps}"


def main():
    args = parse_args()
    log_dir = ROOT / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # One BLAS/OpenMP thread per child: six runs each spawning a full thread pool would fight over
    # the cores and all slow down (train.py also calls torch.set_num_threads(1)).
    child_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    # Seed-major order interleaves the two arms, so if runs have to be batched (max_parallel < 6)
    # both arms are always in progress together and neither gets a head start.
    queue = [(exp, seed) for seed in SEEDS for exp in CONFIGS]
    total = len(queue)
    running = {}  # name -> (process, open log file, log path)
    results = {}  # name -> exit code
    last_status = 0.0
    print(f"launching {total} runs, max {args.max_parallel} in parallel", flush=True)

    try:
        while queue or running:
            while queue and len(running) < args.max_parallel:
                exp, seed = queue.pop(0)
                name = f"{exp}_seed{seed}"
                # sys.executable: children use the same (venv) interpreter as the launcher.
                cmd = [sys.executable, "-m", "ppo.train", "--config", CONFIGS[exp], "--seed", str(seed)]
                if args.track:
                    cmd.append("--track")
                if args.total_timesteps:
                    cmd += ["--total_timesteps", str(args.total_timesteps)]
                log_path = log_dir / f"{name}.log"
                log_fh = open(log_path, "w")
                # cwd=ROOT so `-m ppo.train` resolves and every run writes under ROOT/runs/.
                proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env, stdout=log_fh, stderr=subprocess.STDOUT)
                running[name] = (proc, log_fh, log_path)
                print(f"started {name} (pid {proc.pid}) -> {log_path.relative_to(ROOT)}", flush=True)

            for name, (proc, log_fh, _) in list(running.items()):
                if proc.poll() is not None:  # poll() returns the exit code once the process has ended
                    log_fh.close()
                    results[name] = proc.returncode
                    del running[name]
                    print(f"finished {name} (exit {proc.returncode})", flush=True)

            if time.time() - last_status >= STATUS_EVERY:
                last_status = time.time()
                parts = [f"{name} {progress(path)}" for name, (_, _, path) in running.items()]
                head = (f"[{time.strftime('%H:%M:%S')}] {len(results)}/{total} done, "
                        f"{len(running)} running, {len(queue)} queued")
                print(head + "".join(f" | {p}" for p in parts), flush=True)

            if queue or running:
                time.sleep(CHECK_EVERY)
    except KeyboardInterrupt:
        # Stop the children too, instead of leaving orphaned training runs burning CPU.
        for proc, log_fh, _ in running.values():
            proc.terminate()
            proc.wait()
            log_fh.close()
        raise

    failed = {name: code for name, code in results.items() if code != 0}
    print(f"[{time.strftime('%H:%M:%S')}] all runs finished: {total - len(failed)}/{total} succeeded", flush=True)
    for name, code in failed.items():
        print(f"  FAILED {name} (exit {code}), see runs/logs/{name}.log", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
