"""Record one episode of the trained policy (or of random actions) as a GIF.

    python -m scripts.record_gif [--checkpoint models/best.pt] [--out assets/hopper.gif]
    python -m scripts.record_gif --random      # "before" clip: assets/hopper_random.gif

On headless Linux set MUJOCO_GL=egl (or osmesa) first.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch

from ppo.utils import load_agent
from scripts.evaluate import torch_policy

MAX_MB = 8.0  # keep the README demo light enough to load quickly on GitHub
SMALL_SIZE = 320  # fallback render size (px) if the GIF is too large


def record(env_id, act, seed, max_steps, every, camera, size=None):
    """One episode; returns (frames, episode return, steps). `act` is None for random actions."""
    kwargs = {"width": size, "height": size} if size else {}
    if camera:
        kwargs["camera_name"] = camera
    env = gym.make(env_id, render_mode="rgb_array", **kwargs)
    env.action_space.seed(seed)  # so the random-action clip is reproducible too
    obs, _ = env.reset(seed=seed)
    frames, ep_return, steps = [env.render()], 0.0, 0
    for steps in range(1, max_steps + 1):
        action = env.action_space.sample() if act is None else act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += reward
        if steps % every == 0:  # every 3rd frame: 3 x 8 ms = 24 ms of sim time per GIF frame
            frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    return frames, ep_return, steps


def main():
    parser = argparse.ArgumentParser(description="Record a GIF of one Hopper episode.")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--random", action="store_true", help="random actions instead of the policy")
    parser.add_argument("--env_id", default="Hopper-v5", help="used with --random (otherwise from the checkpoint)")
    parser.add_argument("--out", default=None, help="default: assets/hopper.gif, or assets/hopper_random.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--every", type=int, default=3, help="keep every n-th frame")
    parser.add_argument("--fps", type=int, default=30)
    # Hopper's model defines a "track" camera (mode="trackcom") that follows the robot's centre of
    # mass. The env's default free camera stays put, so a good policy hops out of view within ~1.5 s.
    parser.add_argument("--camera", default="track", help="MuJoCo camera name; '' for the env default")
    args = parser.parse_args()
    torch.set_num_threads(1)

    if args.random:
        env_id, act = args.env_id, None
    else:
        agent, cfg = load_agent(args.checkpoint)
        env_id, act = cfg["env_id"], torch_policy(agent)  # deterministic (mean) actions
    out = Path(args.out or ("assets/hopper_random.gif" if args.random else "assets/hopper.gif"))
    out.parent.mkdir(parents=True, exist_ok=True)

    frames, ep_return, steps = record(env_id, act, args.seed, args.max_steps, args.every, args.camera)
    # duration is milliseconds per frame (Pillow GIF writer); loop=0 repeats forever.
    iio.imwrite(out, np.stack(frames), duration=1000 / args.fps, loop=0)
    if out.stat().st_size > MAX_MB * 2**20:
        # Too big: re-record at a smaller render size. Same seed and a deterministic policy (or the
        # seeded action space) give exactly the same episode.
        frames, ep_return, steps = record(env_id, act, args.seed, args.max_steps, args.every, args.camera,
                                          SMALL_SIZE)
        iio.imwrite(out, np.stack(frames), duration=1000 / args.fps, loop=0)

    h, w = frames[0].shape[:2]
    print(f"{'random actions' if args.random else args.checkpoint}: episode return {ep_return:.1f}, "
          f"{steps} steps -> {len(frames)} frames {w}x{h} at {args.fps} fps -> {out} "
          f"({out.stat().st_size / 2**20:.2f} MB)")


if __name__ == "__main__":
    main()
