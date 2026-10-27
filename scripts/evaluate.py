"""Evaluate a trained policy with deterministic (mean) actions on true, unnormalized returns.

    python -m scripts.evaluate --checkpoint runs/<run>/checkpoints/final.pt [more.pt ...] [--save_best models/best.pt]
    python -m scripts.evaluate --onnx models/policy_fp32.onnx
"""

import argparse
import shutil

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

from ppo.utils import load_agent


def evaluate_policy(act, env_id="Hopper-v5", episodes=10, seed_start=1000):
    """Run `episodes` episodes; `act` maps one raw observation (obs_dim,) to one action (act_dim,).

    Returns (returns, lengths) as numpy arrays.
    """
    # Plain env: no reward normalization, so these are the true returns. Any observation
    # normalization happens inside the policy, with frozen statistics.
    env = gym.make(env_id)
    returns, lengths = [], []
    for i in range(episodes):
        # Fixed per-episode seeds: every policy (PyTorch, ONNX, INT8) faces the same start states, and
        # 1000+ keeps them apart from the training seeds 1-3.
        obs, _ = env.reset(seed=seed_start + i)
        done, ep_return, ep_length = False, 0.0, 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(act(obs))
            ep_return += reward
            ep_length += 1
            done = terminated or truncated  # truncated = the 1000-step time limit
        returns.append(ep_return)
        lengths.append(ep_length)
    env.close()
    return np.array(returns), np.array(lengths)


def torch_policy(agent):
    def act(obs):
        # inference_mode: no autograd bookkeeping at all (cheaper than no_grad); we never backprop here.
        with torch.inference_mode():
            raw = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # add a batch dim of 1
            return agent.deterministic_action(raw)[0].numpy()
    return act


def onnx_policy(path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    def act(obs):
        # The ONNX graph takes raw float32 observations shaped [batch, obs_dim] and returns clamped actions.
        return session.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
    return act


def summarize(returns, lengths):
    return f"return {returns.mean():.1f} ± {returns.std():.1f} | mean length {lengths.mean():.0f}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy checkpoint or ONNX model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", nargs="+", help="one or more .pt checkpoints")
    source.add_argument("--onnx", help="an exported ONNX policy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=1000)
    parser.add_argument("--env_id", default=None, help="defaults to the checkpoint's env (Hopper-v5 for ONNX)")
    parser.add_argument("--save_best", default=None, help="copy the checkpoint with the highest mean return here")
    args = parser.parse_args()
    torch.set_num_threads(1)  # batch-1 inference on a tiny MLP; extra threads only add overhead

    seeds = f"seeds {args.seed_start}..{args.seed_start + args.episodes - 1}"
    if args.onnx:
        returns, lengths = evaluate_policy(onnx_policy(args.onnx), args.env_id or "Hopper-v5",
                                           args.episodes, args.seed_start)
        print(f"{args.onnx}: {summarize(returns, lengths)} ({args.episodes} episodes, {seeds})")
        return

    results = {}
    for path in args.checkpoint:
        agent, cfg = load_agent(path)
        returns, lengths = evaluate_policy(torch_policy(agent), args.env_id or cfg["env_id"],
                                           args.episodes, args.seed_start)
        results[path] = returns.mean()
        print(f"{path} [{cfg['exp_name']} seed {cfg['seed']}]: {summarize(returns, lengths)} "
              f"({args.episodes} episodes, {seeds})")

    if args.save_best:
        best = max(results, key=results.get)
        shutil.copyfile(best, args.save_best)
        print(f"best: {best} (mean return {results[best]:.1f}) -> copied to {args.save_best}")


if __name__ == "__main__":
    main()
