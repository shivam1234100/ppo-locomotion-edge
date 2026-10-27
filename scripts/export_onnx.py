"""Export the deployable policy to ONNX (FP32) and check numerical parity with PyTorch.

    python -m scripts.export_onnx [--checkpoint models/best.pt] [--out models/policy_fp32.onnx]
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from ppo.utils import load_agent
from scripts.evaluate import torch_policy

OPSET = 17
PARITY_TOL = 1e-5


class DeployPolicy(nn.Module):
    """What ships to the robot: frozen normalizer -> actor mean -> clamp. Raw observations in, actions out.

    The critic and log-std are only needed for training and are left out. Normalization is inside
    the graph, so the robot side has no preprocessing code that could drift from training.
    """

    def __init__(self, agent):
        super().__init__()
        self.normalizer = agent.normalizer
        self.actor_mean = agent.actor_mean

    def forward(self, obs):
        # Clamp to the actuator limits: whatever the input, the policy can never command an
        # out-of-range action (a simple safety envelope).
        return torch.clamp(self.actor_mean(self.normalizer(obs)), -1.0, 1.0)


def export_onnx(policy, path, obs_dim):
    """Export to a single self-contained ONNX file and validate it with the ONNX checker."""
    policy.eval()
    example = torch.zeros(1, obs_dim)  # fixes dtype and feature size; the batch axis is made dynamic below
    torch.onnx.export(
        policy,
        (example,),
        str(path),
        input_names=["obs"],
        output_names=["action"],
        # Dynamic batch: the same file serves one control step (batch 1) or batched evaluation.
        dynamic_shapes={"obs": {0: torch.export.Dim("batch")}},
        # The torch.export-based exporter (PyTorch's default since 2.9; the TorchScript one is deprecated).
        # It builds at opset 18 and converts down to 17, logging a warning while it does so.
        opset_version=OPSET,
        dynamo=True,
        # Weights inside the .onnx file. The default would write them to a separate .data file,
        # and the deployed policy should be one self-contained artifact.
        external_data=False,
        verbose=False,
    )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)  # structural validity: operator signatures, types, graph wiring
    return model


def collect_observations(agent, env_id, n, seed):
    """Raw observations from rolling out the policy: realistic inputs, not random noise."""
    act = torch_policy(agent)
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    observations = []
    for _ in range(n):
        observations.append(obs)
        obs, _, terminated, truncated, _ = env.step(act(obs))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.asarray(observations, dtype=np.float32)  # the ONNX input is float32


def max_parity_diff(policy, onnx_path, observations):
    """Largest absolute action difference between PyTorch and ONNX Runtime on the same inputs."""
    with torch.inference_mode():
        expected = policy(torch.from_numpy(observations)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"obs": observations})[0]  # all observations as one batch
    return float(np.abs(actual - expected).max())


def main():
    parser = argparse.ArgumentParser(description="Export a checkpoint's policy to ONNX with a parity check.")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--out", default="models/policy_fp32.onnx")
    parser.add_argument("--parity_obs", type=int, default=2000, help="real observations for the parity check")
    parser.add_argument("--seed", type=int, default=0, help="env seed for collecting those observations")
    args = parser.parse_args()
    torch.set_num_threads(1)

    agent, cfg = load_agent(args.checkpoint)
    policy = DeployPolicy(agent).eval()
    obs_dim = agent.actor_mean[0].in_features
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = export_onnx(policy, out, obs_dim)

    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    dims = lambda v: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
    inp, outp = model.graph.input[0], model.graph.output[0]
    print(f"exported {args.checkpoint} [{cfg['exp_name']} seed {cfg['seed']}, obs_norm={cfg['obs_norm']}] -> {out}")
    print(f"  opset {opset} | input {inp.name}{dims(inp)} -> output {outp.name}{dims(outp)} | "
          f"{out.stat().st_size / 1024:.1f} KB | ops {sorted({n.op_type for n in model.graph.node})}")

    observations = collect_observations(agent, cfg["env_id"], args.parity_obs, args.seed)
    diff = max_parity_diff(policy, out, observations)
    print(f"parity: max |PyTorch - ONNX Runtime| over {len(observations)} real observations = {diff:.2e} "
          f"(tolerance {PARITY_TOL:.0e})")
    assert diff < PARITY_TOL, "ONNX output does not match PyTorch"


if __name__ == "__main__":
    main()
