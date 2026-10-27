import numpy as np
import onnxruntime as ort
import torch

from ppo.agent import Agent, layer_init
from scripts.export_onnx import DeployPolicy, export_onnx


def test_onnx_matches_pytorch(tmp_path):
    # Untrained policy, so the test needs no trained model and runs in a few seconds.
    torch.manual_seed(0)
    agent = Agent(obs_dim=11, act_dim=3, obs_norm=True)
    # Non-trivial normalizer stats. With mean 0 / var 1 the normalizer is an identity, so an export
    # that dropped or mis-wired it would go unnoticed.
    agent.normalizer.mean.copy_(torch.randn(11))
    agent.normalizer.var.copy_(torch.rand(11) * 5 + 0.1)
    # Gain 1 instead of 0.01 on the last actor layer, so actions span the full range: ~0.01 outputs
    # would make a 1e-5 tolerance weak, and the clamp would never activate.
    layer_init(agent.actor_mean[4], std=1.0)
    policy = DeployPolicy(agent).eval()

    path = tmp_path / "policy.onnx"
    export_onnx(policy, path, obs_dim=11)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 256):  # batch 1 = one control step; 256 exercises the dynamic batch axis
        obs = torch.randn(batch, 11) * 3
        with torch.inference_mode():
            expected = policy(obs).numpy()
        actual = session.run(None, {"obs": obs.numpy()})[0]
        assert actual.shape == (batch, 3)
        assert np.abs(actual - expected).max() < 1e-5
    # Guard on the test itself: some outputs must hit the +-1 clamp, so the clamp is checked too.
    assert (np.abs(expected) == 1.0).any()
