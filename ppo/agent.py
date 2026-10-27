"""Actor-critic network for continuous control (the CleanRL ppo_continuous_action architecture)."""

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from ppo.normalizer import ObsNormalizer


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    # Orthogonal weights keep the activation scale roughly constant from layer to layer; a gain of
    # sqrt(2) is the standard hidden-layer choice ("The 37 Implementation Details of PPO").
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, obs_norm=True):
        super().__init__()
        # Owned by the agent so its buffers end up in the same state_dict (and ONNX graph) as the weights.
        self.normalizer = ObsNormalizer(obs_dim, enabled=obs_norm)
        # Separate actor and critic networks (no shared trunk), so value-loss gradients cannot
        # disturb the policy's features. This is the CleanRL default for MuJoCo.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # gain 1.0: the output is a value, not a hidden feature
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # Gain 0.01: initial action means are ~0, so early exploration comes from the Gaussian
            # noise alone rather than from an arbitrary random bias in the untrained network.
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        # State-independent log-std: one learned exploration scale per action dimension.
        # Shape (1, act_dim) so it broadcasts over a batch; zeros mean std = 1 at the start.
        # Learning log(std) rather than std keeps std positive without any constraint.
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs_n):
        """V(s) for already-normalized observations; shape (batch, 1)."""
        return self.critic(obs_n)

    def get_action_and_value(self, obs_n, action=None):
        """Takes normalized observations. Returns action, log_prob, entropy, value."""
        action_mean = self.actor_mean(obs_n)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))  # same std for every sample
        dist = Normal(action_mean, action_std)
        if action is None:
            # Rollout phase: sample a new action. It is NOT clipped here; the env's ClipAction
            # wrapper clips it, and keeping the raw sample keeps its log_prob exact for the ratio.
            action = dist.sample()
        # Update phase passes the stored action in, to get its log_prob under the *current* policy.
        # Action dimensions are independent, so joint log_prob and entropy are sums over dimensions.
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(obs_n)

    def deterministic_action(self, raw_obs):
        """Mean action from RAW observations, clamped to the actuator limits. Used for eval and export."""
        # The normalizer is applied here (frozen; update() is never called) so callers pass sensor
        # readings straight in. The clamp keeps every command inside [-1, 1]: a simple safety envelope.
        return torch.clamp(self.actor_mean(self.normalizer(raw_obs)), -1.0, 1.0)

    def actor_param_count(self):
        """Actor size: mean network plus log-std. Only the mean network is exported; the critic never is."""
        return sum(p.numel() for p in self.actor_mean.parameters()) + self.actor_logstd.numel()
