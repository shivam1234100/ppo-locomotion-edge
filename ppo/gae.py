"""Generalized Advantage Estimation (Schulman et al., 2016).

A[t] is an exponentially weighted average of n-step advantage estimates: lam = 0 gives the
one-step TD error (low variance, more bias), lam = 1 gives the Monte Carlo return minus V
(unbiased, high variance).
"""

import torch


def compute_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """Advantages and value targets for one rollout.

    rewards, values, dones: tensors of shape (T, num_envs) from the rollout buffer.
    next_value, next_done: shape (num_envs,), for the observation *after* the last stored step,
    which is needed to bootstrap the final transition.

    CleanRL convention: dones[t] = 1 means obs[t] is the FIRST observation of a new episode,
    i.e. the transition at step t-1 ended an episode. That is why step t looks at dones[t + 1].
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0  # A[T] = 0: nothing is known beyond the rollout except the bootstrapped value
    # Backwards, because A[t] is defined recursively in terms of A[t + 1].
    for t in reversed(range(T)):
        if t == T - 1:
            # The last step's successor is not in the buffer; use the bootstrap inputs instead.
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        # One-step TD error. next_nonterminal = 0 at an episode boundary drops V(s_{t+1}), which
        # (with same-step autoreset) belongs to the NEXT episode's first state and must not leak back.
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        # Same mask cuts the recursion, so advantages never flow across an episode boundary.
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    # Value targets: A = R - V, so R = A + V. These are TD(lambda) returns, used to train the critic.
    returns = advantages + values
    return advantages, returns
