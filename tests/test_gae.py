import torch

from ppo.gae import compute_gae


def test_gae_hand_computed_with_done_in_the_middle():
    # 3 steps, 1 env, shape (T, num_envs) like the rollout buffer. float64 so the 1e-6 tolerance
    # tests the maths, not float32 rounding.
    rewards = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    values = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float64)
    # dones[1] = 1: obs[1] starts a new episode, so the transition at t=0 was terminal.
    dones = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)
    next_value = torch.tensor([2.0], dtype=torch.float64)
    next_done = torch.tensor([0.0], dtype=torch.float64)
    # gamma != lam so that swapping them in the code would fail the test.
    gamma, lam = 0.9, 0.8

    advantages, returns = compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)

    # Hand computation, backwards in time:
    # t=2: not terminal, bootstrap from next_value.
    #      delta2 = 3 + 0.9 * 2.0 - 1.5 = 3.3;              A2 = 3.3
    # t=1: obs[2] continues the episode, bootstrap from values[2].
    #      delta1 = 2 + 0.9 * 1.5 - 1.0 = 2.35;             A1 = 2.35 + 0.9 * 0.8 * 3.3 = 4.726
    # t=0: dones[1] = 1, so no bootstrap and no recursion across the boundary.
    #      delta0 = 1 - 0.5 = 0.5;                           A0 = 0.5
    expected_adv = torch.tensor([[0.5], [4.726], [3.3]], dtype=torch.float64)
    # returns = A + V; at the terminal step t=0 it equals the reward, 1.0, as it should.
    expected_ret = torch.tensor([[1.0], [5.726], [4.8]], dtype=torch.float64)

    torch.testing.assert_close(advantages, expected_adv, rtol=0, atol=1e-6)
    torch.testing.assert_close(returns, expected_ret, rtol=0, atol=1e-6)
