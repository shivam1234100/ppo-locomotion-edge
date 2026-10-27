"""PPO training loop (entry point).

    python -m ppo.train --config configs/hopper_baseline.yaml --seed 1 [--track] [--total_timesteps 20000]
"""

import dataclasses
import time
from collections import deque

import numpy as np
import torch
import yaml
from torch import nn, optim

from ppo.agent import Agent
from ppo.config import parse_config
from ppo.envs import episode_stats, make_vector_env
from ppo.gae import compute_gae
from ppo.utils import RunLogger, make_run_name, save_checkpoint, seed_everything


def observe(agent, raw_obs):
    """Fold newly arrived raw observations into the running stats, then normalize them.

    Called exactly once per observation, when it arrives. The normalized result is what gets
    stored in the rollout buffer, so it is never re-normalized later with stats that have since
    moved (that would make the PPO probability ratio start away from 1).
    """
    raw = torch.as_tensor(raw_obs, dtype=torch.float32)  # env returns float64; the model is float32
    agent.normalizer.update(raw)  # no-op when obs_norm is off
    with torch.no_grad():
        return agent.normalizer(raw)


def train(cfg):
    seed_everything(cfg.seed)
    # One intra-op thread: a 64-unit MLP gains nothing from more, and six parallel runs would
    # otherwise oversubscribe the cores and all slow down.
    torch.set_num_threads(1)
    run_name = make_run_name(cfg)
    print(f"run: {run_name}\n" + yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    logger = RunLogger(cfg, run_name)
    envs = make_vector_env(cfg.env_id, cfg.num_envs, cfg.seed, cfg.gamma, cfg.reward_norm)

    try:
        obs_dim = envs.single_observation_space.shape[0]
        act_dim = envs.single_action_space.shape[0]
        agent = Agent(obs_dim, act_dim, obs_norm=cfg.obs_norm)
        print(f"actor parameters: {agent.actor_param_count():,}")
        # eps=1e-5 instead of PyTorch's 1e-8 (a detail of the original PPO code that CleanRL keeps):
        # stops tiny second-moment estimates from producing huge parameter steps.
        optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

        # Rollout buffers, shape (num_steps, num_envs, ...). Observations are stored already normalized.
        T, N = cfg.num_steps, cfg.num_envs
        obs = torch.zeros((T, N, obs_dim))
        actions = torch.zeros((T, N, act_dim))
        logprobs = torch.zeros((T, N))
        rewards = torch.zeros((T, N))
        dones = torch.zeros((T, N))
        values = torch.zeros((T, N))

        global_step = 0
        next_checkpoint = cfg.checkpoint_every
        recent_returns = deque(maxlen=10)  # console readout: mean of the last 10 finished episodes
        start_time = time.time()

        raw_obs, _ = envs.reset(seed=cfg.seed)  # seeds sub-env i with seed + i
        envs.action_space.seed(cfg.seed)
        next_obs_n = observe(agent, raw_obs)
        # 0 for the very first obs: no transition precedes it in the buffer, so there is nothing to cut.
        next_done = torch.zeros(N)

        for iteration in range(1, cfg.num_iterations + 1):
            # (1) Anneal the LR linearly to 0: large steps early, fine-tuning late.
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            # (2) Collect a rollout of num_steps steps with the current (now fixed) policy.
            for step in range(T):
                global_step += N
                obs[step] = next_obs_n
                dones[step] = next_done  # 1 if obs[step] starts a new episode (convention in gae.py)
                with torch.no_grad():  # acting only; gradients are recomputed in the update phase
                    action, logprob, _, value = agent.get_action_and_value(next_obs_n)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob  # log pi_old(a|s), the denominator of the PPO ratio

                raw_obs, reward, terminations, truncations, infos = envs.step(action.numpy())
                # Known simplification (documented in the README): a time-limit truncation is treated
                # like a real termination, so the 1000th step is not bootstrapped with V(s).
                next_done = torch.as_tensor(np.logical_or(terminations, truncations), dtype=torch.float32)
                rewards[step] = torch.as_tensor(reward, dtype=torch.float32)  # normalized + clipped reward
                # With same-step autoreset, raw_obs is already the next episode's first obs on a done step.
                next_obs_n = observe(agent, raw_obs)

                for _, ep_return, ep_length in episode_stats(infos):
                    logger.log_episode(global_step, ep_return, ep_length)  # raw return, see envs.py
                    recent_returns.append(ep_return)

            # (3) Bootstrap the value of the observation after the last step, then run GAE.
            with torch.no_grad():
                next_value = agent.get_value(next_obs_n).reshape(-1)
                advantages, returns = compute_gae(
                    rewards, values, dones, next_value, next_done, cfg.gamma, cfg.gae_lambda
                )

            # (4) Flatten (num_steps, num_envs) into one batch and optimize.
            b_obs = obs.reshape((-1, obs_dim))
            b_actions = actions.reshape((-1, act_dim))
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(cfg.batch_size)
            clipfracs = []
            for epoch in range(cfg.update_epochs):
                # Fresh shuffle each epoch, so minibatches are decorrelated (consecutive steps are similar).
                np.random.shuffle(b_inds)
                for start in range(0, cfg.batch_size, cfg.minibatch_size):
                    mb = b_inds[start:start + cfg.minibatch_size]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()  # pi_new(a|s) / pi_old(a|s), computed in log space for stability

                    with torch.no_grad():
                        # Low-variance KL(old || new) estimator, (r - 1) - log r, always >= 0.
                        approx_kl = ((ratio - 1) - logratio).mean()
                        # Share of samples where the clip is active: how hard the trust region is binding.
                        clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                    mb_adv = b_advantages[mb]
                    if cfg.norm_adv:
                        # Zero mean, unit std per minibatch: the gradient scale no longer depends on reward scale.
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    # Clipped surrogate objective (negated, because we minimize). Taking the max of the
                    # two losses = the min of the two objectives, a pessimistic bound: the policy gets no
                    # credit for moving the ratio beyond 1 +- clip_coef in the advantage's direction.
                    pg_loss = torch.max(
                        -mb_adv * ratio,
                        -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),
                    ).mean()

                    newvalue = newvalue.view(-1)
                    if cfg.clip_vloss:
                        # Same trust-region idea for the critic: the value prediction may not move more
                        # than clip_coef away from the rollout-time value without being penalized.
                        v_unclipped = (newvalue - b_returns[mb]) ** 2
                        v_clipped = b_values[mb] + (newvalue - b_values[mb]).clamp(-cfg.clip_coef, cfg.clip_coef)
                        v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # One optimizer for actor and critic; the entropy bonus is subtracted (we want more
                    # entropy), though with ent_coef = 0 it has no effect here.
                    loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    # Global gradient-norm clipping: a safety net against rare huge gradients.
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                # Optional early stop if the policy has already moved too far from the rollout policy.
                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            # Explained variance: how much of the return variance the critic accounts for
            # (1 = perfect, 0 = no better than predicting the mean, < 0 = worse).
            y_pred, y_true = b_values.numpy(), b_returns.numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            sps = int(global_step / (time.time() - start_time))
            # Losses/KL are from the last minibatch (CleanRL convention): approx_kl is then how far the
            # policy moved by the END of the update. clipfrac is averaged over all minibatches.
            logger.log_metrics({
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": float(np.mean(clipfracs)),
                "explained_variance": float(explained_var),
                "sps": sps,
            })
            mean_return = np.mean(recent_returns) if recent_returns else float("nan")
            print(f"iter {iteration:4d}/{cfg.num_iterations} | step {global_step:8d} | "
                  f"return(last 10) {mean_return:8.1f} | SPS {sps}", flush=True)

            if global_step >= next_checkpoint:
                save_checkpoint(logger.ckpt_dir / f"step_{global_step}.pt", agent, cfg, global_step)
                # Next multiple of checkpoint_every; robust even if it is smaller than one batch.
                next_checkpoint = (global_step // cfg.checkpoint_every + 1) * cfg.checkpoint_every

        save_checkpoint(logger.ckpt_dir / "final.pt", agent, cfg, global_step)
        print(f"done: {global_step} steps in {time.time() - start_time:.0f}s -> {logger.run_dir}")
    finally:
        # Runs even on Ctrl-C or a crash, so the CSVs are closed and the W&B run is finished cleanly.
        envs.close()
        logger.close()


if __name__ == "__main__":
    train(parse_config())
