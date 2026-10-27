"""Seeding, run directories, CSV / W&B logging and checkpoints."""

import csv
import dataclasses
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ppo.agent import Agent

EPISODE_FIELDS = ["global_step", "episodic_return", "episodic_length"]
METRIC_FIELDS = [
    "global_step", "learning_rate", "value_loss", "policy_loss", "entropy",
    "approx_kl", "clipfrac", "explained_variance", "sps",
]
# W&B panel prefix per metric (CleanRL layout): progress under charts/, optimisation diagnostics
# under losses/. Anything not listed here goes to losses/.
_CHART_METRICS = {"learning_rate", "sps", "episodic_return", "episodic_length"}


def seed_everything(seed):
    # Python, NumPy (minibatch shuffling) and PyTorch (weight init, action sampling) each have
    # their own RNG; all three must be seeded for a run to be repeatable.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_run_name(cfg):
    # The unix-time suffix stops a repeated run of the same config and seed from overwriting the old one.
    return f"{cfg.env_id}__{cfg.exp_name}__seed{cfg.seed}__{int(time.time())}"


class RunLogger:
    """Writes runs/{run_name}/ (config.yaml, episodes.csv, train_metrics.csv) and, optionally, W&B."""

    def __init__(self, cfg, run_name, root="runs"):
        self.run_dir = Path(root) / run_name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        # The fully resolved config (YAML + CLI overrides) makes every run reproducible, and the
        # plotting script groups runs by its exp_name.
        with open(self.run_dir / "config.yaml", "w") as fh:
            yaml.safe_dump(dataclasses.asdict(cfg), fh, sort_keys=False)
        self._episodes_fh, self._episodes = self._open_csv("episodes.csv", EPISODE_FIELDS)
        self._metrics_fh, self._metrics = self._open_csv("train_metrics.csv", METRIC_FIELDS)

        self.wandb_run = None
        if cfg.track:
            import wandb  # imported lazily: slow to import, and only needed when tracking is on

            self.wandb_run = wandb.init(
                project=cfg.wandb_project,
                group=cfg.exp_name,  # groups the seeds of one ablation arm together in the UI
                name=run_name,
                config=dataclasses.asdict(cfg),
            )
            # Plot everything against environment steps, not W&B's internal call counter.
            self.wandb_run.define_metric("global_step")
            self.wandb_run.define_metric("*", step_metric="global_step")

    def _open_csv(self, name, fields):
        fh = open(self.run_dir / name, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=fields)  # raises on unexpected keys: catches typos
        writer.writeheader()
        return fh, writer

    def log_episode(self, global_step, episodic_return, episodic_length):
        row = {"global_step": global_step, "episodic_return": episodic_return, "episodic_length": episodic_length}
        self._episodes.writerow(row)
        # Flush every row, so progress can be watched (and plotted) while a run is still going,
        # and nothing is lost if a run is killed.
        self._episodes_fh.flush()
        self._wandb_log(row)

    def log_metrics(self, row):
        self._metrics.writerow(row)
        self._metrics_fh.flush()
        self._wandb_log(row)

    def _wandb_log(self, row):
        if self.wandb_run is None:
            return
        payload = {"global_step": row["global_step"]}
        for key, value in row.items():
            if key != "global_step":
                payload[f"{'charts' if key in _CHART_METRICS else 'losses'}/{key}"] = value
        self.wandb_run.log(payload)

    def close(self):
        self._episodes_fh.close()
        self._metrics_fh.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def save_checkpoint(path, agent, cfg, global_step):
    torch.save(
        {
            # state_dict includes the normalizer buffers (mean/var/count), so the stats travel with the weights.
            "agent": agent.state_dict(),
            # A plain dict, not the Config dataclass: torch.load defaults to weights_only=True
            # (PyTorch >= 2.6), which refuses to unpickle custom classes but accepts plain containers.
            "config": dataclasses.asdict(cfg),
            "global_step": global_step,
        },
        path,
    )


def load_agent(path):
    """Rebuild an Agent from a checkpoint written by save_checkpoint. Returns (agent, config dict)."""
    ckpt = torch.load(path, map_location="cpu")  # weights_only=True by default: loads tensors/containers only
    state = ckpt["agent"]
    # Network sizes come from the weight shapes, so no environment has to be built just to load a model.
    obs_dim = state["actor_mean.0.weight"].shape[1]  # first Linear is (64, obs_dim)
    act_dim = state["actor_logstd"].shape[1]  # (1, act_dim)
    # obs_norm is not in the state_dict (it is a plain bool), so it comes from the saved config.
    agent = Agent(obs_dim, act_dim, obs_norm=ckpt["config"]["obs_norm"])
    agent.load_state_dict(state)  # strict: fails loudly if any weight or normalizer buffer is missing
    agent.eval()
    return agent, ckpt["config"]
