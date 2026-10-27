"""Run configuration: YAML file + CLI overrides -> a typed dataclass.

Precedence: dataclass defaults < YAML file < command line. Any config key can be overridden,
e.g. `--total_timesteps 20000`, `--track`, or `--obs_norm false`.
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class Config:
    # Defaults are the CleanRL continuous-control values (spec Section 5); the YAML normally sets them all.
    exp_name: str = "baseline"
    env_id: str = "Hopper-v5"
    seed: int = 1
    total_timesteps: int = 1_000_000
    num_envs: int = 1
    num_steps: int = 2048
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    obs_norm: bool = True
    reward_norm: bool = True
    checkpoint_every: int = 100_000
    track: bool = False
    wandb_project: str = "ppo-locomotion-edge"

    # Derived values (init=False): computed from the inputs so they can never be inconsistent,
    # and still written to the saved config.yaml for reference.
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)
    num_iterations: int = field(init=False)

    def __post_init__(self):
        self.batch_size = self.num_envs * self.num_steps  # 1 x 2048
        # Every minibatch must be the same size, so the batch has to split evenly.
        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches  # 2048 / 32 = 64
        # Whole rollouts only, so a run does floor(total / batch) iterations (1M -> 488).
        self.num_iterations = self.total_timesteps // self.batch_size
        assert self.num_iterations >= 1, "total_timesteps must cover at least one rollout"


def _input_fields():
    """Fields that can be set from YAML/CLI (excludes the derived init=False ones)."""
    return [f for f in dataclasses.fields(Config) if f.init]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def _coerce(f, value):
    """Cast a YAML or CLI value to the field's declared type (CLI values arrive as strings)."""
    if f.type is bool:
        return _parse_bool(value)
    if f.type == Optional[float]:
        return None if value is None or str(value).lower() in ("null", "none") else float(value)
    return f.type(value)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="Train PPO on a Gymnasium MuJoCo task.")
    parser.add_argument("--config", required=True, help="YAML file with the run's hyperparameters")
    # One optional flag per config key. default=None lets us tell "not given" from a real value.
    for f in _input_fields():
        if f.type is bool:
            # nargs="?" with const=True: a bare `--track` means true; `--obs_norm false` also works.
            parser.add_argument(f"--{f.name}", type=_parse_bool, nargs="?", const=True, default=None)
        else:
            parser.add_argument(f"--{f.name}", default=None)
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        values = yaml.safe_load(fh) or {}
    known = {f.name: f for f in _input_fields()}
    unknown = set(values) - set(known)
    if unknown:
        # Fail loudly on a typo in the YAML instead of silently training with a default.
        raise ValueError(f"unknown config keys in {args.config}: {sorted(unknown)}")
    for name in known:
        cli_value = getattr(args, name)
        if cli_value is not None:
            values[name] = cli_value  # CLI wins over the YAML
    return Config(**{name: _coerce(known[name], value) for name, value in values.items()})
