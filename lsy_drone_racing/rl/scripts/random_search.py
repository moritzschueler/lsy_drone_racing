"""Hyperparameter random search for multi-agent self-play drone racing (IPPO).

Runs several independent training trials, each with randomly sampled competition-reward
hyperparameters (see ``sample_hyperparameters``), logged as its own W&B run.
"""

import random
import time
import traceback
from pathlib import Path
from typing import Any

import fire

import wandb
from lsy_drone_racing.rl.tasks import get_task
from lsy_drone_racing.utils import load_config

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"
DEFAULT_INIT_CHECKPOINT = (
    "lsy_drone_racing/rl/checkpoints/single_agent_racing/"
    "single_agent_racing_20260710-131954_g3.88_best.ckpt"
)

rand = random.SystemRandom()


def sample_hyperparameters() -> dict[str, Any]:
    """Sample a random set of competition-reward hyperparameters for self-play racing."""
    coefficients = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    return {
        "competition_rank_coef": rand.choice(coefficients),
        "competition_segment_lead_coef": rand.choice(coefficients),
        "competition_proximity_coef": rand.choice(coefficients),
        "competition_proximity_threshold": rand.choice([0.1, 0.15, 0.2, 0.25, 0.3]),
        "competition_victory_coef": rand.choice([20.0, 50.0, 100.0]),
        "competition_downwash_coef": rand.choice(coefficients),
    }


def main(
    num_trials: int = 20,
    task: str = "multi_agent_racing",
    config: str = "multi_level2.toml",
    total_timesteps: int = 50_000_000,
    init_checkpoint: str | None = DEFAULT_INIT_CHECKPOINT,
    wandb_enabled: bool = True,
):
    """Run random search over multi-agent self-play competition-reward hyperparameters.

    Args:
        num_trials: Number of independently sampled training runs.
        task: Task name (must be a self-play task, e.g. ``multi_agent_racing``).
        config: Env config file under ``config/`` (needs a multi-drone track, e.g.
            ``multi_level2.toml``).
        total_timesteps: Training length applied to every trial.
        init_checkpoint: Warm-start checkpoint shared by every trial (seeds the ego and the whole
            self-play pool). Pass ``None`` to train from scratch instead.
        wandb_enabled: Whether to log each trial to Weights & Biases.
    """
    task_spec = get_task(task)

    checkpoint_dir = CHECKPOINT_DIR / task
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_config = load_config(Path(__file__).parents[3] / "config" / config)

    print(f"Starting random search for task '{task}' ({num_trials} trials)...")

    for trial in range(num_trials):
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        sampled_kwargs = sample_hyperparameters()
        sampled_kwargs["total_timesteps"] = total_timesteps
        if init_checkpoint:
            sampled_kwargs["init_checkpoint"] = init_checkpoint

        print("Sampled hyperparameters:")
        for k, v in sampled_kwargs.items():
            print(f"  {k}: {v}")

        run_name = f"rs_{task}_trial_{trial + 1}_{int(time.time())}"
        args = task_spec.args_cls.create(**sampled_kwargs)

        try:
            print(f"Launching training run: {run_name}")
            task_spec.train_fn(
                args, task_spec.make_env, env_config, checkpoint_dir, run_name, wandb_enabled
            )
        except Exception:
            print(f"Trial {trial + 1} failed:")
            traceback.print_exc()
        finally:
            # train_fn never finishes the run itself (that's normally left to train.py's
            # post-training eval) -- without this, every trial after the first would keep
            # logging into the first trial's run, since wandb.init() only fires when
            # wandb.run is None.
            if wandb_enabled and wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
