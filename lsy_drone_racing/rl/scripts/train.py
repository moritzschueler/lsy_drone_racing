"""CLI entry point for training/evaluating a PPO agent on a drone RL task.

Examples:
    python -m lsy_drone_racing.rl.scripts.train --task racing
    python -m lsy_drone_racing.rl.scripts.train --task hover --train False --num_eval_iterations 3
    python -m lsy_drone_racing.rl.scripts.train --task racing --num_envs 2048 --learning_rate 1e-3
"""

from pathlib import Path
from typing import Any

import fire
import numpy as np

import wandb
from lsy_drone_racing.rl.tasks import get_task
from lsy_drone_racing.utils import load_config

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"


def _latest_checkpoint(task: str) -> Path | None:
    """Return the most recently modified checkpoint for a task, or None if none exist."""
    candidates = list((CHECKPOINT_DIR / task).glob("*.ckpt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main(
    task: str = "multi_agent_racing",
    config: str = "rl_multi_level2_norand.toml",
    wandb_enabled: bool = True,
    num_eval_iterations: int = 30,
    **kwargs: Any,
):
    """Train and/or evaluate a single PPO agent on the given task.

    Args:
        task: Task name (one of the keys in ``lsy_drone_racing.rl.tasks.TASKS``).
        config: Env config file name under ``config/`` (e.g. ``level0.toml``).
        wandb_enabled: Whether to log to Weights & Biases.
        train: Whether to run training. If False, only evaluation runs.
        num_eval_iterations: Number of (headless) evaluation episodes to run after training.
            Rendering is handled separately by ``rl/scripts/render.py``.
        **kwargs: Any ``Args`` field override (e.g. num_envs=2048).
    """
    task_spec = get_task(task)
    args = task_spec.args_cls.create(**kwargs)

    checkpoint_dir = CHECKPOINT_DIR / task
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_config = load_config(Path(__file__).parents[3] / "config" / config)

    model_path = task_spec.train_fn(
        args, task_spec.make_env, env_config, checkpoint_dir, task, wandb_enabled
    )

    if num_eval_iterations > 0:
        eval_path = model_path or _latest_checkpoint(task)
        if eval_path is None:
            raise FileNotFoundError(
                f"Can't evaluate because there is no checkpoint for task {task} in "
                f"{checkpoint_dir} and training is disabled."
            )
        episode_rewards, episode_lengths = task_spec.eval_fn(
            args, task_spec.make_env, env_config, num_eval_iterations, eval_path
        )

        for i in range(num_eval_iterations):
            print(
                f"Episode {i + 1}: Reward: {episode_rewards[i]:.2f}, "
                f"Length: {int(episode_lengths[i])}"
            )
        print(
            f"Average Episode Reward: {np.mean(episode_rewards):.2f}, "
            f"Average Episode Length: {np.mean(episode_lengths):.1f}"
        )
        print(
            f"Episode Reward Std: {np.std(episode_rewards):.2f}, Episode Length Std: {np.std(episode_lengths):.1f}"
        )
        if wandb_enabled:
            wandb.log(
                {
                    "eval/rewards_mean": np.mean(episode_rewards),
                    "eval/steps_mean": np.mean(episode_lengths),
                    "eval/rewards_std": np.std(episode_rewards),
                    "eval/steps_std": np.std(episode_lengths),
                }
            )
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
