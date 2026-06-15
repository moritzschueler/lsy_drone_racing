"""CLI entry point for training/evaluating a PPO agent on a drone RL task.

Examples:
    python -m lsy_drone_racing.rl.scripts.train --task racing
    python -m lsy_drone_racing.rl.scripts.train --task hover --train False --num_eval_iterations 3
    python -m lsy_drone_racing.rl.scripts.train --task racing --num_envs 2048 --learning_rate 1e-3
"""

import os
from pathlib import Path

import fire
import numpy as np
import wandb

from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.ppo import evaluate_ppo, train_ppo
from lsy_drone_racing.rl.tasks import get_task

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"


def _latest_checkpoint(task: str) -> Path | None:
    """Return the most recently modified checkpoint for a task, or None if none exist."""
    candidates = list((CHECKPOINT_DIR / task).glob("*.ckpt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main(
    task: str = "single_agent_racing",
    wandb_enabled: bool = True,
    train: bool = True,
    num_eval_iterations: int = 1,
    **kwargs,
):
    """Train and/or evaluate a PPO agent on the given task.

    Args:
        task: Task name (one of the keys in ``lsy_drone_racing.rl.tasks.TASKS``).
        wandb_enabled: Whether to log to Weights & Biases.
        train: Whether to run training. If False, only evaluation runs.
        num_eval_iterations: Number of evaluation episodes to run after training.
        **kwargs: Any ``Args`` field override (e.g. num_envs=2048).
    """
    os.environ.pop("WAYLAND_DISPLAY", None)
    task_spec = get_task(task)
    args = Args.create(**{**task_spec.defaults, **kwargs})

    # Each task gets its own checkpoint subfolder; runs are saved with a timestamp + reward.
    task_dir = CHECKPOINT_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    jax_device = args.jax_device

    model_path = None
    if train:
        model_path = train_ppo(args, task_spec.make_env, task_dir, task, jax_device, wandb_enabled)

    if num_eval_iterations > 0:
        eval_path = model_path or _latest_checkpoint(task)
        if eval_path is None:
            raise FileNotFoundError(f"Can't evaluate because there is no checkpoint for task {task} in {task_dir} and training is disabled.")
        episode_rewards, episode_lengths = evaluate_ppo(
            args, task_spec.make_env, num_eval_iterations, eval_path
        )
        if wandb_enabled and train:
            wandb.log(
                {
                    "eval/mean_rewards": np.mean(episode_rewards),
                    "eval/mean_steps": np.mean(episode_lengths),
                }
            )
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
