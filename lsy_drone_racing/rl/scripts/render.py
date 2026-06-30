import pickle
import fire
import numpy as np
import jax.numpy as jnp

from pathlib import Path
from typing import Any
from flax import nnx

from lsy_drone_racing.rl.agents.ppo_agent import Agent
from lsy_drone_racing.rl.tasks import get_task

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"


def _latest_checkpoint(task: str) -> Path | None:
    """Return the most recently modified checkpoint for a task, or None if none exist."""
    candidates = list((CHECKPOINT_DIR / task).glob("*.ckpt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main(
    task: str = "single_agent_racing",
    checkpoint: str | None = None,
    **kwargs: Any
):
    """Render the simulation for a single trained PPO agent on the given task.

    Args:
        task: Task name (one of the keys in ``lsy_drone_racing.rl.tasks.TASKS``).
        checkpoint: Path to a specific checkpoint to evaluate.
    """
    task_spec = get_task(task)
    args = task_spec.args_cls.create(**kwargs)

    checkpoint_dir = CHECKPOINT_DIR / task

    if not checkpoint:
        latest_checkpoint = _latest_checkpoint(task)
        if not latest_checkpoint:
            raise FileNotFoundError(f"No checkpoints found for task '{task}' in {checkpoint_dir}, can't render. Aborting!")
        else:
            print(f"No checkpoint specified, using latest checkpoint: {latest_checkpoint}")
            model_path = latest_checkpoint
    else:
        model_path = CHECKPOINT_DIR / task / checkpoint

    eval_env = task_spec.make_env(args, 1, args.jax_device)
    action_dim = int(np.prod(eval_env.single_action_space.shape))
    obs_dim = int(np.prod(eval_env.single_observation_space.shape))

    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(0))
    with open(model_path, "rb") as f:
        nnx.update(agent, pickle.load(f))
    
    obs, _ = eval_env.reset(seed=args.seed)
    done = False
        
    while not done:
        obs_jax = jnp.asarray(obs)
        act = agent(obs_jax)[0]
        obs, _, terminated, truncated, _ = eval_env.step(act)
        eval_env.render()
        done = bool(jnp.any(jnp.asarray(terminated) | jnp.asarray(truncated)))
    eval_env.close() 

if __name__ == "__main__":
    fire.Fire(main)
