import pickle
from pathlib import Path
from typing import Any

import fire
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from lsy_drone_racing.rl.agents.ppo_agent import Agent
from lsy_drone_racing.rl.tasks import get_task
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper
from lsy_drone_racing.utils import load_config

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"


def _latest_checkpoint(task: str) -> Path | None:
    """Return the most recently modified checkpoint for a task, or None if none exist."""
    candidates = list((CHECKPOINT_DIR / task).glob("*.ckpt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main(
    task: str = "single_agent_racing",
    config: str = "level0.toml",
    checkpoint: str | None = None,
    **kwargs: Any,
):
    """Render the simulation for a single trained PPO agent on the given task.

    Args:
        task: Task name (one of the keys in ``lsy_drone_racing.rl.tasks.TASKS``).
        checkpoint: Path to a specific checkpoint to evaluate.
    """
    task_spec = get_task(task)
    # Render a single env: the viewer only ever shows world 0, so building the training default
    # (num_envs=1024) would step 1024 mjx envs eagerly per frame (crippling on a laptop / CPU) and
    # make ``done`` fire as soon as *any* of the 1024 ends -- aborting the shown drone mid-track.
    kwargs.setdefault("num_envs", 1)
    args = task_spec.args_cls.create(**kwargs)

    checkpoint_dir = CHECKPOINT_DIR / task

    if not checkpoint:
        latest_checkpoint = _latest_checkpoint(task)
        if not latest_checkpoint:
            raise FileNotFoundError(
                f"No checkpoints found for task '{task}' in {checkpoint_dir}, can't render. Aborting!"
            )
        else:
            print(f"No checkpoint specified, using latest checkpoint: {latest_checkpoint}")
            model_path = latest_checkpoint
    else:
        model_path = CHECKPOINT_DIR / task / checkpoint

    env_config = load_config(Path(__file__).parents[3] / "config" / config)

    eval_env: Wrapper = task_spec.make_env(args, env_config)
    action_dim = int(np.prod(eval_env.single_action_space.shape))
    obs_dim = int(np.prod(eval_env.single_observation_space.shape))

    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(0))
    with open(model_path, "rb") as f:
        nnx.update(agent, pickle.load(f))

    @nnx.jit
    def policy_step(agent: Agent, env: Wrapper, obs: Array) -> tuple[Wrapper, Array, Array, Array]:
        """One deterministic (mean-action) env step, compiled so the full wrapper stack fuses.

        ``render()`` stays outside jit -- it mutates the host-side MuJoCo sim/viewer -- but the
        physics + wrapper chain run as a single compiled kernel instead of eager op dispatch.
        """
        mean, _, _ = agent(obs)
        env, (obs, _, terminated, truncated, _) = env.step(env, mean)
        return env, obs, terminated, truncated

    eval_env, (obs, _) = eval_env.reset(eval_env, seed=args.seed)
    eval_env.render()
    done = False

    while not done:
        eval_env, obs, terminated, truncated = policy_step(agent, eval_env, jnp.asarray(obs))
        eval_env.render()
        done = bool(jnp.any(terminated | truncated))
    eval_env.close()


if __name__ == "__main__":
    fire.Fire(main)
