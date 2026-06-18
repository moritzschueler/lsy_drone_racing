"""Task registry mapping a task name to its env factory and default hyperparameters.

A "task" bundles everything that differs between training problems: the environment, its
reward, the observation/wrapper stack, and default ``Args`` overrides. The PPO algorithm in
``lsy_drone_racing.rl.ppo`` is otherwise task-agnostic.
"""

from dataclasses import dataclass
from typing import Callable

from gymnasium.vector import VectorEnv

from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.tasks import hover, single_agent_racing, trajectory


@dataclass
class Task:
    """A trainable task: an env factory plus the ``Args`` subclass holding its defaults.

    ``args_cls`` is an ``Args`` (sub)class whose field defaults are this task's hyperparameters;
    the CLI builds the run config with ``args_cls.create(**cli_overrides)``. Tasks without
    task-specific defaults point at the base ``Args`` directly.
    """

    make_env: Callable[[Args, int, str], VectorEnv]
    args_cls: type[Args]


TASKS: dict[str, Task] = {
    "single_agent_racing": Task(single_agent_racing.make_env, single_agent_racing.RacingArgs),
    "hover": Task(hover.make_env, Args),
    "random_trajectory_following": Task(trajectory.make_env, Args),
}


def get_task(name: str) -> Task:
    """Look up a task by name, raising a helpful error if unknown."""
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {sorted(TASKS)}")
    return TASKS[name]
