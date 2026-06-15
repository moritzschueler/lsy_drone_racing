"""Task registry mapping a task name to its env factory and default hyperparameters.

A "task" bundles everything that differs between training problems: the environment, its
reward, the observation/wrapper stack, and default ``Args`` overrides. The PPO algorithm in
``lsy_drone_racing.rl.ppo`` is otherwise task-agnostic.
"""

from dataclasses import dataclass
from typing import Any, Callable

from gymnasium.vector import VectorEnv

from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.tasks import hover, racing, trajectory


@dataclass
class Task:
    """A trainable task: an env factory plus default Args overrides."""

    make_env: Callable[[Args, int, str], VectorEnv]
    defaults: dict[str, Any]


TASKS: dict[str, Task] = {
    "single_agent_racing": Task(racing.make_env, racing.DEFAULTS),
    "hover": Task(hover.make_env, hover.DEFAULTS),
    "random_trajectory_following": Task(trajectory.make_env, trajectory.DEFAULTS),
}


def get_task(name: str) -> Task:
    """Look up a task by name, raising a helpful error if unknown."""
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {sorted(TASKS)}")
    return TASKS[name]
