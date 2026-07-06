"""Task registry mapping a task name to its env factory and default hyperparameters.

A "task" bundles everything that differs between training problems: the environment, its
reward, the observation/wrapper stack, and default ``Args`` overrides. The PPO algorithm in
``lsy_drone_racing.rl.ppo`` is otherwise task-agnostic.
"""

from dataclasses import dataclass
from typing import Any, Callable

from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.ippo import evaluate_ippo, train_ippo
from lsy_drone_racing.rl.ppo import evaluate_ppo, train_ppo
from lsy_drone_racing.rl.tasks import hover, multi_agent_racing, single_agent_racing, trajectory
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper


@dataclass
class Task:
    """A trainable task: an env factory, its ``Args`` subclass, and its train/eval entry points.

    ``args_cls`` is an ``Args`` (sub)class whose field defaults are this task's hyperparameters;
    the CLI builds the run config with ``args_cls.create(**cli_overrides)``. Tasks without
    task-specific defaults point at the base ``Args`` directly.

    ``make_env`` is the functional env factory ``(args, config) -> Wrapper`` (a scannable
    ``struct.PyTreeNode``); see ``single_agent_racing.make_functional_env``.

    ``train_fn`` / ``eval_fn`` are the PPO train/evaluate entry points. They default to the
    single-agent trainer in ``rl.ppo``; multi-agent (self-play) tasks point them at the IPPO
    trainer in ``rl.ippo`` instead, which is why the single-agent pipeline in ``ppo.py`` is never
    touched by the multi-agent path.
    """

    make_env: Callable[[Args, Any], Wrapper]
    args_cls: type[Args]
    train_fn: Callable[..., Any] = train_ppo
    eval_fn: Callable[..., Any] = evaluate_ppo


TASKS: dict[str, Task] = {
    "single_agent_racing": Task(
        single_agent_racing.make_functional_env, single_agent_racing.RacingArgs
    ),
    "multi_agent_racing": Task(
        multi_agent_racing.make_multi_agent_env,
        multi_agent_racing.MultiAgentRacingArgs,
        train_fn=train_ippo,
        eval_fn=evaluate_ippo,
    ),
    "hover": Task(hover.make_env, Args),
    "random_trajectory_following": Task(trajectory.make_env, Args),
}


def get_task(name: str) -> Task:
    """Look up a task by name, raising a helpful error if unknown."""
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available tasks are {sorted(TASKS)}")
    return TASKS[name]
