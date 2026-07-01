"""Functional (pytree) base env for the JAX racing rollout.

Adapts the gymnasium ``VecDroneRaceEnv`` -- a stateful, host-side class that is *not* a pytree --
into a frozen ``struct.PyTreeNode`` with the functional ``step(env, action) -> (env, payload)`` /
``reset(env, ...)`` signature the scanned rollout and the functional wrappers expect. The env state
(``data``) is the only pytree field, so an instance threads cleanly through ``lax.scan`` as the
carry.

The gym env is used only as a one-time host-side *factory*: its pure ``_step``/``_reset`` closures
(produced by ``RaceCoreEnv.build_step_fn``/``build_reset_fn``) and its initial ``EnvData`` are
captured, after which the gym object can be discarded -- the closures keep the underlying sim
alive. All racing logic (gate advance, crash detection, reward, NEXT_STEP autoreset via
``lax.cond``) already lives inside those closures and is reused, not reimplemented.

Single-agent: the drone axis of the base env's ``(n_envs, n_drones, ...)`` outputs is squeezed to
``[:, 0]`` to match the single-drone observation/action spaces, exactly mirroring
``VecDroneRaceEnv.step``/``reset``. (For multi-agent racing later, drop the squeeze and keep the
``n_drones`` axis.)
"""

from __future__ import annotations

from typing import Any, Callable

import flax.struct as struct
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import EnvData


@struct.dataclass
class RacingEnv(struct.PyTreeNode):
    """Functional pytree adapter over a gym ``VecDroneRaceEnv``'s pure step/reset core."""

    data: EnvData = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    single_observation_space: spaces.Space = struct.field(pytree_node=False)
    single_action_space: spaces.Space = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)

    @property
    def observation_space(self) -> spaces.Space:
        return batch_space(self.single_observation_space, self.num_envs)

    @property
    def action_space(self) -> spaces.Space:
        return batch_space(self.single_action_space, self.num_envs)

    @property
    def unwrapped(self) -> RacingEnv:
        return self

    @property
    def steps(self) -> Array:
        return self.data.steps

    def close(self) -> None:
        """No-op: the underlying sim is released when this env is garbage-collected."""

    def set_progress(self, tau: Array) -> RacingEnv:
        """Leaf no-op: ends the ``set_progress`` recursion (the base env has no curriculum)."""
        return self

    @classmethod
    def create(cls, base_env: VecDroneRaceEnv) -> RacingEnv:
        """Adapt a constructed gym ``VecDroneRaceEnv`` into a functional pytree env.

        ``base_env`` is used only as a one-time host-side factory: its pure ``_step``/``_reset``
        closures and initial ``data`` are captured, after which the gym object can be discarded.
        """
        raw_step = base_env._step
        raw_reset = base_env._reset

        def step(env: RacingEnv, action: Array) -> tuple[RacingEnv, tuple[Any, ...]]:
            data, (obs, reward, terminated, truncated, info) = raw_step(env.data, action)
            obs = {k: v[:, 0] for k, v in obs.items()}
            info = {k: v[:, 0] for k, v in info.items()}
            # Expose the current target gate (next gate to pass, -1 once finished) for metrics.
            info = {**info, "target_gate": data.target_gate[:, 0]}
            env = env.replace(data=data)
            return env, (obs, reward[:, 0], terminated[:, 0], truncated[:, 0], info)

        def reset(
            env: RacingEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[RacingEnv, tuple[Any, Any]]:
            data, (obs, info) = raw_reset(env.data, seed=seed)
            obs = {k: v[:, 0] for k, v in obs.items()}
            info = {k: v[:, 0] for k, v in info.items()}
            env = env.replace(data=data)
            return env, (obs, info)

        return cls(
            data=base_env.data,
            step=step,
            reset=reset,
            single_observation_space=base_env.single_observation_space,
            single_action_space=base_env.single_action_space,
            num_envs=base_env.num_envs,
        )
