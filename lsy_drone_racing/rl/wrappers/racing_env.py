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
import numpy as np
from crazyflow.sim.visualize import draw_points
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.multi_drone_race import VecMultiDroneRaceEnv
from lsy_drone_racing.envs.race_core import EnvData

# Marker colors for MultiRacingEnv.render(): drone 0 (the trainable ego) vs drones 1.. (opponents,
# self-play or scripted-PID) so the two are visually distinguishable in the viewer.
_EGO_MARKER_RGBA = np.array([0.0, 1.0, 0.0, 1.0])  # green
_OPPONENT_MARKER_RGBA = np.array([1.0, 0.0, 0.0, 1.0])  # red


@struct.dataclass
class RacingEnv(struct.PyTreeNode):
    """Functional pytree adapter over a gym ``VecDroneRaceEnv``'s pure step/reset core."""

    data: EnvData = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    single_observation_space: spaces.Space = struct.field(pytree_node=False)
    single_action_space: spaces.Space = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)

    # The host-side gym env, kept only for its viewer/sim so the functional env can render.
    base_env: VecDroneRaceEnv | None = struct.field(pytree_node=False, default=None)

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

    def render(self) -> None:
        """Render the current functional state via the captured gym env's viewer.

        The pure rollout keeps its state in ``self.data``; the gym env renders from *its own*
        ``self.data`` (and owns the MuJoCo sim/viewer the functional step drops). Push the current
        functional ``data`` into the gym env, then delegate to its ``render()`` (which lazily syncs
        the drone/gate poses into MuJoCo). Requires an env built with a live ``base_env`` (the
        render factory keeps one; headless training envs pass ``base_env=None``).
        """
        if self.base_env is None:
            raise RuntimeError(
                "RacingEnv.render() needs a live gym base_env, but none was captured. Build the "
                "env with a rendering-capable factory (base_env kept) rather than a headless one."
            )
        self.base_env.data = self.data
        self.base_env.render()

    def close(self) -> None:
        """No-op: the underlying sim is released when this env is garbage-collected."""

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
            # Expose the current gate-order progress count (== gate_sequence length once finished)
            # for metrics.
            info = {**info, "n_gates_passed": data.n_gates_passed[:, 0]}
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
            base_env=base_env,
        )


@struct.dataclass
class MultiRacingEnv(struct.PyTreeNode):
    """Functional pytree adapter over a gym ``VecMultiDroneRaceEnv``'s pure step/reset core.

    The multi-agent counterpart of :class:`RacingEnv`. Identical in spirit -- it captures the pure
    ``_step``/``_reset`` closures of a ``VecMultiDroneRaceEnv`` and threads ``data: EnvData`` as the
    only pytree field -- but it keeps the ``n_drones`` axis instead of squeezing ``[:, 0]``: obs
    fields, reward, done flags and ``n_gates_passed`` all stay ``(n_envs, n_drones, ...)``. Drone 0 is
    the trainable ego; drones ``1..n_drones-1`` are opponents whose actions the rollout supplies.

    ``single_observation_space`` / ``single_action_space`` are the **per-drone** spaces (exactly what
    single-agent :class:`RacingEnv` exposes), *not* the drone-batched spaces the underlying gym env
    reports -- so the downstream functional wrappers reuse their single-agent space logic unchanged
    and only need to map their array transforms over the extra ``n_drones`` axis (via the
    ``n_drones`` argument on their ``create``).
    """

    data: EnvData = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    single_observation_space: spaces.Space = struct.field(pytree_node=False)
    single_action_space: spaces.Space = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)
    n_drones: int = struct.field(pytree_node=False)

    base_env: VecMultiDroneRaceEnv | None = struct.field(pytree_node=False, default=None)

    @property
    def observation_space(self) -> spaces.Space:
        return batch_space(self.single_observation_space, self.num_envs)

    @property
    def action_space(self) -> spaces.Space:
        return batch_space(self.single_action_space, self.num_envs)

    @property
    def unwrapped(self) -> MultiRacingEnv:
        return self

    @property
    def steps(self) -> Array:
        return self.data.steps

    def render(self) -> None:
        """Render the current functional state, with a color marker above each drone.

        The viewer only ever shows world 0. Drone 0 (the trainable ego) gets a green marker;
        drones 1.. (opponents, self-play or scripted-PID) get a red marker, so the two are
        distinguishable at a glance -- the drones themselves are otherwise identical.
        """
        if self.base_env is None:
            raise RuntimeError(
                "MultiRacingEnv.render() needs a live gym base_env, but none was captured."
            )
        self.base_env.data = self.data
        # Markers must be (re-)added before every sim.render() call -- they don't persist.
        marker_pos = np.asarray(self.data.sim_data.states.pos[0]) + np.array([0.0, 0.0, 0.12])
        draw_points(self.base_env.sim, marker_pos[:1], rgba=_EGO_MARKER_RGBA, size=0.04)
        if self.n_drones > 1:
            draw_points(self.base_env.sim, marker_pos[1:], rgba=_OPPONENT_MARKER_RGBA, size=0.04)
        self.base_env.render()

    def close(self) -> None:
        """No-op: the underlying sim is released when this env is garbage-collected."""

    @classmethod
    def create(
        cls,
        base_env: VecMultiDroneRaceEnv,
        *,
        single_observation_space: spaces.Space,
        single_action_space: spaces.Space,
    ) -> MultiRacingEnv:
        """Adapt a constructed gym ``VecMultiDroneRaceEnv`` into a functional multi-drone pytree env.

        Args:
            base_env: The gym multi-drone env, used only as a one-time host-side factory (its pure
                ``_step``/``_reset`` closures and initial ``data`` are captured).
            single_observation_space: The **per-drone** observation space (e.g.
                ``build_observation_space(n_gates, n_obstacles)``), not the drone-batched space the
                gym env reports.
            single_action_space: The **per-drone** action space (e.g. ``build_action_space(...)``).
        """
        raw_step = base_env._step
        raw_reset = base_env._reset
        n_drones = base_env.data.sim_data.core.n_drones

        def step(env: MultiRacingEnv, action: Array) -> tuple[MultiRacingEnv, tuple[Any, ...]]:
            # action: (n_envs, n_drones, act_dim). Keep the full drone axis on every output.
            data, (obs, reward, terminated, truncated, info) = raw_step(env.data, action)
            info = {**info, "n_gates_passed": data.n_gates_passed}  # (n_envs, n_drones)
            env = env.replace(data=data)
            return env, (obs, reward, terminated, truncated, info)

        def reset(
            env: MultiRacingEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[MultiRacingEnv, tuple[Any, Any]]:
            data, (obs, info) = raw_reset(env.data, seed=seed)
            env = env.replace(data=data)
            return env, (obs, info)

        return cls(
            data=base_env.data,
            step=step,
            reset=reset,
            single_observation_space=single_observation_space,
            single_action_space=single_action_space,
            num_envs=base_env.num_envs,
            n_drones=n_drones,
            base_env=base_env,
        )
