"""Reward / action-shaping wrappers for the vectorized JAX drone environments."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import flax.struct as struct
from gymnasium import spaces
from gymnasium.vector import (
    VectorActionWrapper,
    VectorEnv,
    VectorObservationWrapper,
    VectorRewardWrapper,
)
from typing import Any, Callable
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper

class AngleReward(VectorRewardWrapper):
    """Wrapper to penalize orientation in the reward."""

    def __init__(self, env: VectorEnv, rpy_coef: float = 0.08):
        """Init."""
        super().__init__(env)
        self.rpy_coef = rpy_coef

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Step and add the orientation penalty to the reward."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        rpy_norm = jnp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
        rpy_term = -self.rpy_coef * rpy_norm
        # Surface the term for per-component reward logging (summed per iteration in PPO).
        infos = {**infos, "rew/rpy": rpy_term}
        return observations, rewards + rpy_term, terminations, truncations, infos

    def rewards(self, rewards: Array, observations: dict[str, Array]) -> Array:
        """Additional angular rewards."""
        # apply rpy penalty
        rpy_norm = jnp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
        rewards -= self.rpy_coef * rpy_norm
        return rewards

@struct.dataclass
class ActionPenalty(Wrapper):
    """Energy + thrust/attitude-smoothness action penalty, with ``last_action`` in the observation."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    last_action: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        spec = {**self.base.single_observation_space.spaces}
        spec["last_action"] = spaces.Box(-np.inf, np.inf, shape=(4,))
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode,
        act_coef: float,
        d_act_th_coef: float,
        d_act_xy_coef: float,
        n_drones: int = 1,
    ) -> ActionPenalty:
        """Create an ActionPenalty wrapper around the base environment.

        ``n_drones > 1`` (multi-agent racing) keeps the per-drone action axis: ``last_action`` is
        shaped ``(num_envs, n_drones, 4)`` and the surfaced diagnostic/reward-component ``info``
        entries are sliced to the ego drone (index 0), matching the single-agent ``(num_envs,)``
        shape PPO logs. The reward penalty itself stays per-drone (``(num_envs, n_drones)``); only
        the ego drone's reward is trained on downstream.
        """
        multi = n_drones > 1
        act_shape = (base.num_envs, n_drones, 4) if multi else (base.num_envs, 4)
        last_action = jnp.zeros(act_shape)

        def reset(
            env: ActionPenalty, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[ActionPenalty, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env, last_action=jnp.zeros_like(env.last_action))
            obs = {**obs, "last_action": env.last_action}
            return env, (obs, info)

        def step(
            env: ActionPenalty, action: Array
        ) -> tuple[ActionPenalty, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            action_diff = action - env.last_action
            act_term = -act_coef * action[..., -1] ** 2  # energy
            d_act_th_term = -d_act_th_coef * action_diff[..., -1] ** 2  # thrust smoothness
            # roll/pitch smoothness
            d_act_xy_term = -d_act_xy_coef * jnp.sum(action_diff[..., :3] ** 2, axis=-1)
            reward = reward + act_term + d_act_th_term + d_act_xy_term
            env = env.replace(base=base_env, last_action=action)
            obs = {**obs, "last_action": action}
            act_tilt = jnp.sqrt(action[..., 0] ** 2 + action[..., 1] ** 2)  # roll/pitch lean
            # Slice info terms to the ego drone (0) in multi-agent, so logged shapes stay (num_envs,).
            ego = (lambda x: x[:, 0]) if multi else (lambda x: x)
            info = {
                **info,
                "rew/act": ego(act_term),
                "rew/d_act_th": ego(d_act_th_term),
                "rew/d_act_xy": ego(d_act_xy_term),
                "diagnostics/act_thrust": ego(action[..., -1]),
                "diagnostics/act_tilt": ego(act_tilt),
            }
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, last_action=last_action, step=step, reset=reset)


@struct.dataclass
class ZeroYaw(Wrapper):
    """Force the yaw command to zero before it reaches the environment.

    A quadrotor reaches every gate regardless of heading, and the drone/physics used here are
    yaw-symmetric, so yaw is a redundant degree of freedom for this task. Zeroing it at the
    outermost action boundary (i.e. outside ``ActionPenalty``) keeps it out of the
    action-smoothness penalty and out of the observed ``last_action``, which would otherwise
    report and penalize a yaw command that is never actually applied.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    reset: Callable = struct.field(pytree_node=False)
    step: Callable = struct.field(pytree_node=False)


    @classmethod
    def create(cls, base: struct.PyTreeNode) -> ZeroYaw:
        """ Create a ZeroYaw wrapper around the base environment. """

        def reset(env: ZeroYaw, *, seed: int | None = None, options: dict | None = None) -> tuple[ZeroYaw, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)
        
        def step(env: ZeroYaw, actions: Array) -> tuple[ZeroYaw, tuple[Any, ...]]:
            """Zero the yaw channel of the (normalized) action."""
            actions = actions.at[..., 2].set(0.0)
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, actions)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


@struct.dataclass
class NormalizeActions(Wrapper):
    """Expose actions in ``[-1, 1]`` and rescale them to the simulator's action range.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_action_space(self) -> spaces.Space:
        base_space = self.base.single_action_space
        return spaces.Box(-1.0, 1.0, shape=base_space.shape, dtype=base_space.dtype)

    @property
    def action_space(self) -> spaces.Space:
        return batch_space(self.single_action_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> NormalizeActions:
        """Create a NormalizeActions wrapper around the base environment."""
        low = jnp.asarray(base.single_action_space.low)
        high = jnp.asarray(base.single_action_space.high)
        scale = (high - low) / 2.0
        mean = (high + low) / 2.0

        def reset(
            env: NormalizeActions, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[NormalizeActions, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            return env.replace(base=base_env), (obs, info)

        def step(env: NormalizeActions, action: Array) -> tuple[NormalizeActions, tuple[Any, ...]]:
            action = jnp.clip(action, -1.0, 1.0) * scale + mean
            base_env, payload = env.base.step(env.base, action)
            return env.replace(base=base_env), payload

        return cls(base=base, step=step, reset=reset)

