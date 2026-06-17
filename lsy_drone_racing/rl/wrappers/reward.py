"""Reward / action-shaping wrappers for the vectorized JAX drone environments."""

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import (
    VectorActionWrapper,
    VectorEnv,
    VectorObservationWrapper,
    VectorRewardWrapper,
)
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

jp = jnp


class AngleReward(VectorRewardWrapper):
    """Wrapper to penalize orientation in the reward."""

    def __init__(self, env: VectorEnv, rpy_coef: float = 0.08):
        """Init."""
        super().__init__(env)
        self.rpy_coef = rpy_coef

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Step and add the orientation penalty to the reward."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        rpy_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
        rpy_term = -self.rpy_coef * rpy_norm
        # Surface the term for per-component reward logging (summed per iteration in PPO).
        infos = {**infos, "rew/rpy": rpy_term}
        return observations, rewards + rpy_term, terminations, truncations, infos

    def rewards(self, rewards: Array, observations: dict[str, Array]) -> Array:
        """Additional angular rewards."""
        # apply rpy penalty
        rpy_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
        rewards -= self.rpy_coef * rpy_norm
        return rewards


class ActionPenalty(VectorObservationWrapper):
    """Wrapper to apply action penalty."""

    def __init__(
        self,
        env: VectorEnv,
        act_coef: float = 0.01,
        d_act_th_coef: float = 0.2,
        d_act_xy_coef: float = 0.4,
    ):
        """Init."""
        super().__init__(env)
        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["last_action"] = spaces.Box(-np.inf, np.inf, shape=(4,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self._last_action = jp.zeros((self.num_envs, 4))
        self.act_coef = act_coef
        self.d_act_th_coef = d_act_th_coef
        self.d_act_xy_coef = d_act_xy_coef

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Override step."""
        obs, reward, terminated, truncated, info = super().step(action)
        action_diff = action - self._last_action
        act_term = -self.act_coef * action[..., -1] ** 2  # energy
        d_act_th_term = -self.d_act_th_coef * action_diff[..., -1] ** 2  # thrust smoothness
        d_act_xy_term = -self.d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)  # rp smoothness
        reward = reward + act_term + d_act_th_term + d_act_xy_term
        self._last_action = action
        # Surface the terms for per-component reward logging (summed per iteration in PPO).
        info = {**info, "rew/act": act_term, "rew/d_act_th": d_act_th_term, "rew/d_act_xy": d_act_xy_term}
        # Diagnostics (normalized action: thrust 0 == hover, +1 == max): mean commanded thrust and
        # lean magnitude. Distinguishes "climbs because it over-thrusts and never tilts" (act_thrust>0,
        # act_tilt~0) from a thrust deficit. Logged as diagnostics/* (mean-per-step) by PPO.
        act_tilt = jp.sqrt(action[..., 0] ** 2 + action[..., 1] ** 2)  # roll/pitch lean
        info = {**info, "diagnostics/act_thrust": action[..., -1], "diagnostics/act_tilt": act_tilt}
        return self.observations(obs), reward, terminated, truncated, info

    def observations(self, observations: dict) -> dict:
        """Override observation."""
        observations["last_action"] = self._last_action
        return observations


class ZeroYaw(VectorActionWrapper):
    """Force the yaw command to zero before it reaches the environment.

    A quadrotor reaches every gate regardless of heading, and the drone/physics used here are
    yaw-symmetric, so yaw is a redundant degree of freedom for this task. Zeroing it at the
    outermost action boundary (i.e. outside ``ActionPenalty``) keeps it out of the
    action-smoothness penalty and out of the observed ``last_action``, which would otherwise
    report and penalize a yaw command that is never actually applied.
    """

    def actions(self, actions: Array) -> Array:
        """Zero the yaw channel of the (normalized) action."""
        return actions.at[..., 2].set(0.0)
