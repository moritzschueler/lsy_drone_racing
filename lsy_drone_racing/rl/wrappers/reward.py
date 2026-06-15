"""Reward / action-shaping wrappers for the vectorized JAX drone environments."""

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorRewardWrapper
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
        """Set yaw command to zero."""
        actions = actions.at[..., 2].set(0.0)  # block yaw output because we don't need it
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return observations, self.rewards(rewards, observations), terminations, truncations, infos

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
        # penalty on actions
        action_diff = action - self._last_action
        # energy
        reward -= self.act_coef * action[..., -1] ** 2
        # smoothness
        reward -= self.d_act_th_coef * action_diff[..., -1] ** 2
        reward -= self.d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)
        self._last_action = action
        return self.observations(obs), reward, terminated, truncated, info

    def observations(self, observations: dict) -> dict:
        """Override observation."""
        observations["last_action"] = self._last_action
        return observations
