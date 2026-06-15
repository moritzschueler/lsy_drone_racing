"""Observation wrappers for the vectorized JAX drone environments."""

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from gymnasium.vector.utils import batch_space
from jax import Array

jp = jnp


class StackObs(VectorObservationWrapper):
    """Wrapper to stack history observations."""

    def __init__(self, env: VectorEnv, n_obs: int = 0):
        """Init."""
        super().__init__(env)
        self.n_obs = n_obs
        if self.n_obs > 0:
            # Update observation space
            spec = {k: v for k, v in self.single_observation_space.items()}
            spec["prev_obs"] = spaces.Box(-np.inf, np.inf, shape=(13 * self.n_obs,))
            self.single_observation_space = spaces.Dict(spec)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            # Init obs buffer. VecDroneRaceEnv has no standalone obs() method, so seed the
            # history from the initial observation returned by reset().
            init_obs, _ = env.reset()
            self._prev_obs = jp.zeros((self.num_envs, self.n_obs, 13))
            for _ in range(n_obs):
                self._prev_obs = self._update_prev_obs(self._prev_obs, init_obs)

    def observations(self, observations: dict) -> dict:
        """Override observation."""
        if self.n_obs > 0:
            observations["prev_obs"] = self._prev_obs.reshape(self.num_envs, -1)
            self._prev_obs = self._update_prev_obs(self._prev_obs, observations)
        return observations

    @staticmethod
    @jax.jit
    def _update_prev_obs(prev_obs: Array, obs: dict) -> Array:
        """Update previous observations."""
        basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = jp.concatenate(
            [jp.reshape(obs[k], (obs[k].shape[0], -1)) for k in basic_obs_key], axis=-1
        )
        prev_obs = jp.concatenate([prev_obs[:, 1:, :], basic_obs[:, None, :]], axis=1)
        return prev_obs


class FlattenJaxObservation(VectorObservationWrapper):
    """Wrapper to flatten the dict observations into a single float32 vector.

    gym's ``flatten_space`` one-hot-encodes ``Discrete`` spaces (e.g. ``target_gate``),
    which would not match a plain concatenation of the raw observation values. To keep the
    declared space consistent with what ``observations`` actually produces, we build the
    flattened space ourselves: each entry contributes ``prod(shape)`` features (a
    ``Discrete`` contributes 1), and ``observations`` concatenates the same keys, cast to
    float32, in the same fixed order.
    """

    def __init__(self, env: VectorEnv):
        """Init."""
        super().__init__(env)
        self._keys = list(env.single_observation_space.keys())
        flat_dim = sum(self._flat_size(env.single_observation_space[k]) for k in self._keys)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(flat_dim,))
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    @staticmethod
    def _flat_size(space: spaces.Space) -> int:
        """Number of scalar features a sub-space contributes to the flat vector."""
        if isinstance(space, spaces.Discrete):
            return 1
        return int(np.prod(space.shape))

    def observations(self, observations: dict) -> Array:
        """Flatten observations into one float32 vector per environment."""
        return jp.concatenate(
            [
                jp.reshape(observations[k], (observations[k].shape[0], -1)).astype(jnp.float32)
                for k in self._keys
            ],
            axis=-1,
        )
