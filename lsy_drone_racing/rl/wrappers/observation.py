"""Observation wrappers for the vectorized JAX drone environments."""

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

jp = jnp

# Number of upcoming gates (current target + the following one) exposed to the policy.
N_NEXT_GATES = 2


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


@jax.jit
def _relative_racing_obs(obs: dict) -> dict:
    """Recast a raw racing observation into the relative, next-2-gates representation."""
    pos = obs["pos"]  # (E, 3)
    n_envs = pos.shape[0]
    n_gates = obs["gates_pos"].shape[1]
    # Drone attitude as a flattened rotation matrix (body -> world).
    drone_rot = R.from_quat(obs["quat"]).as_matrix().reshape(n_envs, 9)
    # Indices of the next N_NEXT_GATES gates, clamped to the last gate (and to 0 once finished).
    base_idx = jp.maximum(obs["target_gate"], 0)  # target_gate is -1 when the track is done
    idx = jp.minimum(base_idx[:, None] + jp.arange(N_NEXT_GATES)[None, :], n_gates - 1)  # (E, k)
    env_idx = jp.arange(n_envs)[:, None]
    gates_pos = obs["gates_pos"][env_idx, idx]  # (E, k, 3)
    gates_quat = obs["gates_quat"][env_idx, idx]  # (E, k, 4)
    gates_visited = obs["gates_visited"][env_idx, idx]  # (E, k)
    gates_rel_pos = gates_pos - pos[:, None, :]
    gates_rot = R.from_quat(gates_quat.reshape(-1, 4)).as_matrix().reshape(n_envs, N_NEXT_GATES, 9)
    return {
        "ang_vel": obs["ang_vel"],
        "drone_rot": drone_rot,
        "gates_rel_pos": gates_rel_pos,
        "gates_rot": gates_rot,
        "gates_visited": gates_visited,
        "last_action": obs["last_action"],
        "obstacles_rel_pos": obs["obstacles_pos"] - pos[:, None, :],
        "obstacles_visited": obs["obstacles_visited"],
        "vel": obs["vel"],
    }


class RelativeRacingObs(VectorObservationWrapper):
    """Recast the observation into a relative, track-length-invariant racing representation.

    Compared to the raw observation this wrapper:

    * expresses gate and obstacle positions relative to the drone (``obj_pos - drone_pos``,
      in world axes) and drops the drone's absolute position,
    * keeps only the next ``N_NEXT_GATES`` gates (current target + the following one), so the
      observation size is independent of the track length,
    * represents the drone and gate orientations as flattened 3x3 rotation matrices instead of
      quaternions (no double-cover discontinuity, and the gate's +x traversal axis is the first
      column),
    * drops the ``target_gate`` index (implicit once only the upcoming gates are shown).

    Velocity and angular velocity are kept in world axes; the drone rotation matrix is provided
    so the policy can rotate into the body frame itself if useful.

    Requires ``last_action`` to already be present (i.e. wrap *after* ``ActionPenalty``).
    """

    def __init__(self, env: VectorEnv):
        """Init."""
        super().__init__(env)
        base = self.single_observation_space
        n_obstacles = base["obstacles_pos"].shape[0]
        spec = {
            "ang_vel": base["ang_vel"],
            "drone_rot": spaces.Box(-1.0, 1.0, shape=(9,)),
            "gates_rel_pos": spaces.Box(-np.inf, np.inf, shape=(N_NEXT_GATES, 3)),
            "gates_rot": spaces.Box(-1.0, 1.0, shape=(N_NEXT_GATES, 9)),
            "gates_visited": spaces.Box(0, 1, shape=(N_NEXT_GATES,), dtype=bool),
            "last_action": base["last_action"],
            "obstacles_rel_pos": spaces.Box(-np.inf, np.inf, shape=(n_obstacles, 3)),
            "obstacles_visited": base["obstacles_visited"],
            "vel": base["vel"],
        }
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def observations(self, observations: dict) -> dict:
        """Transform the raw observation dict into the relative racing representation."""
        return _relative_racing_obs(observations)
