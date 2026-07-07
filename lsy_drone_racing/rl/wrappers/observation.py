"""Observation wrappers for the vectorized JAX drone environments."""

from __future__ import annotations

from typing import Any, Callable

import flax.struct as struct
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper

# Number of upcoming gates (current target + the following one) exposed to the policy.
N_NEXT_GATES = 2

# Half-extent (m) of the square gate opening. Each gate is observed via its four opening corners,
# which sit at gate-local (0, ±h, ±h): x=0 is the gate plane and the +x column of the gate rotation
# is the traversal normal, so y/z span the opening. Mirrors GATE_HALF_EXTENT in
# tasks/single_agent_racing.py (duplicated here to avoid a circular import); keep both in sync with
# the env's gate_size (0.45 -> 0.225).
GATE_OPENING_HALF_EXTENT = 0.225
_h = GATE_OPENING_HALF_EXTENT
_GATE_CORNERS_LOCAL = jnp.array(
    [[0.0, _h, _h], [0.0, _h, -_h], [0.0, -_h, _h], [0.0, -_h, -_h]]
)  # (4, 3) gate-frame corner offsets


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
            self._prev_obs = jnp.zeros((self.num_envs, self.n_obs, 13))
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
        basic_obs = jnp.concatenate(
            [jnp.reshape(obs[k], (obs[k].shape[0], -1)) for k in basic_obs_key], axis=-1
        )
        prev_obs = jnp.concatenate([prev_obs[:, 1:, :], basic_obs[:, None, :]], axis=1)
        return prev_obs


def _flat_size(space: spaces.Space) -> int:
    """Number of scalar features a sub-space contributes to the flat vector."""
    if isinstance(space, spaces.Discrete):
        return 1
    return int(np.prod(space.shape))


@struct.dataclass
class FlattenJaxObservation(Wrapper):
    """Flatten the dict observation into a single float32 vector per environment.

    gym's ``flatten_space`` one-hot-encodes ``Discrete`` spaces (e.g. ``target_gate``), which would
    not match a plain concatenation of the raw observation values. We build the flattened space
    ourselves: each entry contributes ``prod(shape)`` features (a ``Discrete`` contributes 1), and
    ``step``/``reset`` concatenate the same keys, cast to float32, in the same fixed order.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        space = self.base.single_observation_space
        flat_dim = sum(_flat_size(space[k]) for k in space.keys())
        return spaces.Box(-np.inf, np.inf, shape=(flat_dim,))

    @property
    def observation_space(self) -> spaces.Space:
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode, n_drones: int = 1) -> FlattenJaxObservation:
        """Create a FlattenJaxObservation wrapper around the base environment.

        ``n_drones > 1`` (multi-agent racing) preserves the per-drone axis: each field is flattened
        to ``(n_envs, n_drones, features)`` and concatenated on the last axis, yielding a
        ``(n_envs, n_drones, obs_dim)`` observation. The per-drone flat space is unchanged.
        """
        keys = list(base.single_observation_space.keys())
        # Number of leading axes to preserve before flattening the per-field features: (n_envs,) for
        # single-agent, (n_envs, n_drones) for multi-agent.
        lead = 2 if n_drones > 1 else 1

        def flatten(obs: dict) -> Array:
            return jnp.concatenate(
                [
                    jnp.reshape(obs[k], obs[k].shape[:lead] + (-1,)).astype(jnp.float32)
                    for k in keys
                ],
                axis=-1,
            )

        def reset(
            env: FlattenJaxObservation, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[FlattenJaxObservation, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            return env.replace(base=base_env), (flatten(obs), info)

        def step(
            env: FlattenJaxObservation, action: Array
        ) -> tuple[FlattenJaxObservation, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            return env.replace(base=base_env), (flatten(obs), reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


@jax.jit
def _relative_racing_obs(obs: dict) -> dict:
    """Recast a raw racing observation into the fully body-frame, next-2-gates representation."""
    pos = obs["pos"]  # (E, 3)
    n_envs = pos.shape[0]
    n_gates = obs["gates_pos"].shape[1]
    # Drone attitude, body -> world. Its transpose (world -> body) rotates world quantities into
    # the drone frame.
    rot_bw = R.from_quat(obs["quat"]).as_matrix()  # (E, 3, 3)

    def to_body_vec(v_world: Array) -> Array:  # (E, ..., 3) world vectors -> drone frame
        return jnp.einsum("eji,e...j->e...i", rot_bw, v_world)

    # Gravity direction in the body frame: the only world reference a relative observation needs
    # (tells the policy which way is "down" for thrust/attitude). Equals -rot_bw[:, 2, :].
    grav_body = to_body_vec(jnp.array([0.0, 0.0, -1.0]) + jnp.zeros_like(pos))
    # Indices of the next N_NEXT_GATES gates, clamped to the last gate (and to 0 once finished).
    base_idx = jnp.maximum(obs["target_gate"], 0)  # target_gate is -1 when the track is done
    idx = jnp.minimum(base_idx[:, None] + jnp.arange(N_NEXT_GATES)[None, :], n_gates - 1)  # (E, k)
    env_idx = jnp.arange(n_envs)[:, None]
    gates_pos = obs["gates_pos"][env_idx, idx]  # (E, k, 3)
    gates_quat = obs["gates_quat"][env_idx, idx]  # (E, k, 4)
    gates_visited = obs["gates_visited"][env_idx, idx]  # (E, k)
    # Each gate is observed via the drone-frame positions of its four opening corners. Place the
    # gate-local corner offsets (0, ±h, ±h) into the world (gate frame -> world via the gate
    # rotation, then translate by the gate centre), offset from the drone, and rotate into the drone
    # body frame. The corners jointly encode the gate centre, orientation and scale, so they replace
    # the old centre + relative-rotation-matrix pair.
    gates_rot_bw = (
        R.from_quat(gates_quat.reshape(-1, 4)).as_matrix().reshape(n_envs, N_NEXT_GATES, 3, 3)
    )
    corners_world = gates_pos[:, :, None, :] + jnp.einsum(
        "ekij,cj->ekci", gates_rot_bw, _GATE_CORNERS_LOCAL
    )  # (E, k, 4, 3)
    gates_corners = to_body_vec(corners_world - pos[:, None, None, :])
    return {
        "grav_body": grav_body,
        "gates_corners": gates_corners,
        "gates_visited": gates_visited,
        "last_action": obs["last_action"],
        "obstacles_rel_pos": to_body_vec(obs["obstacles_pos"] - pos[:, None, :]),
        "obstacles_visited": obs["obstacles_visited"],
        "vel": to_body_vec(obs["vel"]),
    }


@struct.dataclass
class RelativeRacingObs(Wrapper):
    """Recast the observation into a relative, track-length-invariant racing representation.

    The observation is fully expressed in the drone's body frame. Compared to the raw observation
    this wrapper:

    * expresses obstacle positions and velocity in the drone body frame, and drops the drone's
      absolute position,
    * drops the drone's angular velocity (body rates): under attitude control the 500 Hz onboard
      controller closes the rate loop, so the 50 Hz policy relies on the attitude (``grav_body``)
      instead of the noisy rate derivative,
    * keeps only the next ``N_NEXT_GATES`` gates (current target + the following one), so the
      observation size is independent of the track length,
    * represents each upcoming gate by the drone-frame positions of its four opening corners
      (``(N_NEXT_GATES, 4, 3)``) instead of a gate-centre + relative-rotation-matrix pair; the four
      corners jointly encode the gate centre, orientation and scale with no quaternion double-cover
      discontinuity,
    * drops the ``target_gate`` index (implicit once only the upcoming gates are shown),
    * replaces the full world-frame drone attitude with ``grav_body``, the gravity direction in
      the body frame -- the only world reference a body-frame observation still needs (which way
      is "down" for thrust/attitude). Yaw-about-vertical is intentionally not observable.

    Requires ``last_action`` to already be present (i.e. wrap *after* the wrapper that adds it:
    ``ActionSmoothnessPenalty`` for racing, ``ActionPenalty`` for hover / trajectory).
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        base = self.base.single_observation_space
        n_obstacles = base["obstacles_pos"].shape[0]
        spec = {
            "grav_body": spaces.Box(-1.0, 1.0, shape=(3,)),
            "gates_corners": spaces.Box(-np.inf, np.inf, shape=(N_NEXT_GATES, 4, 3)),
            "gates_visited": spaces.Box(0, 1, shape=(N_NEXT_GATES,), dtype=bool),
            "last_action": base["last_action"],
            "obstacles_rel_pos": spaces.Box(-np.inf, np.inf, shape=(n_obstacles, 3)),
            "obstacles_visited": base["obstacles_visited"],
            "vel": base["vel"],
        }
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode, n_drones: int = 1) -> RelativeRacingObs:
        """Create a RelativeRacingObs wrapper around the base environment.

        Requires ``last_action`` to already be present in the observation, i.e. wrap *after*
        ``ActionPenalty``.

        ``n_drones > 1`` (multi-agent racing) applies the single-drone body-frame transform to each
        drone by mapping :func:`_relative_racing_obs` over the ``n_drones`` axis (axis 1) of every
        observation field, so the transform code is reused unchanged and the output keeps its
        ``(n_envs, n_drones, ...)`` shape. The per-drone observation *space* is identical to the
        single-agent case.
        """
        transform = (
            jax.vmap(_relative_racing_obs, in_axes=1, out_axes=1)
            if n_drones > 1
            else _relative_racing_obs
        )

        def reset(
            env: RelativeRacingObs, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[RelativeRacingObs, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            return env.replace(base=base_env), (transform(obs), info)

        def step(
            env: RelativeRacingObs, action: Array
        ) -> tuple[RelativeRacingObs, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            return env.replace(base=base_env), (transform(obs), reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)
