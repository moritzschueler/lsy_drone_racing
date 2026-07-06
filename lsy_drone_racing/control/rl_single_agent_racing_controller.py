"""Deployment controller that runs a trained single-agent racing PPO policy.

This wraps the policy produced by ``lsy_drone_racing.rl`` for the ``single_agent_racing`` task so it
can be driven by ``scripts/sim.py`` / ``scripts/deploy.py``. It reproduces, at inference time, the
exact wrapper stack the policy was trained with (see
``lsy_drone_racing.rl.tasks.single_agent_racing.make_functional_env``):

* observation: the raw per-drone observation is recast into the body-frame, next-``N_NEXT_GATES``
  representation by reusing :func:`_relative_racing_obs` (the same jitted function training uses), then
  flattened in the same key order ``FlattenJaxObservation`` uses -- so the vector fed to the network
  is byte-for-byte what training produced, with no duplicated geometry math here;
* action: the network's deterministic mean (already in ``[-1, 1]`` via ``tanh``) has its yaw channel
  zeroed (``ZeroYaw``) and is then denormalized to the simulator's attitude range (``NormalizeActions``)
  using the env's own action-space bounds. The observed ``last_action`` is the *denormalized,
  yaw-zeroed* action initialized to zeros -- matching that ``ActionPenalty`` sits *inside*
  ``NormalizeActions`` in the training stack.

Checkpoint selection: set :data:`CHECKPOINT_OVERRIDE` to a specific ``.ckpt`` path to force it;
otherwise the most recent checkpoint in ``rl/checkpoints/single_agent_racing/`` is used, "most recent"
meaning the largest ``YYYYMMDD-HHMMSS`` timestamp embedded in the filename.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING    

import jax.numpy as jnp
import numpy as np
from flax import nnx

from lsy_drone_racing.control import Controller
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.rl.agents.ppo_agent import Agent
from lsy_drone_racing.rl.wrappers.observation import N_NEXT_GATES, _relative_racing_obs

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------------------------
# Set this to a specific checkpoint to force it (absolute, or relative to the single_agent_racing
# checkpoints folder). Leave it empty ("") to auto-load the most recent checkpoint in that folder.
CHECKPOINT_OVERRIDE: str = ""
# ---------------------------------------------------------------------------------------------

_ACTION_DIM = 4
_CHECKPOINT_DIR = Path(__file__).parents[1] / "rl" / "checkpoints" / "single_agent_racing"
# The YYYYMMDD-HHMMSS stamp training embeds in each checkpoint filename (see rl/ppo.py).
_TIMESTAMP_RE = re.compile(r"\d{8}-\d{6}")

# Keys the relative-racing observation reads from the raw per-drone observation, plus the ones this
# controller supplies itself (``last_action``). Mirrors what ActionPenalty + RelativeRacingObs need.
_RAW_OBS_KEYS = (
    "pos",
    "quat",
    "vel",
    "gates_pos",
    "gates_quat",
    "gates_visited",
    "obstacles_pos",
    "obstacles_visited",
)


def _resolve_checkpoint() -> Path:
    """Return the checkpoint to load: the override if set, else the newest by filename timestamp."""
    if CHECKPOINT_OVERRIDE:
        path = Path(CHECKPOINT_OVERRIDE)
        if not path.is_absolute():
            path = _CHECKPOINT_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"CHECKPOINT_OVERRIDE points at a missing file: {path}")
        return path

    candidates = list(_CHECKPOINT_DIR.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in {_CHECKPOINT_DIR}")
    stamped = [(m.group(0), p) for p in candidates if (m := _TIMESTAMP_RE.search(p.name))]
    if not stamped:
        raise FileNotFoundError(
            f"No checkpoint in {_CHECKPOINT_DIR} has a YYYYMMDD-HHMMSS timestamp in its name; "
            "set CHECKPOINT_OVERRIDE to pick one explicitly."
        )
    # Return most recent checkpoint from the checkpoints folder
    return max(stamped, key=lambda item: item[0])[1]


class RLSingleAgentRacingController(Controller):
    """Runs a trained single-agent racing PPO policy under the attitude control interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load the policy and prepare the observation/action transforms.

        Args:
            obs: Initial observation from ``env.reset``.
            info: Additional reset information (unused).
            config: Race configuration; provides the control mode and drone model used to build the
                attitude action-space bounds.
        """
        super().__init__(obs, info, config)

        # Attitude action-space bounds -> the exact [-1, 1] <-> sim-range rescaling NormalizeActions
        # applies. Order is [roll, pitch, yaw, thrust].
        action_space = build_action_space(config.env.control_mode, config.sim.drone_model)
        low = jnp.asarray(action_space.low, dtype=jnp.float32)
        high = jnp.asarray(action_space.high, dtype=jnp.float32)
        self._act_scale = (high - low) / 2.0
        self._act_offset = (high + low) / 2.0

        # Observed last_action is the previous *denormalized, yaw-zeroed* action; zeros at the start
        # (ActionPenalty initializes it to zeros before any step).
        self._last_action = jnp.zeros(_ACTION_DIM, dtype=jnp.float32)

        # Observation size: grav_body(3) + gates_corners(N*4*3) + gates_visited(N) + last_action(4)
        # + obstacles_rel_pos(O*3) + obstacles_visited(O) + vel(3). O read from the obs (works
        # regardless of any leading drone/env dim).
        n_obstacles = int(np.asarray(obs["obstacles_pos"]).shape[-2])
        obs_dim = 3 + N_NEXT_GATES * 4 * 3 + N_NEXT_GATES + _ACTION_DIM + n_obstacles * 3 + n_obstacles + 3

        # Build the network and load the trained parameters (nnx owns its weights; update in place).
        self._agent = Agent(obs_dim, _ACTION_DIM, rngs=nnx.Rngs(0))
        checkpoint_path = _resolve_checkpoint()
        with open(checkpoint_path, "rb") as f:
            nnx.update(self._agent, pickle.load(f))
        self._checkpoint_path = checkpoint_path

        print(f"RL Single Agent Racing controller uses checkpoint: {checkpoint_path.name}")

        # JIT-compile deterministic inference once so the first control call has no compile latency.
        self._infer = nnx.jit(self._deterministic_action)
        self._infer(self._agent, jnp.zeros((1, obs_dim), dtype=jnp.float32))

        self._finished = False

    @staticmethod
    def _deterministic_action(agent: Agent, obs: jnp.ndarray) -> jnp.ndarray:
        """Return the policy mean action (deterministic policy)."""
        mean, _, _ = agent(obs)
        return mean

    def _obs_vector(self, obs: dict[str, NDArray[np.floating]]) -> jnp.ndarray:
        """Build the flat policy observation from a raw per-drone observation.

        Reuses the training-time :func:`_relative_racing_obs` (adding a singleton env axis) and
        flattens its outputs in insertion order -- the same order ``FlattenJaxObservation`` uses.
        """
        raw = {k: jnp.asarray(obs[k])[None] for k in _RAW_OBS_KEYS}
        raw["target_gate"] = jnp.atleast_1d(jnp.asarray(obs["target_gate"]))
        raw["last_action"] = self._last_action[None]
        rel = _relative_racing_obs(raw)  # dict of (1, ...) arrays, keys in the flatten order
        return jnp.concatenate(
            [v.reshape(1, -1).astype(jnp.float32) for v in rel.values()], axis=-1
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the attitude command [roll, pitch, yaw, thrust] for the current observation."""
        obs_vec = self._obs_vector(obs)
        action = self._infer(self._agent, obs_vec)[0]  # (4,) in [-1, 1]
        action = action.at[2].set(0.0)  # ZeroYaw
        action = jnp.clip(action, -1.0, 1.0) * self._act_scale + self._act_offset  # NormalizeActions
        self._last_action = action  # feed the denormalized, yaw-zeroed action to the next obs
        return np.asarray(action, dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Signal completion once the whole track has been passed (target_gate == -1).

        Returns:
            True if the track is finished, False otherwise.
        """
        target_gate = np.asarray(obs["target_gate"]).reshape(-1)[0]
        self._finished = bool(target_gate == -1)
        return self._finished

    def episode_callback(self):
        """Reset per-episode state (called after each episode)."""
        self.episode_reset()

    def episode_reset(self):
        """Reset the tracked last action and finished flag."""
        self._last_action = jnp.zeros(_ACTION_DIM, dtype=jnp.float32)
        self._finished = False
