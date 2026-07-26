r"""Deployment controller that races trained multi-agent (self-play) racing PPO checkpoints.

Meant to be loaded twice (once per drone) via ``scripts/multi_sim.py``'s ``--controllers``
comma-separated list, e.g.::

    pixi run python scripts/multi_sim.py --config multi_level0.toml \
        --controllers "rl_multi_agent_racing_controller.py,rl_multi_agent_racing_controller.py"

``multi_sim.py`` batches observations across drones and instantiates one controller per drone,
tagging each instance with ``info["rank"]``. This controller slices out its own drone's slot and
otherwise reproduces, at inference time, the exact per-drone wrapper stack the self-play policy
was trained with (see ``lsy_drone_racing.rl.tasks.multi_agent_racing.make_multi_agent_env``): the
per-drone observation/action transform is *identical* to the single-agent case (no
opponent-relative observations are wired yet), so this mirrors
``rl_single_agent_racing_controller.py`` with per-drone slicing and checkpoint selection.

Checkpoint selection: each rank loads a different snapshot, newest first, from
``rl/checkpoints/multi_agent_racing/`` (ranked by the ``YYYYMMDD-HHMMSS`` timestamp embedded in
the filename) -- rank 0 gets the most recent checkpoint, rank 1 the second most recent, etc. Set
``CHECKPOINT_OVERRIDES`` to force specific files per rank.
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
from lsy_drone_racing.rl.wrappers.observation import (
    N_NEXT_GATES,
    _opponent_body_frame,
    _relative_racing_obs,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------------------------
# Force specific checkpoints per drone rank (index 0 = drone 0, index 1 = drone 1, ...). Entries
# are absolute paths, or filenames resolved against the multi_agent_racing checkpoints folder.
# Leave an entry as "" (or the list short) to auto-pick the rank-th most recent checkpoint.
CHECKPOINT_OVERRIDES: list[str] = []
# ---------------------------------------------------------------------------------------------

_ACTION_DIM = 4
# Default for whether the policy expects the opponent obs slots; overridable per run via the config
# (``[controller] include_opponent_obs``) or the deploy/sim CLI. Must match the
# ``include_opponent_obs`` flag the checkpoint was trained with. Multi-agent racing defaults it ON,
# so the ego observes the nearest opponent's ego-relative, body-frame position and velocity (see
# _obs_vector / _opponent_body_frame). Set False only for a checkpoint trained without it. A
# mismatch vs. the checkpoint fails loudly at network load.
_INCLUDE_OPPONENT_OBS = True
# opponent_rel_pos(3) + opponent_rel_vel(3) appended when _INCLUDE_OPPONENT_OBS is set.
_OPPONENT_OBS_DIM = 6
_CHECKPOINT_DIR = Path(__file__).parents[1] / "rl" / "checkpoints" / "multi_agent_racing"
_TIMESTAMP_RE = re.compile(r"\d{8}-\d{6}")

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


def _ranked_checkpoints() -> list[Path]:
    """All checkpoints in the multi_agent_racing dir, newest first by embedded timestamp."""
    candidates = list(_CHECKPOINT_DIR.glob("*.ckpt"))
    stamped = [(m.group(0), p) for p in candidates if (m := _TIMESTAMP_RE.search(p.name))]
    if not stamped:
        raise FileNotFoundError(f"No timestamped checkpoints found in {_CHECKPOINT_DIR}")
    return [p for _, p in sorted(stamped, key=lambda item: item[0], reverse=True)]


def _resolve_checkpoint(rank: int) -> Path:
    """Return the checkpoint for ``rank``: the override if set, else the rank-th most recent."""
    if rank < len(CHECKPOINT_OVERRIDES) and CHECKPOINT_OVERRIDES[rank]:
        path = Path(CHECKPOINT_OVERRIDES[rank])
        if not path.is_absolute():
            path = _CHECKPOINT_DIR / path
        if not path.exists():
            raise FileNotFoundError(
                f"CHECKPOINT_OVERRIDES[{rank}] points at a missing file: {path}"
            )
        return path

    ranked = _ranked_checkpoints()
    if rank >= len(ranked):
        raise FileNotFoundError(
            f"Need a checkpoint for drone rank {rank}, but only {len(ranked)} exist in "
            f"{_CHECKPOINT_DIR}. Train more runs, or set CHECKPOINT_OVERRIDES."
        )
    return ranked[rank]


class RLMultiAgentRacingController(Controller):
    """Runs one drone's trained self-play racing PPO policy under the attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load this drone's policy and prepare the observation/action transforms.

        Args:
            obs: The initial (batched, one row per drone) observation from ``env.reset``.
            info: Additional reset information; must contain this instance's drone ``rank``.
            config: Race configuration; provides the control mode and drone model used to build
                the attitude action-space bounds.
        """
        super().__init__(obs, info, config)
        self.rank = int(info["rank"])
        obs0 = {k: np.asarray(v)[self.rank] for k, v in obs.items()}

        # Multi-drone configs (`multi_sim.py`) put control_mode under the per-drone
        # `env.kwargs[rank]` table rather than a single top-level `env.control_mode`.
        control_mode = config.env.kwargs[self.rank]["control_mode"]
        action_space = build_action_space(control_mode, config.sim.drone_model)
        low = jnp.asarray(action_space.low, dtype=jnp.float32)
        high = jnp.asarray(action_space.high, dtype=jnp.float32)
        self._act_scale = (high - low) / 2.0
        self._act_offset = (high + low) / 2.0
        self._last_action = jnp.zeros(_ACTION_DIM, dtype=jnp.float32)

        # Whether the policy was trained with the opponent obs slots. Read from this drone's
        # ``[[controller]]`` table (``config.controller`` is a list, one entry per drone, for
        # multi-drone configs -- unlike the single-agent controller's single ``[controller]``
        # table), settable via the deploy/sim CLI. Defaults to _INCLUDE_OPPONENT_OBS. Must match
        # the checkpoint or the network load below fails loudly.
        controller_cfg = config.controller[self.rank] if self.rank < len(config.controller) else {}
        self._include_opponent_obs = bool(
            controller_cfg.get("include_opponent_obs", _INCLUDE_OPPONENT_OBS)
        )

        n_obstacles = int(np.asarray(obs0["obstacles_pos"]).shape[-2])
        obs_dim = (
            3
            + N_NEXT_GATES * 4 * 3
            + N_NEXT_GATES
            + _ACTION_DIM
            + n_obstacles * 3
            + n_obstacles
            + 3
            + (_OPPONENT_OBS_DIM if self._include_opponent_obs else 0)
        )

        self._agent = Agent(obs_dim, _ACTION_DIM, rngs=nnx.Rngs(0))
        checkpoint_path = "/home/moritz_schueler/lsy_drone_racing/lsy_drone_racing/rl/checkpoints/multi_agent_racing/multi_agent_racing_20260725-190703_g3.66_step799014912.ckpt"
        with open(checkpoint_path, "rb") as f:
            nnx.update(self._agent, pickle.load(f))
        self._checkpoint_path = checkpoint_path

        print(
            f"Drone {self.rank}: RL multi-agent racing controller uses checkpoint: "
            f"{checkpoint_path}"
        )

        self._infer = nnx.jit(self._deterministic_action)
        self._infer(self._agent, jnp.zeros((1, obs_dim), dtype=jnp.float32))

        self._finished = False

    @staticmethod
    def _deterministic_action(agent: Agent, obs: jnp.ndarray) -> jnp.ndarray:
        """Return the policy mean action (deterministic policy)."""
        mean, _, _ = agent(obs)
        return mean

    def _obs_vector(self, obs: dict[str, NDArray[np.floating]]) -> jnp.ndarray:
        """Build this drone's flat policy observation from the batched multi-drone observation."""
        raw = {k: jnp.asarray(obs[k])[self.rank][None] for k in _RAW_OBS_KEYS}
        raw["n_gates_passed"] = jnp.atleast_1d(jnp.asarray(obs["n_gates_passed"])[self.rank])
        # gate_sequence is (n_drones, k); slice this drone's row and normalize to the (E=1, k)
        # shape _relative_racing_obs expects.
        gate_sequence = jnp.asarray(obs["gate_sequence"])[self.rank]
        raw["gate_sequence"] = gate_sequence.reshape((1, gate_sequence.shape[-1]))
        raw["last_action"] = self._last_action[None]
        rel = _relative_racing_obs(raw)  # dict of (1, ...) arrays
        if self._include_opponent_obs:
            # Reuse the training-time helper: build the full multi-drone pos/vel/quat with a singleton
            # env axis (E=1), then take this drone's row. This drone sees the nearest opponent's
            # ego-relative, body-frame position and velocity.
            batched = {k: jnp.asarray(obs[k])[None] for k in ("pos", "vel", "quat")}
            n_drones = batched["pos"].shape[1]
            opp_pos, opp_vel = _opponent_body_frame(batched, n_drones)  # each (1, n_drones, 3)
            rel = {
                **rel,
                "opponent_rel_pos": opp_pos[:, self.rank],
                "opponent_rel_vel": opp_vel[:, self.rank],
            }
        # gymnasium's spaces.Dict sorts its keys, and FlattenJaxObservation flattens in that sorted
        # order at train time -- so concatenate by SORTED key here (not dict insertion order) to feed
        # the network the identical layout. A mismatch permutes the obs and destroys the policy.
        return jnp.concatenate(
            [rel[k].reshape(1, -1).astype(jnp.float32) for k in sorted(rel)], axis=-1
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the attitude command [roll, pitch, yaw, thrust] for this drone."""
        obs_vec = self._obs_vector(obs)
        action = self._infer(self._agent, obs_vec)[0]  # (4,) in [-1, 1]
        action = action.at[2].set(0.0)  # ZeroYaw
        action = jnp.clip(action, -1.0, 1.0) * self._act_scale + self._act_offset  # normalize
        self._last_action = action
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
        """Signal completion once this drone has passed the whole configured gate order."""
        n_gates_passed = np.asarray(obs["n_gates_passed"]).reshape(-1)[self.rank]
        n_gate_passes = np.asarray(obs["gate_sequence"]).shape[-1]
        self._finished = bool(n_gates_passed >= n_gate_passes)
        return self._finished

    def episode_callback(self):
        """Reset per-episode state (called after each episode)."""
        self.episode_reset()

    def episode_reset(self):
        """Reset the tracked last action and finished flag."""
        self._last_action = jnp.zeros(_ACTION_DIM, dtype=jnp.float32)
        self._finished = False
