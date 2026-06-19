"""Modular wrappers for the drone racing environment.

Wrapper stack (bottom to top):
    VecDroneRaceEnv
    └── NormalizeRaceActions
    └── RandomGateSpawnWrapper      – teleport drones to random gate positions on reset/autoreset
    └── CrashPenaltyWrapper         – -1 on terminated
    └── GatePassageRewardWrapper    – +gate_bonus each time target_gate index increases
    └── ProximityRewardWrapper      – exp(-coef*dist) or progress_coef*Δdist_xy toward gate
    └── SpeedRewardWrapper          – speed_coef * max(v · gate_dir, 0)
    └── GateInViewRewardWrapper     – gate_in_view_coef * max(drone_fwd · gate_dir, 0)
    └── AltitudeRewardWrapper       – alt_coef * exp(-3 * |z - gate_z|)
    └── SurviveRewardWrapper        – +survive_coef every active step
    └── RPYPenaltyWrapper           – -rpy_coef * ||rpy||
    └── OOBPenaltyWrapper           – -oob_coef * altitude_violation (+ optional termination)
    └── VerticalSpeedPenaltyWrapper – -vz_coef * max(vz, 0) when z > vz_threshold
    └── SoftCollisionWrapper        – suppress crash terminations during early training
    └── GateObsWrapper              – convert raw dict obs to compact relative-position obs
    └── RaceStackObs                – observation history stacking
    └── ActionPenalty               – action smoothness / energy penalty
    └── FlattenJaxObservation       – flatten dict obs to a single vector
    └── JaxToTorch                  – move arrays to the torch device
"""


from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from jax.scipy.spatial.transform import Rotation as R
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorRewardWrapper
from gymnasium.spaces import flatten_space
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv

if TYPE_CHECKING:
    from jax import Array


# ---------------------------------------------------------------------------
# 1. GateObsWrapper
# ---------------------------------------------------------------------------
 
class GateObsWrapper(VectorEnv):
    """Convert raw RaceCoreEnv dict obs to compact relative-position observations.
 
    Body-frame output keys:
        pos(3), quat(4), vel(3), ang_vel(3),
        rel_target_body(3), target_normal_body(3),
        rel_next_body(3),   next_normal_body(3)         → 28 dims
 
    This wrapper only touches observations; it never modifies reward, terminated,
    or truncated.
    """
 
    def __init__(self, env: VectorEnv, n_gates: int = 4, body_frame_obs: bool = False):
        self.env = env
        self.num_envs = env.num_envs
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
 
        self.n_gates = n_gates
        self.body_frame_obs = body_frame_obs
 
    
        obs_spec = {
            "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "quat": spaces.Box(-1.0, 1.0, shape=(4,)),
            "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "rel_target_body": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "target_normal_body": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "rel_next_body": spaces.Box(-np.inf, np.inf, shape=(3,)),
            "next_normal_body": spaces.Box(-np.inf, np.inf, shape=(3,)),
        }
        
 
        self.single_observation_space = spaces.Dict(obs_spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
 
    # ------------------------------------------------------------------
    # Static JIT-compiled obs transformations
    # ------------------------------------------------------------------
    @staticmethod
    @jax.jit
    def _obs_body(obs: dict, n_gates: int) -> dict:
        drone_pos = obs["pos"]
        drone_quat = obs["quat"]
        target_gate = obs["target_gate"]
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
 
        safe_target = jp.clip(target_gate, 0, n_gates - 1)
        next_target = jp.clip(target_gate + 1, 0, n_gates - 1)
        idx = jp.arange(drone_pos.shape[0])
 
        target_pos = gates_pos[idx, safe_target]
        target_quat = gates_quat[idx, safe_target]
        next_pos = gates_pos[idx, next_target]
        next_quat = gates_quat[idx, next_target]
 
        drone_rot_inv = R.from_quat(drone_quat).inv()
        rel_target_body = drone_rot_inv.apply(target_pos - drone_pos)
        rel_next_body = drone_rot_inv.apply(next_pos - drone_pos)
 
        unit_x = jp.broadcast_to(
            jp.array([1.0, 0.0, 0.0], dtype=drone_pos.dtype), drone_pos.shape
        )
        target_fwd = R.from_quat(target_quat).apply(unit_x)
        next_fwd = R.from_quat(next_quat).apply(unit_x)
 
        return {
            "pos": obs["pos"],
            "quat": obs["quat"],
            "vel": obs["vel"],
            "ang_vel": obs["ang_vel"],
            "rel_target_body": rel_target_body,
            "target_normal_body": drone_rot_inv.apply(target_fwd),
            "rel_next_body": rel_next_body,
            "next_normal_body": drone_rot_inv.apply(next_fwd),
        }
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def _transform(self, obs: dict) -> dict:
        return self._obs_body(obs, self.n_gates)
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform(obs), info
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs), reward, terminated, truncated, info
 
    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()
 
    @property
    def unwrapped(self):
        return self.env.unwrapped
 
 
# ---------------------------------------------------------------------------
# 2. Reward wrapper base class
# ---------------------------------------------------------------------------
 
class _RewardWrapper(VectorEnv):
    """Minimal base for wrappers that only add a scalar term to the reward.
 
    Subclasses implement ``_reward(obs, reward, terminated, truncated, info)``
    and return ``(delta, terminated)``.  The base class adds ``delta`` to the
    incoming ``reward`` and wires up all boilerplate.
 
    All obs-only fields (observation_space, action_space, …) are forwarded
    from the inner env so subclasses never have to repeat them.
    """
 
    def __init__(self, env: VectorEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
 
    def _reward(
        self,
        obs: dict,
        reward: Array,
        terminated,
        truncated,
        info: dict,
    ) -> tuple[Array, Array]:
        """Return ``(delta_reward, terminated)``."""
        raise NotImplementedError
 
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        delta, terminated = self._reward(obs, reward, terminated, truncated, info)
        # print('wrapper reward ', reward)
        # print('delta wrapper ', delta)
        # print(f"wrapper nonzero={( reward + delta != 0).sum().item()}")
        return obs, reward + delta, terminated, truncated, info
 
    def close(self):
        return self.env.close()
 
    @property
    def unwrapped(self):
        return self.env.unwrapped
 
 
# ---------------------------------------------------------------------------
# 2a. CrashPenaltyWrapper
# ---------------------------------------------------------------------------
 
class CrashPenaltyWrapper(_RewardWrapper):
    """-1 per env on the step where it terminates (physics crash).
 
    The penalty is applied regardless of *why* the episode ended —
    SoftCollisionWrapper above this in the stack can suppress or override it
    during early training.
    """
 
    @staticmethod
    @jax.jit
    def _compute(terminated) -> Array:
        return -terminated.astype(jp.float32)
 
    def _reward(self, obs, reward, terminated, truncated, info):
        # print(self._compute(terminated))
        return self._compute(terminated), terminated
    
    def render(self):
        return self.env.render()
 
 
# ---------------------------------------------------------------------------
# 2b. GatePassageRewardWrapper
# ---------------------------------------------------------------------------
 
class GatePassageRewardWrapper(_RewardWrapper):
    """+gate_bonus each time the target_gate index advances.
 
    Tracks the previous target_gate and awards the bonus whenever the index
    increases and the previous index was valid (≥ 0).
 
    Autoreset handling (Option B): re-syncs _prev_target_gate independently
    after any env autoreset, with no coupling to RandomGateSpawnWrapper.
    """
 
    def __init__(self, env: VectorEnv, gate_bonus: float = 5.0):
        super().__init__(env)
        self.gate_bonus = gate_bonus
        self._prev_target_gate: Array | None = None
        self._was_done: np.ndarray | None = None
 
    @staticmethod
    @jax.jit
    def _compute(target_gate, prev_target_gate, gate_bonus: float) -> Array:
        gate_passed = (target_gate > prev_target_gate) & (prev_target_gate >= 0)
        return gate_bonus * gate_passed.astype(jp.float32)
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_target_gate = jp.array(obs["target_gate"])
        self._was_done = None
        return obs, info
 
    def _reward(self, obs, reward, terminated, truncated, info):
        # Option B: sync for any env that autoreset last step
        if self._was_done is not None and self._was_done.any():
            new_target = jp.array(obs["target_gate"])
            self._prev_target_gate = jp.where(
                jp.array(self._was_done), new_target, self._prev_target_gate
            )
 
        target_gate = jp.array(obs["target_gate"])
        delta = self._compute(target_gate, self._prev_target_gate, self.gate_bonus)
        self._prev_target_gate = target_gate
        self._was_done = np.array(terminated | truncated)
        return delta, terminated
    
    def render(self):
        return self.env.render()
 
 
# ---------------------------------------------------------------------------
# 2c. ProximityRewardWrapper
# ---------------------------------------------------------------------------
 
class ProximityRewardWrapper(_RewardWrapper):
    """Reward for being close to / making progress toward the target gate.
 
    Two modes, selected by whether progress_coef > 0:
 
    Exponential proximity (default):
        reward = exp(-proximity_coef * dist_3d)
 
    Progress / potential shaping:
        reward = progress_coef * Δdist_xy   (unilateral or bilateral)
 
    Autoreset handling (Option B): re-syncs _prev_dist independently.
    """
 
    def __init__(
        self,
        env: VectorEnv,
        n_gates: int = 4,
        proximity_coef: float = 2.0,
        progress_coef: float = 0.0,
        bilateral_progress: bool = False,
    ):
        super().__init__(env)
        self.n_gates = n_gates
        self.proximity_coef = proximity_coef
        self.progress_coef = progress_coef
        self.bilateral_progress = bilateral_progress
        self._prev_dist: Array | None = None
        self._was_done: np.ndarray | None = None
 
    @staticmethod
    @jax.jit
    def _dist_xy(pos, target_gate, gates_pos, n_gates: int) -> Array:
        safe_t = jp.clip(target_gate, 0, n_gates - 1)
        target_pos = gates_pos[jp.arange(pos.shape[0]), safe_t]
        return jp.linalg.norm((target_pos - pos)[:, :2], axis=-1)
 
    @staticmethod
    @partial(jax.jit, static_argnames=("n_gates", "use_progress", "use_bilateral"))
    def _compute(
        obs: dict,
        prev_dist,
        n_gates: int,
        proximity_coef: float,
        progress_coef: float,
        use_progress: bool,
        use_bilateral: bool,
    ) -> tuple[Array, Array]:
        drone_pos = obs["pos"]
        target_gate = obs["target_gate"]
        gates_pos = obs["gates_pos"]
 
        safe_target = jp.clip(target_gate, 0, n_gates - 1)
        idx = jp.arange(drone_pos.shape[0])
        target_pos = gates_pos[idx, safe_target]
        rel_pos = target_pos - drone_pos
        dist_3d = jp.linalg.norm(rel_pos, axis=-1)
        dist_xy = jp.linalg.norm(rel_pos[:, :2], axis=-1)
 
        active = (target_gate >= 0).astype(jp.float32)
 
        if use_progress:
            delta_d = prev_dist - dist_xy
            reward = progress_coef * (delta_d if use_bilateral else jp.maximum(delta_d, 0.0))
        else:
            reward = jp.exp(-proximity_coef * dist_3d)
 
        return active * reward, dist_xy
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_dist = self._dist_xy(
            jp.array(obs["pos"]),
            jp.array(obs["target_gate"]),
            jp.array(obs["gates_pos"]),
            self.n_gates,
        ) if self.progress_coef > 0.0 else None
        self._was_done = None
        return obs, info
 
    def _reward(self, obs, reward, terminated, truncated, info):
        # Option B: sync prev_dist for autoreset envs
        if self.progress_coef > 0.0 and self._was_done is not None and self._was_done.any():
            new_dist = self._dist_xy(
                jp.array(obs["pos"]),
                jp.array(obs["target_gate"]),
                jp.array(obs["gates_pos"]),
                self.n_gates,
            )
            self._prev_dist = jp.where(jp.array(self._was_done), new_dist, self._prev_dist)
 
        delta, dist_xy = self._compute(
            obs=obs,
            prev_dist=self._prev_dist,
            n_gates=self.n_gates,
            proximity_coef=self.proximity_coef,
            progress_coef=self.progress_coef,
            use_progress=self.progress_coef > 0.0 and self._prev_dist is not None,
            use_bilateral=self.bilateral_progress,
        )
 
        if self.progress_coef > 0.0:
            self._prev_dist = dist_xy
        self._was_done = np.array(terminated | truncated)

        return delta, terminated
    
    def render(self):
        return self.env.render()
 
 
# ---------------------------------------------------------------------------
# 2d. SpeedRewardWrapper
# ---------------------------------------------------------------------------
 
class SpeedRewardWrapper(_RewardWrapper):
    """speed_coef * max(velocity · direction_to_gate, 0).
 
    Only active when target_gate >= 0.
    """
 
    def __init__(self, env: VectorEnv, n_gates: int = 4, speed_coef: float = 0.1):
        super().__init__(env)
        self.n_gates = n_gates
        self.speed_coef = speed_coef
 
    @staticmethod
    @jax.jit
    def _compute(obs: dict, n_gates: int, speed_coef: float) -> Array:
        drone_pos = obs["pos"]
        drone_vel = obs["vel"]
        target_gate = obs["target_gate"]
        gates_pos = obs["gates_pos"]
 
        safe_target = jp.clip(target_gate, 0, n_gates - 1)
        idx = jp.arange(drone_pos.shape[0])
        target_pos = gates_pos[idx, safe_target]
        rel_pos = target_pos - drone_pos
        dist = jp.linalg.norm(rel_pos, axis=-1)
        direction = rel_pos / (dist[:, None] + 1e-6)
        speed_toward = jp.sum(drone_vel * direction, axis=-1)
        active = (target_gate >= 0).astype(jp.float32)
        return active * speed_coef * jp.maximum(speed_toward, 0.0)
 
    def _reward(self, obs, reward, terminated, truncated, info):
        return self._compute(obs, self.n_gates, self.speed_coef), terminated
 
 
# ---------------------------------------------------------------------------
# 2e. GateInViewRewardWrapper
# ---------------------------------------------------------------------------
 
class GateInViewRewardWrapper(_RewardWrapper):
    """gate_in_view_coef * max(drone_forward · gate_direction, 0).
 
    Rewards the drone for facing the target gate.  Only active when
    target_gate >= 0.
    """
 
    def __init__(self, env: VectorEnv, n_gates: int = 4, gate_in_view_coef: float = 0.0):
        super().__init__(env)
        self.n_gates = n_gates
        self.gate_in_view_coef = gate_in_view_coef
 
    @staticmethod
    @jax.jit
    def _compute(obs: dict, n_gates: int, gate_in_view_coef: float) -> Array:
        drone_pos = obs["pos"]
        drone_quat = obs["quat"]
        target_gate = obs["target_gate"]
        gates_pos = obs["gates_pos"]
 
        safe_target = jp.clip(target_gate, 0, n_gates - 1)
        idx = jp.arange(drone_pos.shape[0])
        target_pos = gates_pos[idx, safe_target]
        rel_pos = target_pos - drone_pos
        dist = jp.linalg.norm(rel_pos, axis=-1)
        gate_dir = rel_pos / (dist[:, None] + 1e-6)
 
        rot_mat = R.from_quat(drone_quat).as_matrix()
        drone_forward = rot_mat[:, :, 0]  # local +x is drone forward
        alignment = jp.sum(drone_forward * gate_dir, axis=-1)
 
        active = (target_gate >= 0).astype(jp.float32)
        return active * gate_in_view_coef * jp.maximum(alignment, 0.0)
 
    def _reward(self, obs, reward, terminated, truncated, info):
        return self._compute(obs, self.n_gates, self.gate_in_view_coef), terminated
 
 
# ---------------------------------------------------------------------------
# 2f. AltitudeRewardWrapper
# ---------------------------------------------------------------------------
 
class AltitudeRewardWrapper(_RewardWrapper):
    """alt_coef * exp(-3 * |drone_z - gate_z|).
 
    Rewards altitude alignment with the target gate.  Only active when
    target_gate >= 0.
    """
 
    def __init__(self, env: VectorEnv, n_gates: int = 4, alt_coef: float = 0.0):
        super().__init__(env)
        self.n_gates = n_gates
        self.alt_coef = alt_coef
 
    @staticmethod
    @jax.jit
    def _compute(obs: dict, n_gates: int, alt_coef: float) -> Array:
        drone_pos = obs["pos"]
        target_gate = obs["target_gate"]
        gates_pos = obs["gates_pos"]
 
        safe_target = jp.clip(target_gate, 0, n_gates - 1)
        idx = jp.arange(drone_pos.shape[0])
        target_pos = gates_pos[idx, safe_target]
 
        alt_error = jp.abs(drone_pos[:, 2] - target_pos[:, 2])
        active = (target_gate >= 0).astype(jp.float32)
        return active * alt_coef * jp.exp(-3.0 * alt_error)
 
    def _reward(self, obs, reward, terminated, truncated, info):
        return self._compute(obs, self.n_gates, self.alt_coef), terminated
 
 
# ---------------------------------------------------------------------------
# 2g. SurviveRewardWrapper
# ---------------------------------------------------------------------------
 
class SurviveRewardWrapper(_RewardWrapper):
    """+survive_coef every step while target_gate >= 0 (episode is active)."""
 
    def __init__(self, env: VectorEnv, survive_coef: float = 0.0):
        super().__init__(env)
        self.survive_coef = survive_coef
 
    @staticmethod
    @jax.jit
    def _compute(obs: dict, survive_coef: float) -> Array:
        active = (obs["target_gate"] >= 0).astype(jp.float32)
        return active * jp.full(obs["pos"].shape[0], survive_coef, dtype=jp.float32)
 
    def _reward(self, obs, reward, terminated, truncated, info):
        return self._compute(obs, self.survive_coef), terminated
 
 
# ---------------------------------------------------------------------------
# 2h. RPYPenaltyWrapper
# ---------------------------------------------------------------------------
 
class RPYPenaltyWrapper(_RewardWrapper):
    """-rpy_coef * ||roll, pitch, yaw||.
 
    Penalises tilting / spinning.  Applied unconditionally every step.
    """
 
    def __init__(self, env: VectorEnv, rpy_coef: float = 0.06):
        super().__init__(env)
        self.rpy_coef = rpy_coef
 
    @staticmethod
    @jax.jit
    def _compute(obs: dict, rpy_coef: float) -> Array:
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        return -rpy_coef * jp.linalg.norm(rpy, axis=-1)
 
    def _reward(self, obs, reward, terminated, truncated, info):
        return self._compute(obs, self.rpy_coef), terminated
 
 
# ---------------------------------------------------------------------------
# 2i. OOBPenaltyWrapper
# ---------------------------------------------------------------------------
 
class OOBPenaltyWrapper(_RewardWrapper):
    """-oob_coef * altitude_violation, with optional episode termination.
 
    When oob_coef > 0 and the drone exits [z_low, z_high], a proportional
    penalty is applied.  If terminate_on_oob is True the episode is also
    terminated (same behaviour as the original monolith when oob_coef > 0).
    """
 
    def __init__(
        self,
        env: VectorEnv,
        oob_coef: float = 0.0,
        z_low: float = 0.0,
        z_high: float = 2.0,
        terminate_on_oob: bool = True,
    ):
        super().__init__(env)
        self.oob_coef = oob_coef
        self.z_low = z_low
        self.z_high = z_high
        self.terminate_on_oob = terminate_on_oob
 
    @staticmethod
    @jax.jit
    def _compute(
        obs: dict,
        terminated,
        oob_coef: float,
        z_low: float,
        z_high: float,
        terminate_on_oob: bool,
    ) -> tuple[Array, Array]:
        z = obs["pos"][:, 2]
        oob = (z > z_high) | (z < z_low)
        penalty = -oob_coef * (
            jp.maximum(z - z_high, 0.0) + jp.maximum(z_low - z, 0.0)
        )
        new_terminated = jp.where(terminate_on_oob, terminated | oob, terminated)
        return penalty, new_terminated
 
    def _reward(self, obs, reward, terminated, truncated, info):
        delta, terminated = self._compute(
            obs, terminated, self.oob_coef, self.z_low, self.z_high, self.terminate_on_oob
        )
        return delta, terminated
 
 
# ---------------------------------------------------------------------------
# 2j. VerticalSpeedPenaltyWrapper
# ---------------------------------------------------------------------------
 
class VerticalSpeedPenaltyWrapper(_RewardWrapper):
    """-vz_coef * max(vz, 0) when the drone is above vz_threshold.
 
    Discourages gaining altitude above the threshold zone.
    """
 
    def __init__(
        self,
        env: VectorEnv,
        vz_coef: float = 0.0,
        vz_threshold: float = 0.5,
    ):
        super().__init__(env)
        self.vz_coef = vz_coef
        self.vz_threshold = vz_threshold
 
    @staticmethod
    @jax.jit
    def _compute(obs: dict, vz_coef: float, vz_threshold: float) -> Array:
        z = obs["pos"][:, 2]
        vz = obs["vel"][:, 2]
        above = (z > vz_threshold).astype(jp.float32)
        return -vz_coef * jp.maximum(vz, 0.0) * above
 
    def _reward(self, obs, reward, terminated, truncated, info):
        return self._compute(obs, self.vz_coef, self.vz_threshold), terminated
 
 
# ---------------------------------------------------------------------------
# 3. SoftCollisionWrapper
# ---------------------------------------------------------------------------
 
class SoftCollisionWrapper(VectorEnv):
    """Suppress crash terminations during the early phase of training.
 
    During phase 1 (global_step < soft_collision_steps):
    - Crash terminations (terminated=True, truncated=False) are suppressed;
      the episode continues.
    - The reward for those envs is replaced with a flat -soft_collision_penalty.
 
    During phase 2 (global_step >= soft_collision_steps):
    - Crashes terminate normally; this wrapper is a transparent pass-through.
 
    This wrapper never reads or modifies observations.
    """
 
    def __init__(
        self,
        env: VectorEnv,
        soft_collision_penalty: float = 5.0,
        soft_collision_steps: int = 5_000_000,
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
 
        self.soft_collision_penalty = soft_collision_penalty
        self.soft_collision_steps = soft_collision_steps
        self._total_steps = 0
 
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += self.num_envs
 
        if self._total_steps <= self.soft_collision_steps:
            # Identify envs that crashed (terminated by physics, not by time limit)
            soft_crashed = np.array(terminated) & ~np.array(truncated)
            if np.any(soft_crashed):
                sc = jp.array(soft_crashed)
                reward = jp.where(sc, jp.float32(-self.soft_collision_penalty), reward)
                terminated = jp.where(sc, False, terminated)
 
            # Log exactly once when phase 2 begins
            if (self._total_steps - self.num_envs < self.soft_collision_steps
                    <= self._total_steps):
                print(
                    f"[SoftCollisionWrapper] Phase 2 at step {self._total_steps}: "
                    f"hard termination on crash now active."
                )
 
        return obs, reward, terminated, truncated, info
 
    def close(self):
        return self.env.close()
 
    @property
    def unwrapped(self):
        return self.env.unwrapped
 
 
# ---------------------------------------------------------------------------
# 4. RandomGateSpawnWrapper
# ---------------------------------------------------------------------------
 
class RandomGateSpawnWrapper(VectorEnv):
    """Teleport drones to random gate positions on reset and autoreset.
 
    On every reset (and every subsequent autoreset detected via `_was_done`),
    each env has a `random_gate_ratio` probability of being teleported to a
    random gate.  The drone is placed `spawn_offset` metres before the gate
    along its forward axis, facing the gate, with small noise in position,
    orientation and velocity.
 
    This wrapper only modifies `obs` (and the underlying sim state).  It does
    not touch reward, terminated, or truncated.
 
    Autoreset handling (Option B)
    -----------------------------
    This wrapper maintains its own `_was_done` flag.  It re-spawns envs that
    were done at the *previous* step after the inner env has already autoreset
    them.  GateRewardWrapper does the same independently — no shared state.
    """
 
    def __init__(
        self,
        env: VectorEnv,
        n_gates: int = 4,
        random_gate_ratio: float = 1.0,
        spawn_offset: float = 0.75,
        spawn_pos_noise: float = 0.15,
        spawn_vel_noise: float = 0.3,
        seed: int = 0,
    ):
        self.env = env
        self.num_envs = env.num_envs
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
 
        self.n_gates = n_gates
        self.random_gate_ratio = random_gate_ratio
        self.spawn_offset = spawn_offset
        self.spawn_pos_noise = spawn_pos_noise
        self.spawn_vel_noise = spawn_vel_noise
 
        self._rng_key = jax.random.PRNGKey(seed)
        self._was_done: np.ndarray | None = None
 
    # ------------------------------------------------------------------
    # Spawn logic
    # ------------------------------------------------------------------
 
    def _spawn(self, obs: dict, mask: np.ndarray | None = None) -> dict:
        """Teleport masked envs to a random gate.
 
        Args:
            obs:  Raw observation dict from the inner env.
            mask: Boolean array (n_envs,).  If None, apply to all envs with
                  probability random_gate_ratio.
        """
        # Decide which envs to spawn this call
        self._rng_key, mask_key = jax.random.split(self._rng_key)
        prob_mask = jax.random.uniform(mask_key, (self.num_envs,)) < self.random_gate_ratio
 
        if mask is None:
            effective_mask = prob_mask
        else:
            effective_mask = jp.asarray(mask) & prob_mask
 
        if not bool(jp.any(effective_mask)):
            return obs
 
        self._rng_key, gate_key, pos_key, vel_key, angle_key = jax.random.split(
            self._rng_key, 5
        )
 
        target_gate = jp.asarray(obs["target_gate"], dtype=jp.int32)
        gates_pos = jp.asarray(obs["gates_pos"])
        gates_quat = jp.asarray(obs["gates_quat"])
        idx = jp.arange(self.num_envs)
 
        # Pick a random gate for each env; keep current gate for non-masked envs
        random_gates = jax.random.randint(gate_key, (self.num_envs,), 0, self.n_gates)
        spawn_gate = jp.where(effective_mask, random_gates, target_gate)
 
        spawn_gate_pos = gates_pos[idx, spawn_gate]
        spawn_gate_quat = gates_quat[idx, spawn_gate]
 
        # Gate forward direction (local +x axis)
        gate_rot = R.from_quat(spawn_gate_quat)
        unit_x = jp.broadcast_to(
            jp.array([1.0, 0.0, 0.0], dtype=spawn_gate_pos.dtype), spawn_gate_pos.shape
        )
        forward = gate_rot.apply(unit_x)
 
        # Position: before gate + uniform noise
        spawn_pos = spawn_gate_pos - self.spawn_offset * forward
        pos_noise = jax.random.uniform(
            pos_key,
            (self.num_envs, 3),
            minval=-self.spawn_pos_noise,
            maxval=self.spawn_pos_noise,
        )
        spawn_pos = spawn_pos + pos_noise
 
        # Orientation: match gate yaw + small RPY noise
        gate_euler = gate_rot.as_euler("xyz")
        angle_min = jp.array(
            [-np.radians(5), -np.radians(5), -np.radians(15)],
            dtype=spawn_gate_pos.dtype,
        )
        angle_max = jp.array(
            [np.radians(5), np.radians(5), np.radians(15)],
            dtype=spawn_gate_pos.dtype,
        )
        angle_noise = jax.random.uniform(
            angle_key, (self.num_envs, 3), minval=angle_min, maxval=angle_max
        )
        drone_rpy = angle_noise.at[:, 2].add(gate_euler[:, 2])
        spawn_quat = R.from_euler("xyz", drone_rpy).as_quat()
 
        # Velocity: small uniform noise
        spawn_vel = jax.random.uniform(
            vel_key,
            (self.num_envs, 3),
            minval=-self.spawn_vel_noise,
            maxval=self.spawn_vel_noise,
        )
 
        # Merge with existing sim state for non-masked envs
        core_env = self.env.unwrapped
        old_pos = core_env.sim.data.states.pos[:, 0]
        old_quat = core_env.sim.data.states.quat[:, 0]
        old_vel = core_env.sim.data.states.vel[:, 0]
 
        new_pos = jp.where(effective_mask[:, None], spawn_pos, old_pos).astype(jp.float32)
        new_quat = jp.where(effective_mask[:, None], spawn_quat, old_quat).astype(jp.float32)
        new_vel = jp.where(effective_mask[:, None], spawn_vel, old_vel).astype(jp.float32)
        new_ang_vel = jp.where(
            effective_mask[:, None],
            jp.zeros((self.num_envs, 3), dtype=jp.float32),
            core_env.sim.data.states.ang_vel[:, 0],
        ).astype(jp.float32)
 
        # Write new state into simulator
        core_env.sim.data = core_env.sim.data.replace(
            states=core_env.sim.data.states.replace(
                pos=core_env.sim.data.states.pos.at[:, 0, :].set(new_pos),
                quat=core_env.sim.data.states.quat.at[:, 0, :].set(new_quat),
                vel=core_env.sim.data.states.vel.at[:, 0, :].set(new_vel),
                ang_vel=core_env.sim.data.states.ang_vel.at[:, 0, :].set(new_ang_vel),
            )
        )
 
        # Update target gate and last_drone_pos in race env data
        old_target = core_env.data.target_gate[:, 0]
        new_target = jp.where(effective_mask, spawn_gate, old_target).astype(jp.int32)
        core_env.data = core_env.data.replace(
            target_gate=new_target[:, None],
            last_drone_pos=core_env.sim.data.states.pos,
        )
 
        # Return updated obs dict (shallow copy so caller's dict is not mutated)
        obs = dict(obs)
        obs["pos"] = core_env.sim.data.states.pos[:, 0]
        obs["quat"] = core_env.sim.data.states.quat[:, 0]
        obs["vel"] = core_env.sim.data.states.vel[:, 0]
        obs["ang_vel"] = core_env.sim.data.states.ang_vel[:, 0]
        obs["target_gate"] = core_env.data.target_gate[:, 0]
        return obs
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng_key = jax.random.PRNGKey(int(seed))
 
        obs, info = self.env.reset(**kwargs)
        obs = self._spawn(obs)  # spawn all envs on fresh reset
        self._was_done = None
        return obs, info
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
 
        # Re-spawn any env that just autoreset (Option B: self-contained)
        if self._was_done is not None and self._was_done.any():
            obs = self._spawn(obs, mask=self._was_done)
 
        self._was_done = np.array(terminated | truncated)
        return obs, reward, terminated, truncated, info
 
    def close(self):
        return self.env.close()
 
    @property
    def unwrapped(self):
        return self.env.unwrapped



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
        # print('Action penalty :', reward)
        return self.observations(obs), reward, terminated, truncated, info

    def observations(self, observations: dict) -> dict:
        """Override observation."""
        observations["last_action"] = self._last_action
        return observations

class FlattenJaxObservation(VectorObservationWrapper):
    """Flatten dict observations into a single array matching flatten_space exactly.

    Sorts keys alphabetically (matching flatten_space ordering).
    One-hot encodes MultiDiscrete spaces to match flatten_space output.
    """

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._sorted_keys = sorted(env.single_observation_space.spaces.keys())
        self._spaces = env.single_observation_space.spaces
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)

    def observations(self, observations: dict) -> Array:
        parts = []
        for k in self._sorted_keys:
            space = self._spaces[k]
            v = observations[k]

            if isinstance(space, spaces.MultiDiscrete):
                v = jp.asarray(v, dtype=jp.int32)
                if v.ndim == 1:
                    v = v[:, None]   # (n_envs,) → (n_envs, 1)
                one_hots = []
                for i, (n, s) in enumerate(zip(space.nvec, space.start)):
                    n_cats = int(n)              # nvec IS the number of categories
                    idx = v[:, i] - int(s)      # shift so min value maps to index 0
                    one_hots.append(
                        jp.eye(n_cats, dtype=jp.float32)[idx]   # (n_envs, n_cats)
                    )
                parts.append(jp.concatenate(one_hots, axis=-1))

            else:
                v = jp.asarray(v, dtype=jp.float32)
                parts.append(v.reshape(v.shape[0], -1))

        return jp.concatenate(parts, axis=-1)
    
    def render(self):
        return self.env.render()
    
class RaceStackObs(VectorObservationWrapper):
    """Observation history stacking for RaceCoreEnv pipeline.

    Custom version of StackObs that doesn't call env.unwrapped.obs(),
    which would return unsqueezed obs from RaceCoreEnv (wrong shape).
    Initializes history buffer with zeros instead.
    """

    def __init__(self, env: VectorEnv, n_obs: int = 0):
        super().__init__(env)
        self.n_obs = n_obs
        if self.n_obs > 0:
            spec = {k: v for k, v in self.single_observation_space.items()}
            spec["prev_obs"] = spaces.Box(-np.inf, np.inf, shape=(13 * self.n_obs,))
            self.single_observation_space = spaces.Dict(spec)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            self._prev_obs = jp.zeros((self.num_envs, self.n_obs, 13))

    def observations(self, observations: dict) -> dict:
        if self.n_obs > 0:
            observations["prev_obs"] = self._prev_obs.reshape(self.num_envs, -1)
            self._prev_obs = self._update_prev_obs(self._prev_obs, observations)
        return observations

    @staticmethod
    @jax.jit
    def _update_prev_obs(prev_obs: Array, obs: dict) -> Array:
        basic_obs_keys = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = jp.concatenate(
            [jp.reshape(obs[k], (obs[k].shape[0], -1)) for k in basic_obs_keys], axis=-1
        )
        return jp.concatenate([prev_obs[:, 1:, :], basic_obs[:, None, :]], axis=1)
    
class NormalizeRaceActions(VectorEnv):
    """Normalize agent actions from [-1, 1] to actual attitude action space.

    Also zeros out yaw command (index 2) for stability, matching the existing
    AngleReward wrapper behavior from train_rl.py.
    """

    def __init__(self, env: VecDroneRaceEnv):
        self.env = env
        self.num_envs = env.num_envs

        low = np.array(env.single_action_space.low, dtype=np.float32)
        high = np.array(env.single_action_space.high, dtype=np.float32)
        self._center = jp.array((high + low) / 2)
        self._scale = jp.array((high - low) / 2)

        self.single_action_space = spaces.Box(-1.0, 1.0, shape=low.shape, dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
        # print('normalize actions: ', self.observation_space.shape)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = jp.asarray(action)
        action = action.at[..., 2].set(0.0)
        # Scale from [-1, 1] to actual attitude bounds without host materialization.
        scaled = action * self._scale + self._center
        return self.env.step(scaled)

    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped