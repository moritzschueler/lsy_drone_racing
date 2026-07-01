from __future__ import annotations

import os

from pathlib import Path
import pickle


# Must be set before scipy is imported so scipy Rotations operate on JAX arrays.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import functools
from typing import Any, Callable
import torch.nn as nn

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.control.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from drone_controllers.mellinger.params import ForceTorqueParams
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space
from jax import Array
from scipy.spatial.transform import Rotation as R
import torch
from gymnasium.vector import VectorEnv

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.control.attitude_rl import AttitudeRL
from lsy_drone_racing.utils.utils import load_config


@struct.dataclass
class Wrapper(struct.PyTreeNode):
    """Base class for jittable wrappers that delegates common metadata to the wrapped base."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    @property
    def single_observation_space(self) -> spaces.Space:
        return getattr(self.base, "single_observation_space")

    @property
    def observation_space(self) -> spaces.Space:
        return getattr(self.base, "observation_space")

    @property
    def single_action_space(self) -> spaces.Space:
        return getattr(self.base, "single_action_space")

    @property
    def action_space(self) -> spaces.Space:
        return getattr(self.base, "action_space")

    @property
    def num_envs(self) -> int:
        return getattr(self.base, "num_envs")

    @property
    def unwrapped(self) -> struct.PyTreeNode:
        return getattr(self.base, "unwrapped", self.base)

    @property
    def steps(self) -> Array:
        return getattr(self.base, "steps")

    @staticmethod
    def recursive_replace(env: struct.PyTreeNode, **kwargs: Any) -> struct.PyTreeNode:
        """Recursively replace fields in the innermost base environment."""
        if isinstance(env, Wrapper):
            new_base = Wrapper.recursive_replace(env.base, **kwargs)
            return env.replace(base=new_base)
        return env.replace(**kwargs)

    def render(self, **kwargs: dict) -> None:
        return self.base.render(**kwargs)

    def close(self, **kwargs: Any) -> None:
        return self.base.close(**kwargs)


@struct.dataclass
class NormalizeActions(Wrapper):
    """Wrapper that exposes actions in [-1, 1] and rescales them to the simulator range."""

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
        """Create a NormalizeActions wrapper around `base`."""
        action_sim_low = jp.array(base.single_action_space.low)
        action_sim_high = jp.array(base.single_action_space.high)

        # rescale [-1,1] -> [low, high]
        scale = (action_sim_high - action_sim_low) / 2.0
        mean = (action_sim_high + action_sim_low) / 2.0

        def reset(
            env: NormalizeActions,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ) -> tuple[NormalizeActions, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def step(
            env: NormalizeActions, actions: Array
        ) -> tuple[NormalizeActions, tuple[Any, ...]]:
            action = jp.clip(actions, -1.0, 1.0) * scale + mean
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(
                env.base, action
            )
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


@struct.dataclass
class NormalizeActionsV2(Wrapper):
    """Like NormalizeActions, exposing actions in [-1, 1] and rescaling to the simulator range."""

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
    def create(cls, base: struct.PyTreeNode) -> "NormalizeActionsV2":
        """Create a NormalizeActionsV2 wrapper around `base`."""
        low = jp.array(base.single_action_space.low)
        high = jp.array(base.single_action_space.high)
        k0 = (high + low) / 2.0
        k1 = (high - low) / 2.0
        k2 = jp.zeros_like(k0)

        def _reset(
            env: NormalizeActionsV2,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ) -> tuple[NormalizeActionsV2, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _step(
            env: NormalizeActionsV2, actions: Array
        ) -> tuple[NormalizeActionsV2, tuple[Any, ...]]:
            actions = k0 + k1 * actions + k2 * (actions**2)  # [-1,1] -> simulator range
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(
                env.base, actions
            )
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


@struct.dataclass
class ZeroYaw(Wrapper):
    """Wrapper to set yaw output to zero."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> ZeroYaw:
        """Create an ZeroYaw around `base`."""

        def reset(
            env: ZeroYaw, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[ZeroYaw, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def step(env: ZeroYaw, actions: Array) -> tuple[ZeroYaw, tuple[Any, ...]]:
            actions = actions.at[..., 2].set(0.0)
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(
                env.base, actions
            )
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


@struct.dataclass
class AngleReward(Wrapper):
    """Wrapper to penalize orientation in the reward."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, weight: float = 0.08) -> AngleReward:
        """Create an AngleReward around `base`."""

        def reset(
            env: AngleReward, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[AngleReward, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _reward(reward: Array, observations: dict[str, Array]) -> Array:
            return reward - weight * R.from_quat(observations["quat"]).magnitude()

        def step(
            env: AngleReward, actions: Array
        ) -> tuple[AngleReward, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(
                env.base, actions
            )
            reward = _reward(reward, obs)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


@struct.dataclass
class ActionPenalty(Wrapper):
    """Wrapper to apply action penalty and augment observations with last_action."""

    base: struct.PyTreeNode | AngleReward = struct.field(pytree_node=True)
    last_actions: Array = struct.field(pytree_node=True)
    num_actions: int = struct.field(pytree_node=False)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        spec = {k: v for k, v in self.base.single_observation_space.items()}
        act_dim = self.base.action_space.shape[-1]
        spec["last_actions"] = spaces.Box(
            -np.inf, np.inf, shape=(self.num_actions, act_dim), dtype=np.float32
        )
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode | AngleReward,
        num_actions: int = 1,
        init_last_actions: tuple | None = None,
        hover_action: Array = jp.zeros((4,)),
        act_coefs: tuple = (0.0,) * 4,
        d_act_coefs: tuple = (0.0,) * 4,
    ) -> "ActionPenalty":
        """Create an ActionPenalty that augments obs with `last_action` and penalizes actions."""
        num_envs = base.num_envs
        act_dim = base.action_space.shape[-1]
        act_coefs = jp.array(act_coefs, dtype=jp.float32)
        d_act_coefs = jp.array(d_act_coefs, dtype=jp.float32)
        last_actions = jp.zeros((num_envs, num_actions, act_dim), dtype=jp.float32)
        if init_last_actions is not None:
            init_last_actions = jp.array(init_last_actions, dtype=jp.float32)
            last_actions = jp.broadcast_to(init_last_actions, last_actions.shape)

        def reset(
            env: ActionPenalty, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[ActionPenalty, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(
                base=base_env, last_actions=jp.zeros_like(env.last_actions)
            )
            obs["last_actions"] = env.last_actions
            return env, (obs, info)

        def step(
            env: ActionPenalty, action: Array
        ) -> tuple[ActionPenalty, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(
                env.base, action
            )
            # energy penalty
            action_deviation = action - hover_action
            reward = reward - jp.sum(act_coefs * (action_deviation**2), axis=-1)
            # smoothness penalty
            action_diff = action - env.last_actions[:, 0, :]
            reward = reward - jp.sum(d_act_coefs * (action_diff**2), axis=-1)
            new_last_actions = jp.roll(env.last_actions, shift=1, axis=1)
            new_last_actions = new_last_actions.at[:, 0, :].set(action)
            env = env.replace(base=base_env, last_actions=new_last_actions)
            obs["last_actions"] = env.last_actions
            if "final_obs" in info:
                info["final_obs"]["last_actions"] = env.last_actions
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            last_actions=last_actions,
            num_actions=num_actions,
            step=step,
            reset=reset,
        )


@struct.dataclass
class FlattenJaxObservation(Wrapper):
    """Wrapper to flatten dict observations into a single array."""

    base: Any = struct.field(pytree_node=True)

    @property
    def single_observation_space(self) -> spaces.Space:
        return flatten_space(self.base.single_observation_space)

    @property
    def observation_space(self) -> spaces.Space:
        return flatten_space(self.base.observation_space)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: Any) -> "FlattenJaxObservation":
        """Create a FlattenJaxObservation that concatenates dict observations."""

        def flatten_obs(observations: dict[str, Array]) -> Array:
            keys = sorted(observations.keys())
            return jp.concatenate(
                [
                    jp.reshape(observations[k], (observations[k].shape[0], -1))
                    for k in keys
                ],
                axis=-1,
            )

        def reset(
            env: FlattenJaxObservation,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ) -> tuple[FlattenJaxObservation, tuple[Array, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            flat_obs = flatten_obs(obs)
            env = env.replace(base=base_env)
            return env, (flat_obs, info)

        def step(
            env: FlattenJaxObservation, action: Array
        ) -> tuple[FlattenJaxObservation, tuple[Array, Any]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(
                env.base, action
            )
            flat_obs = flatten_obs(obs)
            if "final_obs" in info:
                info["final_obs"] = flatten_obs(info["final_obs"])
            env = env.replace(base=base_env)
            return env, (flat_obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


@struct.dataclass
class QuatToMatrixObs(Wrapper):
    """Wrapper to convert quaternion observations to rotation matrices."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        spec = {k: v for k, v in self.base.single_observation_space.items()}
        quat_space = spec.pop("quat")
        rot_mat_shape = quat_space.shape[:-1] + (3, 3)
        spec["rot_mat"] = spaces.Box(
            low=-1.0, high=1.0, shape=rot_mat_shape, dtype=np.float32
        )
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> QuatToMatrixObs:
        """Create a QuatToMatrixObs around `base`."""

        def reset(
            env: QuatToMatrixObs,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ) -> tuple[QuatToMatrixObs, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            obs["rot_mat"] = R.from_quat(obs.pop("quat")).as_matrix()
            env = env.replace(base=base_env)
            return env, (obs, info)

        def step(
            env: QuatToMatrixObs, action: Array
        ) -> tuple[QuatToMatrixObs, tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(
                env.base, action
            )
            obs["rot_mat"] = R.from_quat(obs.pop("quat")).as_matrix()
            if "final_obs" in info:
                quat = info["final_obs"].pop("quat")
                info["final_obs"]["rot_mat"] = R.from_quat(quat).as_matrix()
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


class OpponentWrapper(VectorEnv):
    """Wraps VecMultiDroneRaceEnv to expose a single-agent interface.

    Drone 0 = learner (controlled by actions passed to step()).
    Drone 1 = opponent (controlled internally by AttitudeRL or EgoOpponent instances).

    Opponent pool supports three types:
        0 = fixed AttitudeRL policy
        1 = past ego checkpoint (self-play)
        2 = latest ego checkpoint

    Adds opp_rel_pos and opp_rel_vel to ego observation so the learner
    can react to the opponent's position and velocity.
    """

    def __init__(
        self,
        num_envs: int,
        env: VectorEnv,
        config,
        rank_coef: float = 1.0,
        segment_lead_coef: float = 0.5,
        proximity_coef: float = 1.0,
        proximity_threshold: float = 0.1,
        victory_coef: float = 50.0,
    ):
        self.env = env
        self.config = config
        self.num_envs = num_envs
        self._rank_coef = rank_coef
        self._segment_lead_coef = segment_lead_coef
        self._proximity_coef = proximity_coef
        self._proximity_threshold = proximity_threshold
        self._victory_coef = victory_coef
        self._prev_done: np.ndarray | None = None

        # ── Observation space ─────────────────────────────────────────
        inner_single_obs = env.single_observation_space
        assert isinstance(inner_single_obs, spaces.Dict), (
            f"Expected Dict obs space, got {type(inner_single_obs)}"
        )
        single_obs_space = spaces.Dict({
            k: (
                spaces.Box(v.low[0], v.high[0], v.shape[1:], v.dtype)
                if isinstance(v, spaces.Box)
                else spaces.MultiDiscrete(v.nvec[1:], start=v.start[1:], dtype=v.dtype)
            )
            for k, v in inner_single_obs.spaces.items()
        } | {
            "opp_rel_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "opp_rel_vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

        # ── Action space ──────────────────────────────────────────────
        inner_single_act = env.single_action_space  # Box (2, act_dim)
        act_shape = inner_single_act.shape[1:]
        act_low = inner_single_act.low[0].astype(np.float32)
        act_high = inner_single_act.high[0].astype(np.float32)
        single_act_space = spaces.Box(act_low, act_high, shape=act_shape, dtype=np.float32)

        # ── VectorEnv init ────────────────────────────────────────────
        super().__init__()
        self.num_envs = num_envs
        self.single_observation_space = single_obs_space
        self.single_action_space = single_act_space
        self.observation_space = batch_space(single_obs_space, num_envs)
        self.action_space = batch_space(single_act_space, num_envs)

        # ── Fixed opponent setup ──────────────────────────────────────
        self._attitude_config = load_config(
            Path(__file__).parents[2] / "config" / "level0.toml"
        )
        fixed_model_path = Path(__file__).parent / "trajectory_follow_15s.ckpt"
        self._fixed_weights = torch.load(fixed_model_path, map_location=torch.device("cpu"))

        # ── Opponent pool state ───────────────────────────────────────
        self._opponents: list[AttitudeRL | EgoOpponent] | None = None
        self._opponent_active = False
        self._opponent_types: np.ndarray | None = None  # (n_envs,) int: 0=fixed,1=self,2=latest
        self._self_play_weights: list[dict] = []
        self._latest_weights: dict | None = None

        # Set after full wrapper stack is built (shapes not known yet at init)
        self._ego_obs_shape: tuple | None = None
        self._ego_action_shape: tuple | None = None

        # ── State buffers ─────────────────────────────────────────────
        self.current_ego_obs: dict | None = None
        self.current_opp_obs: dict | None = None
        self.info: dict | None = None

    # ── Pool management ───────────────────────────────────────────────

    def set_ego_shapes(self, obs_shape: tuple, action_shape: tuple):
        """Call after the full wrapper stack is built to register ego network shapes."""
        self._ego_obs_shape = obs_shape
        self._ego_action_shape = action_shape

    def set_opponent_active(self, active: bool):
        self._opponent_active = active

    def update_opponent_pool(
        self,
        self_play_paths: list[Path],
        latest_path: Path | None,
        ratios: tuple[float, float, float],  # (fixed, self_play, latest)
    ):
        """Reload weights and reassign env-to-opponent-type mapping.
        
        Safe to call mid-training; rebuilds all opponents at next episode boundary.
        """
        fixed_ratio, self_ratio, latest_ratio = ratios

        # Load self-play checkpoints
        self._self_play_weights = [
            torch.load(p, map_location="cpu") for p in self_play_paths
        ]
        self._latest_weights = (
            torch.load(latest_path, map_location="cpu")
            if latest_path is not None else None
        )

        # Assign types to envs
        n_fixed  = int(self.num_envs * fixed_ratio)
        n_self   = int(self.num_envs * self_ratio)
        n_latest = self.num_envs - n_fixed - n_self

        self._opponent_types = np.array(
            [0] * n_fixed + [1] * n_self + [2] * n_latest,
            dtype=np.int32,
        )
        np.random.shuffle(self._opponent_types)

        # Rebuild if already running
        if self._opponents is not None and self.current_opp_obs is not None:
            self._rebuild_all_opponents(self.current_opp_obs)

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_weights_for_type(self, opponent_type: int) -> tuple[str, dict]:
        """Return (kind, state_dict) for the given opponent type.
        
        kind is 'fixed' or 'ego', determining which class to instantiate.
        Falls back to fixed if ego shapes not set or no checkpoints available.
        """
        if opponent_type == 0 or self._ego_obs_shape is None:
            return "fixed", self._fixed_weights

        if opponent_type == 1:
            if self._self_play_weights:
                return "ego", np.random.choice(self._self_play_weights)
            return "fixed", self._fixed_weights

        if opponent_type == 2:
            if self._latest_weights is not None:
                return "ego", self._latest_weights
            return "fixed", self._fixed_weights

        return "fixed", self._fixed_weights

    def _build_single_opponent(
        self, opp_obs: dict, env_idx: int
    ) -> AttitudeRL | EgoOpponent:
        """Instantiate the correct opponent class for this env."""
        t = self._opponent_types[env_idx] if self._opponent_types is not None else 0
        kind, weights = self._get_weights_for_type(t)

        if kind == "fixed":
            single_obs = self._make_single_obs(opp_obs, env_idx)
            opp = AttitudeRL(single_obs, None, self._attitude_config)
            opp.agent.load_state_dict(weights)
            return opp
        else:
            return EgoOpponent(
                obs_shape=self._ego_obs_shape,
                action_shape=self._ego_action_shape,
                weights=weights,
            )

    def _build_opponents(self, opp_obs: dict) -> list[AttitudeRL | EgoOpponent]:
        if self._opponent_types is None:
            self._opponent_types = np.zeros(self.num_envs, dtype=np.int32)
        return [self._build_single_opponent(opp_obs, i) for i in range(self.num_envs)]

    def _rebuild_all_opponents(self, opp_obs: dict):
        self._opponents = [
            self._build_single_opponent(opp_obs, i) for i in range(self.num_envs)
        ]

    @staticmethod
    def _squeeze_and_split(obs: dict) -> tuple[dict, dict]:
        """Split (n_envs, 2, feat_dim) obs dict into (ego, opp) dicts."""
        ego, opp = {}, {}
        for k, v in obs.items():
            arr = np.asarray(v)
            while arr.ndim > 2 and arr.shape[1] == 1:
                arr = arr.squeeze(1)
            assert arr.shape[1] == 2, (
                f"Key '{k}': expected 2 drones at axis 1, got shape {arr.shape}"
            )
            ego[k] = arr[:, 0]
            opp[k] = arr[:, 1]
        return ego, opp

    @staticmethod
    def _add_relative_obs(ego_obs: dict, opp_obs: dict) -> dict:
        ego_obs["opp_rel_pos"] = (
            np.asarray(opp_obs["pos"]) - np.asarray(ego_obs["pos"])
        ).astype(np.float32)
        ego_obs["opp_rel_vel"] = (
            np.asarray(opp_obs["vel"]) - np.asarray(ego_obs["vel"])
        ).astype(np.float32)
        return ego_obs

    @staticmethod
    def _process_reward(reward) -> np.ndarray:
        r = np.asarray(reward)
        while r.ndim > 1 and r.shape[1] == 1:
            r = r.squeeze(1)
        if r.ndim == 2:
            r = r[:, 0]
        return r

    def _make_single_obs(self, obs: dict, env_idx: int) -> dict:
        return {k: np.asarray(v[env_idx]) for k, v in obs.items()}

    # ── Reward components ─────────────────────────────────────────────

    @staticmethod
    def _rank_reward(ego_obs: dict, opp_obs: dict) -> np.ndarray:
        ego_gate = np.asarray(ego_obs["target_gate"]).astype(np.float32)
        opp_gate = np.asarray(opp_obs["target_gate"]).astype(np.float32)
        return np.clip(ego_gate - opp_gate, 0, 3)

    @staticmethod
    def _proximity_penalty(
        ego_obs: dict, opp_obs: dict, threshold: float = 0.2
    ) -> np.ndarray:
        rel_pos = np.asarray(opp_obs["pos"]) - np.asarray(ego_obs["pos"])
        dist = np.linalg.norm(rel_pos, axis=-1)
        return -np.clip(threshold - dist, 0.0, threshold)

    @staticmethod
    def _segment_lead_reward(ego_obs: dict, opp_obs: dict) -> np.ndarray:
        ego_gate = np.asarray(ego_obs["target_gate"])
        opp_gate = np.asarray(opp_obs["target_gate"])
        ego_pos  = np.asarray(ego_obs["pos"])
        opp_pos  = np.asarray(opp_obs["pos"])

        ego_next_gate_pos = np.asarray(ego_obs["gates_pos"])[
            np.arange(len(ego_gate)), ego_gate.clip(0)
        ]
        ego_dist = np.linalg.norm(ego_next_gate_pos - ego_pos, axis=-1)
        opp_dist = np.linalg.norm(ego_next_gate_pos - opp_pos, axis=-1)
        dist_advantage = np.clip(opp_dist - ego_dist, 0.0, 5.0)
        same_segment = (ego_gate == opp_gate).astype(np.float32)
        return same_segment * dist_advantage

    @staticmethod
    def _victory_reward(
        ego_obs: dict, opp_obs: dict, terminated: np.ndarray
    ) -> np.ndarray:
        ego_gate = np.asarray(ego_obs["target_gate"])
        opp_gate = np.asarray(opp_obs["target_gate"])
        ego_finished = (ego_gate == -1)
        ego_leading  = (opp_gate != -1)
        victory = (ego_finished & ego_leading & np.asarray(terminated)).astype(np.float32)
        return victory * 50.0

    def _compute_competition_reward(
        self, ego_obs: dict, opp_obs: dict, terminated: np.ndarray
    ) -> np.ndarray:
        reward = np.zeros(self.num_envs, dtype=np.float32)
        reward += self._rank_coef         * self._rank_reward(ego_obs, opp_obs)
        reward += self._segment_lead_coef * self._segment_lead_reward(ego_obs, opp_obs)
        reward += self._proximity_coef    * self._proximity_penalty(
            ego_obs, opp_obs, self._proximity_threshold
        )
        reward += self._victory_coef      * self._victory_reward(
            ego_obs, opp_obs, terminated
        )
        return reward

    # ── Core API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        ego_obs, opp_obs = self._squeeze_and_split(obs)
        self.current_opp_obs = opp_obs
        self._prev_done = np.zeros(self.num_envs, dtype=bool)
        self.info = info

        if self._opponent_active:
            self.current_ego_obs = self._add_relative_obs(ego_obs, opp_obs)
            self._opponents = self._build_opponents(opp_obs)
        else:
            ego_obs["opp_rel_pos"] = np.zeros((self.num_envs, 3), dtype=np.float32)
            ego_obs["opp_rel_vel"] = np.zeros((self.num_envs, 3), dtype=np.float32)
            self.current_ego_obs = ego_obs
            self._opponents = None

        return self.current_ego_obs, info

    def step(self, ego_action):
        if not self._opponent_active:
            dummy_action = np.zeros_like(np.asarray(ego_action))
            action = np.stack([np.asarray(ego_action), dummy_action], axis=1)
            obs, reward, terminated, truncated, info = self.env.step(action)

            ego_obs, opp_obs = self._squeeze_and_split(obs)
            ego_obs["opp_rel_pos"] = np.zeros((self.num_envs, 3), dtype=np.float32)
            ego_obs["opp_rel_vel"] = np.zeros((self.num_envs, 3), dtype=np.float32)
            self.current_ego_obs = ego_obs
            self.current_opp_obs = opp_obs
            self._prev_done = np.asarray(terminated[:, 0] | truncated[:, 0])

            base_reward = self._process_reward(reward)
            return self.current_ego_obs, base_reward, terminated[:, 0], truncated[:, 0], info

        opp_actions = self._compute_opponent_actions(self._prev_done)

        action = np.stack([np.asarray(ego_action), opp_actions], axis=1)
        obs, reward, terminated, truncated, info = self.env.step(action)

        ego_obs, opp_obs = self._squeeze_and_split(obs)
        self.current_ego_obs = self._add_relative_obs(ego_obs, opp_obs)
        self.current_opp_obs = opp_obs
        self.info = info

        self._prev_done = np.asarray(terminated[:, 0] | truncated[:, 0])

        # Rebuild opponents for done envs — respect their assigned type
        if self._prev_done.any():
            for i in np.where(self._prev_done)[0]:
                self._opponents[i] = self._build_single_opponent(opp_obs, i)

        base_reward = self._process_reward(reward)
        competition_reward = self._compute_competition_reward(
            self.current_ego_obs, opp_obs, np.asarray(terminated[:, 0])
        )

        return (
            self.current_ego_obs,
            base_reward + competition_reward,
            terminated[:, 0],
            truncated[:, 0],
            info,
        )

    def _compute_opponent_actions(self, prev_done: np.ndarray) -> np.ndarray:
        """Run each opponent, dispatching to the correct API per type."""
        actions = []
        for i, opponent in enumerate(self._opponents):
            if prev_done[i]:
                if isinstance(opponent, AttitudeRL):
                    opponent.episode_callback()
                else:
                    opponent.reset()

            single_opp_obs = self._make_single_obs(self.current_opp_obs, i)

            if isinstance(opponent, AttitudeRL):
                action = opponent.compute_control(single_opp_obs)
                opponent.step_callback(action, single_opp_obs, 0.0, False, False, {})
            else:
                # EgoOpponent: pass both perspectives so it can see the ego drone
                single_ego_obs = self._make_single_obs(self.current_ego_obs, i)
                action = opponent.compute_control(single_opp_obs, single_ego_obs)

            actions.append(np.asarray(action).reshape(-1))
        return np.stack(actions, axis=0)

    # ── Pass-throughs ─────────────────────────────────────────────────

    def render(self):
        result = self.env.render()
        try:
            sim = self.env.unwrapped.sim
            ego_pos = np.asarray(self.current_ego_obs["pos"])
            opp_pos = np.asarray(self.current_opp_obs["pos"])
            draw_points(
                sim, ego_pos[0:1],
                rgba=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
                size=0.05,
            )
            draw_points(
                sim, opp_pos[0:1],
                rgba=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                size=0.05,
            )
        except Exception as e:
            print(e)
        return result

    def close(self):
        return self.env.close()

    def call(self, method: str, *args, **kwargs):
        return self.env.call(method, *args, **kwargs)

    def get_attr(self, name: str):
        return self.env.get_attr(name)

    def set_attr(self, name: str, values):
        return self.env.set_attr(name, values)
    

class CheckpointPool:
    """Manages a rolling pool of past agent checkpoints."""
    
    def __init__(self, max_checkpoints: int, save_dir: Path):
        self.max_checkpoints = max_checkpoints
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: list[Path] = []  # oldest to newest
    
    def save(self, agent: nn.Module, global_step: int) -> Path:
        path = self.save_dir / f"checkpoint_{global_step}.ckpt"
        torch.save(agent.state_dict(), path)
        self._checkpoints.append(path)
        # Evict oldest if over limit
        if len(self._checkpoints) > self.max_checkpoints:
            old = self._checkpoints.pop(0)
            old.unlink(missing_ok=True)
        return path
    
    def sample(self, n: int, recency_bias: float = 0.7) -> list[Path]:
        """Sample n checkpoints with optional recency bias."""
        if not self._checkpoints:
            return []
        if len(self._checkpoints) == 1:
            return self._checkpoints * n
        # Exponential weights: newer = higher probability
        weights = np.array([recency_bias ** (len(self._checkpoints) - 1 - i) 
                           for i in range(len(self._checkpoints))])
        weights /= weights.sum()
        indices = np.random.choice(len(self._checkpoints), size=n, p=weights)
        return [self._checkpoints[i] for i in indices]
    
    def latest(self) -> Path | None:
        return self._checkpoints[-1] if self._checkpoints else None
    
    def __len__(self):
        return len(self._checkpoints)