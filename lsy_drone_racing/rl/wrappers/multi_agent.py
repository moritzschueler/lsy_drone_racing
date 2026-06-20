"""Multi-agent opponent utilities adapted for the RL task code.

Provides `EgoOpponent`, `OpponentWrapper`, and `CheckpointPool` in this
module so `lsy_drone_racing.rl.tasks.multi_agent_racing` can import them
directly from the `rl.wrappers` package without depending on the
control training script.
"""
from __future__ import annotations

import random
import pickle
from flax import nnx
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.vector.utils import batch_space

from lsy_drone_racing.control.fixed_rl_opponent import AttitudeRL
from lsy_drone_racing.utils import load_config
from crazyflow.sim.visualize import draw_points
import jax
import jax.numpy as jnp

__all__ = ["EgoOpponent", "OpponentWrapper", "CheckpointPool"]


class EgoOpponent:
    """Wraps a saved ego Agent checkpoint for use as an opponent.

    This is a thin adapter that expects a JAX/Flax NNX Agent state.
    """

    def __init__(
        self,
        obs_shape: tuple,
        action_shape: tuple,
        weights: dict | None,
        single_obs_space: spaces.Dict,
        action_space: spaces.Box,
        device: str | None = None,
    ):
        # Use the JAX/NNX Agent from lsy_drone_racing.rl.agents.ppo_agent
        from lsy_drone_racing.rl.agents.ppo_agent import Agent

        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))
        self._obs_dim = obs_dim
        self._action_dim = action_dim

        # Create a model instance and load the saved NNX state if provided.
        self.agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(0))
        if weights:
            # weights is expected to be the pickled nnx.state(agent, nnx.Param)
            nnx.update(self.agent, weights)

        # Resolve the string device to a JAX device backend
        self._device = None
        if device is not None:
            try:
                # Map common torch string targets (e.g., 'cuda:0') to JAX naming convention
                backend = "gpu" if "cuda" in device.lower() else device.lower()
                self._device = jax.devices(backend)[0]
            except Exception:
                # Fallback to default JAX device if string matching fails
                self._device = None

        self._last_action: np.ndarray | None = None
        self._obs_spaces = single_obs_space.spaces

        self._action_low = action_space.low
        self._action_high = action_space.high

    def reset(self):
        self._last_action = None

    def compute_control(self, opp_obs_dict: dict, ego_obs_dict: dict) -> np.ndarray:
        obs_dict = dict(opp_obs_dict)

        obs_dict["opp_rel_pos"] = (
            np.asarray(ego_obs_dict["pos"]) - np.asarray(opp_obs_dict["pos"])
        ).astype(np.float32)
        obs_dict["opp_rel_vel"] = (
            np.asarray(ego_obs_dict["vel"]) - np.asarray(opp_obs_dict["vel"])
        ).astype(np.float32)

        if self._last_action is None:
            self._last_action = np.zeros(self._action_dim, dtype=np.float32)
        if "last_action" in obs_dict.keys():
            obs_dict["last_action"] = self._last_action

        # Flatten observations according to the single_obs_space spec
        keys = sorted(obs_dict.keys())
        parts = []
        for k in keys:
            v = obs_dict[k]
            space = self._obs_spaces.get(k)

            if space is not None and isinstance(space, spaces.MultiDiscrete):
                v = np.asarray(v, dtype=np.int32)
                if v.ndim == 0:
                    v = v[None]
                one_hots = []
                for i, (n, s) in enumerate(zip(space.nvec, space.start)):
                    n_cats = int(n)
                    idx = int(v[i]) - int(s)
                    one_hots.append(np.eye(n_cats, dtype=np.float32)[idx])
                parts.append(np.concatenate(one_hots))
            else:
                parts.append(np.asarray(v, dtype=np.float32).reshape(-1))

        flat_obs = np.concatenate(parts)

        # Assert against saved expected observation dimensions
        assert flat_obs.shape[0] == self._obs_dim, (
            f"EgoOpponent obs mismatch: got {flat_obs.shape[0]}, expected {self._obs_dim}."
        )

        # Convert to JAX array and add batch dimension (1, obs_dim)
        obs_array = jnp.asarray(flat_obs, dtype=jnp.float32)[None, :]
        if self._device is not None:
            obs_array = jax.device_put(obs_array, self._device)

        # Forward pass on the NNX module: returns (mean, log_std, value)
        # Using the policy mean serves as our deterministic action
        mean, _, _ = self.agent(obs_array)
        
        # Convert the JAX array back to a standard NumPy array and remove batch dim
        action_np = np.asarray(mean[0])
        self._last_action = action_np

        # Normalize action from [-1, 1] -> action_space bounds
        norm_action = self._action_low + (action_np + 1.0) * 0.5 * (
            self._action_high - self._action_low
        )
        return norm_action


class OpponentWrapper(VectorWrapper):
    """Wraps VecMultiDroneRaceEnv to expose a single-agent interface.

    Drone 0 = learner (controlled by actions passed to step()).
    Drone 1 = opponent (controlled internally by AttitudeRL or EgoOpponent instances).
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
        # Build spaces before super().__init__ so we can override them after
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

        inner_single_act = env.single_action_space
        act_shape = inner_single_act.shape[1:]
        act_low = inner_single_act.low[0].astype(np.float32)
        act_high = inner_single_act.high[0].astype(np.float32)
        single_act_space = spaces.Box(act_low, act_high, shape=act_shape, dtype=np.float32)

        # VectorWrapper sets self.env and forwards metadata/autoreset_mode
        super().__init__(env)

        # Override spaces to reflect the single-agent view
        self.single_observation_space = single_obs_space
        self.single_action_space = single_act_space
        self.observation_space = batch_space(single_obs_space, num_envs)
        self.action_space = batch_space(single_act_space, num_envs)

        # Config and coefficients
        self.config = config
        self._rank_coef = rank_coef
        self._segment_lead_coef = segment_lead_coef
        self._proximity_coef = proximity_coef
        self._proximity_threshold = proximity_threshold
        self._victory_coef = victory_coef
        self._prev_done: np.ndarray | None = None

        # Forwarded env attributes
        self.track = env.track
        self.device = env.device
        self.data = env.data
        self.sim = env.sim

        # Opponent pool setup
        self._attitude_config = load_config(Path(__file__).parents[3] / "config" / "level0.toml")
        self._fixed_pool_weights = []
        fixed_pool_dir = Path(__file__).parent.parent / "checkpoints/multi_agent_racing/fixed_policies"
        if fixed_pool_dir is not None and fixed_pool_dir.exists():
            extensions = ["*.ckpt", "*.pth"]
            checkpoint_paths = []
            for ext in extensions:
                checkpoint_paths.extend(fixed_pool_dir.glob(ext))
            for p in checkpoint_paths:
                state = torch.load(p, map_location="cpu", weights_only=True)
                self._fixed_pool_weights.append(state)
            print(f"Loaded {len(self._fixed_pool_weights)} fixed models from {fixed_pool_dir}")

        self._opponents: list[AttitudeRL | EgoOpponent] | None = None
        self._opponent_active = False
        self._opponent_types: np.ndarray | None = None
        self._self_play_weights: list[dict] = []
        self._latest_weights: dict | None = None
        self._ego_obs_shape: tuple | None = None
        self._ego_action_shape: tuple | None = None
        self.current_ego_obs: dict | None = None
        self.current_opp_obs: dict | None = None
        self.info: dict | None = None

    def set_ego_shapes(self, obs_shape: tuple, action_shape: tuple, top_env):
        self._ego_obs_shape = obs_shape
        self._ego_action_shape = action_shape
        self._obs_transform = self._build_obs_transform(top_env)

    def set_opponent_active(self, active: bool):
        self._opponent_active = active

    def update_opponent_pool(self, self_play_paths: list[Path], latest_path: Path | None, ratios: tuple[float, float, float]):
        fixed_ratio, self_ratio, latest_ratio = ratios
        self._self_play_weights = [pickle.loads(Path(p).read_bytes()) for p in self_play_paths]
        self._latest_weights = pickle.loads(Path(latest_path).read_bytes()) if latest_path is not None else None

        n_fixed = int(self.num_envs * fixed_ratio)
        n_self = int(self.num_envs * self_ratio)
        n_latest = self.num_envs - n_fixed - n_self

        self._opponent_types = np.array([0] * n_fixed + [1] * n_self + [2] * n_latest, dtype=np.int32)
        np.random.shuffle(self._opponent_types)

        if self._opponents is not None and self.current_opp_obs is not None:
            self._rebuild_all_opponents(self.current_opp_obs)

    def _get_weights_for_type(self, opponent_type: int) -> tuple[str, dict]:
        if opponent_type == 0 or self._ego_obs_shape is None:
            if self._fixed_pool_weights:
                return "fixed", random.choice(self._fixed_pool_weights)
            return "fixed", None

        if opponent_type == 1:
            if self._self_play_weights:
                return "ego", random.choice(self._self_play_weights)
            if self._fixed_pool_weights:
                return "fixed", random.choice(self._fixed_pool_weights)
            return "fixed", {}
        if opponent_type == 2:
            if self._latest_weights is not None:
                return "ego", self._latest_weights
            if self._fixed_pool_weights:
                return "fixed", random.choice(self._fixed_pool_weights)
            return "fixed", {}

        return "fixed", {}

    def _build_single_opponent(self, opp_obs: dict, env_idx: int) -> AttitudeRL | EgoOpponent:
        t = self._opponent_types[env_idx] if self._opponent_types is not None else 0
        kind, weights = self._get_weights_for_type(t)

        if kind == "fixed":
            single_obs = self._make_single_obs(opp_obs, env_idx)
            opp = AttitudeRL(single_obs, None, self._attitude_config)
            if weights is not None and weights:
                # weights is a pickled nnx state dict
                opp.agent.load_state_dict(weights) if hasattr(opp, 'agent') and hasattr(opp.agent, 'load_state_dict') else nnx.update(opp.agent, weights)
            return opp
        else:
            return EgoOpponent(
                obs_shape=self._ego_obs_shape,
                action_shape=self._ego_action_shape,
                weights=weights,
                single_obs_space=self.single_observation_space,
                action_space=self.single_action_space,
            )

    def _build_opponents(self, opp_obs: dict) -> list[AttitudeRL | EgoOpponent]:
        if self._opponent_types is None:
            self._opponent_types = np.zeros(self.num_envs, dtype=np.int32)
        return [self._build_single_opponent(opp_obs, i) for i in range(self.num_envs)]

    def _rebuild_all_opponents(self, opp_obs: dict):
        self._opponents = [self._build_single_opponent(opp_obs, i) for i in range(self.num_envs)]

    @staticmethod
    def _squeeze_and_split(obs: dict) -> tuple[dict, dict]:
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

    @staticmethod
    def _rank_reward(ego_obs: dict, opp_obs: dict) -> np.ndarray:
        ego_gate = np.asarray(ego_obs["target_gate"]).astype(np.float32)
        opp_gate = np.asarray(opp_obs["target_gate"]).astype(np.float32)
        return np.clip(ego_gate - opp_gate, 0, 3)

    @staticmethod
    def _proximity_penalty(ego_obs: dict, opp_obs: dict, threshold: float = 0.2) -> np.ndarray:
        rel_pos = np.asarray(opp_obs["pos"]) - np.asarray(ego_obs["pos"])
        dist = np.linalg.norm(rel_pos, axis=-1)
        return -np.clip(threshold - dist, 0.0, threshold)

    @staticmethod
    def _segment_lead_reward(ego_obs: dict, opp_obs: dict) -> np.ndarray:
        ego_gate = np.asarray(ego_obs["target_gate"])
        opp_gate = np.asarray(opp_obs["target_gate"])
        ego_pos = np.asarray(ego_obs["pos"])
        opp_pos = np.asarray(opp_obs["pos"])

        ego_next_gate_pos = np.asarray(ego_obs["gates_pos"])[np.arange(len(ego_gate)), ego_gate.clip(0)]
        ego_dist = np.linalg.norm(ego_next_gate_pos - ego_pos, axis=-1)
        opp_dist = np.linalg.norm(ego_next_gate_pos - opp_pos, axis=-1)
        dist_advantage = np.clip(opp_dist - ego_dist, 0.0, 5.0)
        same_segment = (ego_gate == opp_gate).astype(np.float32)
        return same_segment * dist_advantage

    @staticmethod
    def _victory_reward(ego_obs: dict, opp_obs: dict, terminated: np.ndarray) -> np.ndarray:
        ego_gate = np.asarray(ego_obs["target_gate"])
        opp_gate = np.asarray(opp_obs["target_gate"])
        ego_finished = (ego_gate == -1)
        ego_leading = (opp_gate != -1)
        victory = (ego_finished & ego_leading & np.asarray(terminated)).astype(np.float32)
        return victory * 50.0

    def _compute_competition_reward_components(self, ego_obs: dict, opp_obs: dict, terminated: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "rank":         self._rank_coef         * self._rank_reward(ego_obs, opp_obs),
            "segment_lead": self._segment_lead_coef * self._segment_lead_reward(ego_obs, opp_obs),
            "proximity":    self._proximity_coef    * self._proximity_penalty(ego_obs, opp_obs, self._proximity_threshold),
            "victory":      self._victory_coef      * self._victory_reward(ego_obs, opp_obs, terminated),
        }

    def _compute_competition_reward(self, ego_obs: dict, opp_obs: dict, terminated: np.ndarray) -> np.ndarray:
        return sum(self._compute_competition_reward_components(ego_obs, opp_obs, terminated).values())

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
            # print(reward.shape)
            # print(reward)

            base_reward = self._process_reward(reward)
            return self.current_ego_obs, base_reward, terminated[:, 0], truncated[:, 0], info

        opp_actions = self._compute_opponent_actions(self._prev_done)
        # print("ego_action stats:", ego_action.mean(), ego_action.min(), ego_action.max())
        # print("opp_action stats:", opp_actions.mean(), opp_actions.min(), opp_actions.max())
        action = np.stack([np.asarray(ego_action), opp_actions], axis=1)
        obs, reward, terminated, truncated, info = self.env.step(action)

        ego_obs, opp_obs = self._squeeze_and_split(obs)
        # print("ego pos:", ego_obs["pos"][0])
        # print("ego target_gate:", ego_obs["target_gate"][0])
        # print("ego disabled:", )  # need base env data
        # print("base env data disabled_drones:", self.env.unwrapped.data.disabled_drones[0])
        self.current_ego_obs = self._add_relative_obs(ego_obs, opp_obs)
        self.current_opp_obs = opp_obs
        self.info = info

        self._prev_done = np.asarray(terminated[:, 0] | truncated[:, 0])

        if self._prev_done.any():
            for i in np.where(self._prev_done)[0]:
                self._opponents[i] = self._build_single_opponent(opp_obs, i)

        base_reward = self._process_reward(reward)
        competition_components = self._compute_competition_reward_components(
            self.current_ego_obs, opp_obs, np.asarray(terminated[:, 0])
        )
        competition_reward = sum(competition_components.values())

        # Stash each term in info for PPO's per-component wandb logging
        info = {
            **info,
            **{f"rew/comp_{name}": v for name, v in competition_components.items()},
        }
        # print('base',base_reward)
        # print('comp', competition_reward)
        return (
            self.current_ego_obs,
            base_reward + competition_reward,
            terminated[:, 0],
            truncated[:, 0],
            info,
        )
    
    def _build_obs_transform(self, top_env):
        """Walk from the top wrapper down to OpponentWrapper, collecting transform_extra in order."""
        transforms = []
        e = top_env
        while e is not None and e is not self:
            if hasattr(type(e), "transform_extra"):
                print(f"Found transform_extra on {type(e).__name__}")
                transforms.append(e.transform_extra)
            e = getattr(e, "env", None)
        print(f"Total transforms found: {len(transforms)}")

        transforms = list(reversed(transforms))

        def pipeline(obs: dict):
            result = obs
            for fn in transforms:
                result = fn(result)
            assert not isinstance(result, dict), (
                f"_obs_transform returned a dict — FlattenJaxObservation.transform_extra missing? "
                f"Chain: {[fn.__qualname__ for fn in transforms]}"
            )
            return result

        return pipeline

    def _compute_opponent_actions(self, prev_done: np.ndarray) -> np.ndarray:
        has_ego_opponent = any(isinstance(o, EgoOpponent) for o in self._opponents)
        if has_ego_opponent:
            opp_with_rel = dict(self.current_opp_obs)
            opp_with_rel["opp_rel_pos"] = (
                np.asarray(self.current_ego_obs["pos"]) - np.asarray(self.current_opp_obs["pos"])
            ).astype(np.float32)
            opp_with_rel["opp_rel_vel"] = (
                np.asarray(self.current_ego_obs["vel"]) - np.asarray(self.current_opp_obs["vel"])
            ).astype(np.float32)
            # last_action is added by ActionPenalty above OpponentWrapper for the ego only;
            # reconstruct it for the opponent from each EgoOpponent's stored _last_action
            last_actions = []
            for opponent in self._opponents:
                if isinstance(opponent, EgoOpponent) and opponent._last_action is not None:
                    last_actions.append(opponent._last_action)
                else:
                    last_actions.append(np.zeros(self._ego_action_shape, dtype=np.float32))
            opp_with_rel["last_action"] = np.stack(last_actions, axis=0)  # (num_envs, action_dim)

            flat_opp_obs = self._obs_transform(opp_with_rel)

        actions = []
        for i, opponent in enumerate(self._opponents):
            if prev_done[i]:
                if isinstance(opponent, AttitudeRL):
                    opponent.episode_callback()
                else:
                    opponent.reset()

            if isinstance(opponent, AttitudeRL):
                single_opp_obs = self._make_single_obs(self.current_opp_obs, i)
                action = opponent.compute_control(single_opp_obs)
                opponent.step_callback(action, single_opp_obs, 0.0, False, False, {})
            else:
                obs_array = jnp.asarray(flat_opp_obs[i:i+1], dtype=jnp.float32)
                mean, _, _ = opponent.agent(obs_array)
                action = np.asarray(mean[0])
                opponent._last_action = action  # update before next step
                action = opponent._action_low + (action + 1.0) * 0.5 * (
                    opponent._action_high - opponent._action_low
                )

            actions.append(np.asarray(action).reshape(-1))
        return np.stack(actions, axis=0)

    def render(self):
        result = self.env.render()
        try:
            sim = self.env.unwrapped.sim
            ego_pos = np.asarray(self.current_ego_obs["pos"])
            opp_pos = np.asarray(self.current_opp_obs["pos"])
            draw_points(sim, ego_pos[0:1], rgba=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32), size=0.02)
            draw_points(sim, opp_pos[0:1], rgba=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), size=0.02)
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
        self._checkpoints: list[Path] = []

    def save(self, agent: nn.Module, global_step: int) -> Path:
        path = self.save_dir / f"checkpoint_{global_step}.ckpt"
        # For JAX/NNX agents we pickle the nnx.state(...) snapshot (matches rl/ppo.py)
        try:
            state = nnx.state(agent, nnx.Param)
            with open(path, "wb") as f:
                pickle.dump(state, f)
        except Exception:
            # Fallback: try to save PyTorch-style state_dict if provided
            torch.save(agent.state_dict(), path)
        self._checkpoints.append(path)
        if len(self._checkpoints) > self.max_checkpoints:
            old = self._checkpoints.pop(0)
            old.unlink(missing_ok=True)
        return path

    def sample(self, n: int, recency_bias: float = 0.7) -> list[Path]:
        if not self._checkpoints:
            return []
        if len(self._checkpoints) == 1:
            return self._checkpoints * n

        weights = np.array([recency_bias ** (len(self._checkpoints) - 1 - i) for i in range(len(self._checkpoints))])
        weights /= weights.sum()
        indices = np.random.choice(len(self._checkpoints), size=n, p=weights)
        return [self._checkpoints[i] for i in indices]

    def latest(self) -> Path | None:
        return self._checkpoints[-1] if self._checkpoints else None

    def __len__(self):
        return len(self._checkpoints)

def get_wrapper(env, cls):
    """Walk the wrapper stack to find a specific wrapper."""
    while env is not None:
        if isinstance(env, cls):
            return env
        env = getattr(env, "env", None)
    return None