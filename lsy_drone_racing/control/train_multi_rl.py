"""A naive RL pipeline for drone racing."""

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from functools import partial
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorRewardWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from ml_collections import ConfigDict
from scipy.interpolate import CubicSpline
from torch import Tensor
from torch.distributions.normal import Normal
from lsy_drone_racing.control.attitude_rl import AttitudeRL
from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn, rng_spec2fn
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.multi_drone_race import VecMultiDroneRaceEnv

from lsy_drone_racing.utils.utils import load_config
from lsy_drone_racing.control.wrappers import (GateObsWrapper, 
                                               CrashPenaltyWrapper,
                                               GatePassageRewardWrapper,
                                               ProximityRewardWrapper,
                                               SpeedRewardWrapper,
                                               GateInViewRewardWrapper,
                                               AltitudeRewardWrapper,
                                               SurviveRewardWrapper,
                                               RPYPenaltyWrapper,
                                               OOBPenaltyWrapper,
                                               VerticalSpeedPenaltyWrapper,
                                               SoftCollisionWrapper, 
                                               RandomGateSpawnWrapper, 
                                               ActionPenalty, 
                                               FlattenJaxObservation, 
                                               NormalizeRaceActions, 
                                               RaceStackObs)


# region Arguments
@dataclass
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    jax_device: str = "gpu"  
    """environment device"""
    wandb_project_name: str = "ADR-PPO-Racing"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-3
    """the learning rate of the optimizer"""
    num_envs: int = 2048
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.98
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.3
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.005
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings
    n_obs: int = 2
    rpy_coef: float = 0.06
    d_act_th_coef: float = 0.4
    d_act_xy_coef: float = 1.0
    act_coef: float = 0.077
    """reward coefficients for training"""

    opponent_start_step: int = 20_000
    """global step at which the opponent agent is introduced"""

     # Opponent pool distribution (must sum to 1.0)
    opponent_fixed_ratio: float = 0.4    # fixed AttitudeRL policies
    opponent_self_ratio: float = 0.4     # own past checkpoints  
    opponent_latest_ratio: float = 0.2   # latest checkpoint (most recent self)
    
    # Self-play settings
    checkpoint_save_interval: int = 30_000   # save own checkpoint every N steps
    max_self_play_checkpoints: int = 10       # keep last N own checkpoints

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


def set_seeds(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_wrapper(env, cls):
    """Walk the wrapper stack to find a specific wrapper."""
    while env is not None:
        if isinstance(env, cls):
            return env
        env = getattr(env, "env", None)
    return None


# region MakeEnvs
def make_race_envs(
    config: str = "multi_level2.toml",
    num_envs: int = 1024,
    jax_device: str = "cpu",
    torch_device=None,
    coefs: dict | None = None,
) -> VectorEnv:
    """Create VecMultiDroneRaceEnv with training wrappers.

    """

    if torch_device is None:
        torch_device = torch.device("cpu")
    if coefs is None:
        coefs = {}

    cfg = load_config(Path(__file__).parents[2] / "config" / config)
    control_mode = cfg.env.kwargs[0]["control_mode"]
    n_gates = len(cfg.env.track.gates)

    

    # def print_dict_shapes(space, label=""):
    #     print(f"\n--- {label} ---")
    #     if hasattr(space, 'spaces'):  # Check if it's a Dict space
    #         for key, sub_space in space.spaces.items():
    #             if hasattr(sub_space, 'shape'):
    #                 print(f"  {key}: {sub_space.shape}")
    #             else:
    #                 print(f"  {key}: {sub_space}")
    #     else:
    #         # For flattened/Box spaces later in the pipeline
    #         print(f"  Shape: {getattr(space, 'shape', space)}")

    env = VecMultiDroneRaceEnv(
        num_envs=num_envs,
        freq=np.array([kwargs["freq"] for kwargs in cfg.env.kwargs], dtype=np.int64)[0],
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.kwargs[0]["sensor_range"],
        control_mode=control_mode,
        disturbances=cfg.env.get("disturbances", None),
        randomizations=cfg.env.get("randomizations", None),
        max_episode_steps=coefs.get("max_episode_steps", 1500),
        device=jax_device,
    )
    env = OpponentWrapper(num_envs,env, cfg)
    # print_dict_shapes(env.single_observation_space, "Initial Dict Space")
    env = NormalizeRaceActions(env)
    # print_dict_shapes(env.single_observation_space, "normal")
    # env = CrashPenaltyWrapper(env)
    # print_dict_shapes(env.single_observation_space, "crash")
    env = ProximityRewardWrapper(env, progress_coef=10, proximity_coef=0, bilateral_progress=False)
    # print_dict_shapes(env.single_observation_space, "proximity")
    env = GatePassageRewardWrapper(env)
    # print_dict_shapes(env.single_observation_space, "gate passage")
    # print_dict_shapes(env.single_observation_space, "gate obs")
    # env = RaceStackObs(env, n_obs=coefs.get("n_obs", 2))
    # print_dict_shapes(env.single_observation_space, "race stack")
    # env = ActionPenalty(
    #     env,
    #     act_coef=coefs.get("act_coef", 0.00002),
    #     d_act_th_coef=coefs.get("d_act_th_coef", 0.0005),
    #     d_act_xy_coef=coefs.get("d_act_xy_coef", 0.0005),
    # )
    # print_dict_shapes(env.single_observation_space, "action")
    env = FlattenJaxObservation(env)
    # print_dict_shapes(env.single_observation_space, "flatten")
    env = JaxToTorch(env, torch_device)

    opponent_w = get_wrapper(env, OpponentWrapper)
    if opponent_w is not None:
        opponent_w.set_ego_shapes(
            obs_shape=env.single_observation_space.shape,
            action_shape=env.single_action_space.shape,
        )

    return env


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class EgoOpponent:
    """Wraps a saved ego Agent checkpoint for use as an opponent."""
    def __init__(self, obs_shape: tuple, action_shape: tuple,
                 weights: dict, single_obs_space: spaces.Dict,
                 action_space: spaces.Box, 
                 device=torch.device("cpu")):
        self._action_dim = int(np.prod(action_shape))
        self.agent = Agent(obs_shape, action_shape).to(device)
        self.agent.load_state_dict(weights)
        self.agent.eval()
        self._device = device
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
        if 'last_action' in obs_dict.keys():
            obs_dict["last_action"] = self._last_action
        # flatten observations
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

        expected = self.agent.critic[0].weight.shape[1]
        assert flat_obs.shape[0] == expected, (
            f"EgoOpponent obs mismatch: got {flat_obs.shape[0]}, expected {expected}."
        )

        obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(obs_tensor, deterministic=True)
        action_np = action.squeeze(0).cpu().numpy()
        
        
        self._last_action = action_np
        # print("Opponent raw action: ", action)
        
        # normalize action
        norm_action = self._action_low + (action_np + 1.0) * 0.5 * (self._action_high - self._action_low)
        return norm_action

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
        # fixed_model_path = Path(__file__).parent / "trajectory_follow_15s.ckpt"
        # self._fixed_weights = torch.load(fixed_model_path, map_location=torch.device("cpu"))

        self._fixed_pool_weights = []
        fixed_pool_dir = Path(__file__).parent / "fixed_attitude_rl_parameters"
        if fixed_pool_dir is not None and fixed_pool_dir.exists():
            # Find all checkpoints (.ckpt or .pth) in the target directory
            extensions = ["*.ckpt", "*.pth"]
            checkpoint_paths = []
            for ext in extensions:
                checkpoint_paths.extend(fixed_pool_dir.glob(ext))
            
            # Load them all into memory
            for p in checkpoint_paths:
                weights = torch.load(p, map_location="cpu")
                self._fixed_pool_weights.append(weights)
                
            print(f" Loaded {len(self._fixed_pool_weights)} fixed models from {fixed_pool_dir}")

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
        """Return (kind, state_dict) for the given opponent type."""
        # Type 0 or Fallback: Sample from the fixed folder pool if available
        if opponent_type == 0 or self._ego_obs_shape is None:
            if self._fixed_pool_weights:
                return "fixed", random.choice(self._fixed_pool_weights)
            return "fixed", self._fixed_weights

        if opponent_type == 1:
            if self._self_play_weights:
                # Swapped to standard random.choice to avoid NumPy dict-parsing crashes
                return "ego", random.choice(self._self_play_weights)
            if self._fixed_pool_weights:
                return "fixed", random.choice(self._fixed_pool_weights)
            return "fixed", self._fixed_weights

        if opponent_type == 2:
            if self._latest_weights is not None:
                return "ego", self._latest_weights
            if self._fixed_pool_weights:
                return "fixed", random.choice(self._fixed_pool_weights)
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
                single_obs_space=self.single_observation_space,
                action_space=self.single_action_space,  # <--- Pass the unnormalized single action space bounds here
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
        """Run each opponent."""
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

        weights = np.array([recency_bias ** (len(self._checkpoints) - 1 - i) 
                           for i in range(len(self._checkpoints))])
        weights /= weights.sum()
        indices = np.random.choice(len(self._checkpoints), size=n, p=weights)
        return [self._checkpoints[i] for i in indices]
    
    def latest(self) -> Path | None:
        return self._checkpoints[-1] if self._checkpoints else None
    
    def __len__(self):
        return len(self._checkpoints)

# region Agent
class Agent(nn.Module):
    """RL Agent."""

    def __init__(self, obs_shape: tuple, action_shape: tuple, hidden_size: int = 64):
        """Init network structures."""
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, torch.tensor(action_shape).prod()), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 4)
        )

    def get_value(self, x: Tensor) -> Tensor:
        """Value estimation."""
        return self.critic(x)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Action output."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # During learning the agent explores the environment by sampling actions from a Normal
        # distribution. The standard deviation is a learnable parameter that should decrease during
        # training as the agent gets more confident in its actions.
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# region Train
def train_ppo(
    args: Args, model_path: Path, device: torch.device, jax_device: str, wandb_enabled: bool = False
) -> None:
    if wandb_enabled:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    train_start_time = time.time()
    set_seeds(args.seed)
    print("Training on device:", device, "| Environment device:", jax_device)

    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    envs = make_race_envs(
        num_envs=args.num_envs, jax_device=jax_device, torch_device=device, coefs=r_coefs
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    # flatten_wrapper = get_wrapper(envs, FlattenJaxObservation)
    # if flatten_wrapper is not None:
    #     print(f"FlattenJax sorted keys: {flatten_wrapper._sorted_keys}")

    opponent_wrapper = get_wrapper(envs, OpponentWrapper)
    opponent_active = False

    # ── Self-play pool ────────────────────────────────────────────────
    checkpoint_pool = CheckpointPool(
        max_checkpoints=args.max_self_play_checkpoints,
        save_dir=Path(__file__).parent / "self_play_checkpoints",
    )
    last_checkpoint_step = 0

    agent = Agent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs      = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions  = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards  = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones    = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values   = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)  # float, matches PPO logic
    sum_rewards = torch.zeros(args.num_envs).to(device)
    sum_rewards_hist = []

    try:
        for iteration in range(1, args.num_iterations + 1):
            start_time = time.time()

            # ── Activate opponent pool ────────────────────────────────
            if not opponent_active and global_step >= args.opponent_start_step:
                opponent_wrapper.set_opponent_active(True)
                opponent_active = True
                opponent_wrapper.update_opponent_pool(
                    self_play_paths=[],
                    latest_path=None,
                    ratios=(1.0, 0.0, 0.0),
                )
                print(f"[Step {global_step}] Opponent pool activated (fixed only)")
                if wandb_enabled:
                    wandb.log({"curriculum/opponent_active": 1}, step=global_step)
                next_obs, _ = envs.reset(seed=args.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)
                sum_rewards = torch.zeros(args.num_envs).to(device)

            # ── LR annealing ──────────────────────────────────────────
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            # ── Rollout ───────────────────────────────────────────────
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                # print("Ego action: ", action)
                actions[step] = action
                logprobs[step] = logprob

                prev_done = next_done.clone()
                next_obs, reward, terminations, truncations, infos = envs.step(action)
                rewards[step] = reward
                next_done = (terminations | truncations).float()

                sum_rewards += reward
                just_finished = next_done.bool() & ~prev_done.bool()
                if wandb_enabled and just_finished.any():
                    for r in sum_rewards[just_finished]:
                        wandb.log({"train/episode_return": r.item()}, step=global_step)
                        sum_rewards_hist.append(r.item())
                sum_rewards[next_done.bool()] = 0

                # ── Checkpoint save — checked per step, not per iteration ──
                if (
                    opponent_active
                    and global_step - last_checkpoint_step >= args.checkpoint_save_interval
                ):
                    ckpt_path = checkpoint_pool.save(agent, global_step)
                    last_checkpoint_step = global_step

                    pool_fill = min(len(checkpoint_pool) / args.max_self_play_checkpoints, 1.0)
                    self_ratio   = args.opponent_self_ratio   * pool_fill
                    latest_ratio = args.opponent_latest_ratio * pool_fill
                    fixed_ratio  = 1.0 - self_ratio - latest_ratio

                    opponent_wrapper.update_opponent_pool(
                        self_play_paths=checkpoint_pool.sample(
                            n=max(1, len(checkpoint_pool))
                        ),
                        latest_path=checkpoint_pool.latest(),
                        ratios=(fixed_ratio, self_ratio, latest_ratio),
                    )
                    print(
                        f"[Step {global_step}] Checkpoint saved → {ckpt_path.name} | "
                        f"Pool: {len(checkpoint_pool)} ckpts | "
                        f"fixed={fixed_ratio:.2f} self={self_ratio:.2f} latest={latest_ratio:.2f}"
                    )
                    if wandb_enabled:
                        wandb.log({
                            "curriculum/fixed_ratio":   fixed_ratio,
                            "curriculum/self_ratio":    self_ratio,
                            "curriculum/latest_ratio":  latest_ratio,
                            "curriculum/pool_size":     len(checkpoint_pool),
                        }, step=global_step)

            # ── GAE ───────────────────────────────────────────────────
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # ── PPO update ────────────────────────────────────────────
            b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs   = logprobs.reshape(-1)
            b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns    = returns.reshape(-1)
            b_values     = values.reshape(-1)

            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                        )
                        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if wandb_enabled:
                wandb.log({
                    "charts/learning_rate":    optimizer.param_groups[0]["lr"],
                    "losses/value_loss":        v_loss.item(),
                    "losses/policy_loss":       pg_loss.item(),
                    "losses/entropy":           entropy_loss.item(),
                    "losses/old_approx_kl":     old_approx_kl.item(),
                    "losses/approx_kl":         approx_kl.item(),
                    "losses/clipfrac":          np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(args.num_envs * args.num_steps / (time.time() - start_time)),
                }, step=global_step)

            print(f"Iter {iteration}/{args.num_iterations} took {time.time() - start_time:.2f}s")

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted!")
        if model_path is not None:
            torch.save(agent.state_dict(), model_path)
            print(f"[SAFETY] Model saved to {model_path}")
        envs.close()

    print(f"Training for {global_step} steps took {time.time() - train_start_time:.2f}s")
    if model_path is not None:
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    envs.close()
    return sum_rewards_hist


# region Evaluate
def evaluate_ppo(args: Args, n_eval: int, model_path: Path) -> tuple[float, float]:
    """Evaluate against three opponent types: fixed, past checkpoint, latest checkpoint."""
    set_seeds(args.seed)
    device = torch.device("cpu")
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }

    # Find available self-play checkpoints
    ckpt_dir = Path(__file__).parent / "self_play_checkpoints"
    ckpt_files = sorted(ckpt_dir.glob("checkpoint_*.ckpt")) if ckpt_dir.exists() else []
    
    # Build opponent configs to evaluate against
    # Each entry: (label, ratios, self_play_paths, latest_path)
    opponent_configs = [
        ("fixed",   (1.0, 0.0, 0.0), [],         None),
    ]
    if len(ckpt_files) >= 2:
        # A random past checkpoint (not the latest)
        past_ckpt = [random.choice(ckpt_files[:-1])]
        opponent_configs.append(("past_checkpoint", (0.0, 1.0, 0.0), past_ckpt, None))
    if len(ckpt_files) >= 1:
        opponent_configs.append(("latest_checkpoint", (0.0, 0.0, 1.0), [], ckpt_files[-1]))

    agent = None
    all_results = {}

    for label, ratios, self_play_paths, latest_path in opponent_configs:
        print(f"\n── Evaluating vs {label} ──")

        eval_env = make_race_envs(num_envs=1, coefs=r_coefs)
        opponent_wrapper = get_wrapper(eval_env, OpponentWrapper)
        opponent_wrapper.set_opponent_active(True)
        opponent_wrapper.update_opponent_pool(
            self_play_paths=self_play_paths,
            latest_path=latest_path,
            ratios=ratios,
        )

        # Build agent once, reuse across configs
        if agent is None:
            agent = Agent(
                eval_env.single_observation_space.shape,
                eval_env.single_action_space.shape,
            ).to(device)
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()

        episode_rewards = []
        episode_lengths = []
        ep_seed = args.seed

        for episode in range(n_eval):
            obs, _ = eval_env.reset(seed=(ep_seed := ep_seed + 1))
            done = torch.zeros(1, dtype=torch.bool, device=device)
            episode_reward = 0.0
            steps = 0

            while not done.any():
                act, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                # print("Ego action: ", act)

                obs, reward, terminated, truncated, info = eval_env.step(act.detach())
                eval_env.render()
                done = terminated | truncated
                episode_reward += reward[0].item()
                steps += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

        mean_r = np.mean(episode_rewards)
        mean_l = np.mean(episode_lengths)
        print(f"  → Mean Reward = {mean_r:.2f}, Mean Length = {mean_l:.0f}")
        all_results[label] = {"rewards": episode_rewards, "lengths": episode_lengths}

        eval_env.close()

    # Summary table
    print("\n══ Evaluation Summary ══")
    for label, res in all_results.items():
        print(f"  {label:20s}  reward={np.mean(res['rewards']):7.2f}  "
              f"length={np.mean(res['lengths']):6.0f}")

    return all_results


# region Main
def main(wandb_enabled: bool = True, train: bool = True, eval: int = 1):
    """Main."""
    args = Args.create()
    model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device

    if train:  # use "--train False" to skip training
        train_ppo(args, model_path, device, jax_device, wandb_enabled)

    if eval > 0:
        all_results = evaluate_ppo(args, eval, model_path)
        if wandb_enabled and train:
            for label, res in all_results.items():
                wandb.log({
                    f"eval/{label}/mean_reward": np.mean(res["rewards"]),
                    f"eval/{label}/mean_length": np.mean(res["lengths"]),
                })
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
