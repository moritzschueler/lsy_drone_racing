"""A naive RL pipeline for drone racing."""

import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import fire
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
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
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from ml_collections import ConfigDict
from scipy.interpolate import CubicSpline

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn, rng_spec2fn
from lsy_drone_racing.utils import load_config

# Keep jp alias used throughout the environment classes
jp = jnp


# region Arguments
@dataclass
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "gpu"
    """environment and training device"""
    wandb_project_name: str = "ADR-PPO-Racing"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 1_500_000
    """total timesteps of the experiments"""
    learning_rate: float = 1.5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.94
    """the discount factor gamma"""
    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.26
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.007
    """coefficient of the entropy"""
    vf_coef: float = 0.7
    """coefficient of the value function"""
    max_grad_norm: float = 1.5
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
    act_coef: float = 0.02
    """reward coefficients for training"""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


# region Environment
class RandTrajEnv(DroneEnv):
    """Drone environment for following a random trajectory.

    This environment is used to follow a random trajectory. The observations contain the
    relative position errors to the next `n_samples` points that are distanced by `samples_dt`. The
    reward is based on the distance to the next trajectory point.
    """

    def __init__(
        self,
        n_samples: int = 10,
        trajectory_time: float = 15.0,
        samples_dt: float = 0.1,
        *,
        num_envs: int = 1,
        max_episode_time: float = 15.0,
        physics: Literal["so_rpy_rotor_drag", "first_principles"]
        | Physics = Physics.first_principles,
        drone_model: str = "cf21B_500",
        freq: int = 500,
        disturbances: ConfigDict | None = None,
        device: str = "cpu",
    ):
        """Initialize the environment and create the figure-eight trajectory.

        Args:
            n_samples: Number of next trajectory points to sample for observations.
            samples_dt: Time between trajectory sample points in seconds.
            trajectory_time: Total time for completing the figure-eight trajectory in seconds.
            num_envs: Number of environments to run in parallel.
            max_episode_time: Maximum episode time in seconds.
            physics: Physics backend to use.
            drone_model: Drone model of the environment.
            freq: Frequency of the simulation.
            disturbances: Disturbance configuration.
            device: Device to use for the simulation.
        """
        # Override reset randomization function
        self._reset_randomization = self.build_reset_randomization_fn(physics)

        super().__init__(
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=physics,
            drone_model=drone_model,
            freq=freq,
            device=device,
            reset_randomization=self._reset_randomization,
        )
        if trajectory_time < self.max_episode_time:
            raise ValueError("Trajectory time must be greater than max episode time")

        # Define trajectory sampling parameters
        self.num_waypoints = 10
        self.n_samples = n_samples
        self.samples_dt = samples_dt
        self.trajectory_time = trajectory_time
        self.n_steps = int(np.ceil(self.trajectory_time * self.freq))
        self.sample_offsets = np.array(np.arange(n_samples) * self.freq * samples_dt, dtype=int)
        self.trajectories = np.zeros((self.num_envs, self.n_steps, 3))

        # Set takeoff position and build default reset position
        self.takeoff_pos = np.array([-1.5, 1.0, 0.07])
        data = self.sim.data
        self.sim.data = data.replace(
            states=data.states.replace(
                pos=np.broadcast_to(self.takeoff_pos, (data.core.n_worlds, data.core.n_drones, 3))
            )
        )
        self.sim.build_default_data()

        # Apply disturbances specified for racing
        specs = {} if disturbances is None else disturbances
        self.disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        if "dynamics" in self.disturbances:
            disturbance_fn = build_dynamics_disturbance_fn(self.disturbances["dynamics"])
            self.sim.step_pipeline = (
                self.sim.step_pipeline[:2] + (disturbance_fn,) + self.sim.step_pipeline[2:]
            )
            self.sim.build_step_fn()

        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["local_samples"] = spaces.Box(-np.inf, np.inf, shape=(3 * self.n_samples,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Reset."""
        # Create a random trajectory based on spline interpolation
        t = np.linspace(0, self.trajectory_time, self.n_steps)
        scale = np.array([1.2, 1.2, 0.5])
        waypoints = (
            np.random.uniform(-1, 1, size=(self.sim.n_worlds, self.num_waypoints, 3)) * scale
        )
        waypoints = (
            waypoints + 0.3 * self.takeoff_pos + np.array([0.0, 0.0, 0.7])
        )  # shift up in z direction
        waypoints[:, :3, :] = np.array(
            [[-1.5, 1.0, 0.07], [-1.0, 0.55, 0.4], [0.3, 0.35, 0.7]]
        )  # set first three waypoints
        v0 = np.tile(np.array([[0.0, 0.0, 0.4]]), (self.sim.n_worlds, 1))  # takeoff velocity
        spline = CubicSpline(
            np.linspace(0, self.trajectory_time, self.num_waypoints),
            waypoints,
            axis=1,
            bc_type=((1, v0), "not-a-knot"),
        )
        self.trajectories = spline(t)  # (n_worlds, n_steps, 3)

        super().reset(seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        self._reset(options=options)  # call jax rest function
        self._marked_for_reset = self._marked_for_reset.at[...].set(False)
        return self.obs(), {}

    def render(self):
        """Render."""
        idx = np.clip(
            self.steps + self.sample_offsets[None, ...], 0, self.trajectories[0].shape[0] - 1
        )
        next_trajectory = self.trajectories[np.arange(self.trajectories.shape[0])[:, None], idx]
        draw_line(
            self.sim,
            self.trajectories[0, 0:-1:2, :],
            rgba=np.array([1, 1, 1, 0.4]),
            start_size=2.0,
            end_size=2.0,
        )
        draw_line(
            self.sim, next_trajectory[0], rgba=np.array([1, 0, 0, 1]), start_size=3.0, end_size=3.0
        )
        draw_points(self.sim, next_trajectory[0], rgba=np.array([1.0, 0, 0, 1]), size=0.01)
        self.sim.render()

    def obs(self) -> dict[str, Array]:
        """Observations."""
        obs = super().obs()
        idx = np.clip(
            self.steps + self.sample_offsets[None, ...], 0, self.trajectories[0].shape[0] - 1
        )
        dpos = (
            self.trajectories[np.arange(self.trajectories.shape[0])[:, None], idx]
            - self.sim.data.states.pos
        )
        obs["local_samples"] = dpos.reshape(-1, 3 * self.n_samples)
        return obs

    def reward(self) -> Array:
        """Rewards."""
        obs = self.obs()
        pos = obs["pos"]  # (num_envs, 3)
        goal = self.trajectories[np.arange(self.trajectories.shape[0])[:, None], self.steps][
            :, 0, :
        ]  # (num_envs, 3)
        # distance to next trajectory point
        norm_distance = jp.linalg.norm(pos - goal, axis=-1)
        reward = jp.exp(-2.0 * norm_distance)  # encourage flying close to goal
        reward = jp.where(
            self.terminated(), -1.0, reward
        )  # penalize drones that crash into the ground
        return reward

    def apply_action(self, action: Array):
        """Apply the commanded state action to the simulation."""
        action = action.reshape((self.sim.n_worlds, self.sim.n_drones, -1))
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, action.shape)
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        match self.sim.control:
            case "attitude":
                self.sim.attitude_control(action)
            case "state":
                self.sim.state_control(action)
            case _:
                raise ValueError(f"Unsupported control mode: {self.sim.control}")

    @property
    def steps(self) -> Array:
        """The current step in the trajectory."""
        return self.sim.data.core.steps // (self.sim.freq // self.freq) - 1

    @staticmethod
    @jax.jit
    def _terminated(pos: Array) -> Array:
        lower_bounds = jp.array([-4.0, -4.0, -0.0])
        upper_bounds = jp.array([4.0, 4.0, 4.0])
        terminate = jp.any((pos[:, 0, :] < lower_bounds) | (pos[:, 0, :] > upper_bounds), axis=-1)
        return terminate

    def build_reset_randomization_fn(self, physics: str) -> Callable[[SimData, Array], SimData]:
        """Reset randomization."""

        # Spin up rotors to help takeoff
        def _reset_randomization_so_rpy(data: SimData, mask: Array) -> SimData:
            rotor_vel = 0.05 * jp.ones(
                (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
            )
            data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
            return data

        def _reset_randomization_first_principles(data: SimData, mask: Array) -> SimData:
            rotor_vel = 10000.0 * jp.ones(
                (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
            )
            data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
            return data

        match physics:
            case "first_principles":
                return _reset_randomization_first_principles
            case "so_rpy" | "so_rpy_rotor" | "so_rpy_rotor_drag":
                return _reset_randomization_so_rpy
            case _:
                return _reset_randomization_so_rpy


# region Wrappers
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
            # Init obs buffer
            init_obs = env.unwrapped.obs()
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


class FlattenJaxObservation(VectorObservationWrapper):
    """Wrapper to flatten the observations."""

    def __init__(self, env: VectorEnv):
        """Init."""
        super().__init__(env)
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)

    def observations(self, observations: dict) -> dict:
        """Flatten observations."""
        return jp.concatenate(
            [jp.reshape(v, (v.shape[0], -1)) for k, v in observations.items()], axis=-1
        )


def set_seeds(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)


# region MakeEnvs
def make_envs(
    config: str = "level0.toml",
    num_envs: int = None,
    jax_device: str = "cpu",
    coefs: dict = {},
) -> VectorEnv:
    """Make environments for training RL policy."""
    config = load_config(Path(__file__).parents[2] / "config" / config)
    env = RandTrajEnv(
        n_samples=10,
        num_envs=num_envs,
        freq=config.env.freq,
        drone_model=config.sim.drone_model,
        physics=config.sim.physics,
        disturbances=config.env.disturbances,
        device=jax_device,
    )

    env = NormalizeActions(env)
    env = StackObs(env, n_obs=coefs.get("n_obs", 0))
    env = AngleReward(env, rpy_coef=coefs.get("rpy_coef", 0.04))
    env = ActionPenalty(
        env,
        act_coef=coefs.get("act_coef", 0.04),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.4),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 1.0),
    )
    env = FlattenJaxObservation(env)
    return env


# region Agent
class Agent(nn.Module):
    """RL Agent implemented as a Flax linen module.

    The forward pass returns (action_mean, log_std, value). Separate helper functions
    handle action sampling and log-probability computation.
    """

    action_dim: int

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, Array, Array]:
        """Forward pass.

        Args:
            x: Observation tensor of shape (batch, obs_dim).

        Returns:
            Tuple of (action_mean, log_std, value) with shapes
            (batch, action_dim), (1, action_dim), (batch, 1).
        """
        orth = nn.initializers.orthogonal
        zeros = nn.initializers.zeros

        # Critic
        v = nn.Dense(64, kernel_init=orth(jnp.sqrt(2)), bias_init=zeros)(x)
        v = nn.tanh(v)
        v = nn.Dense(64, kernel_init=orth(jnp.sqrt(2)), bias_init=zeros)(v)
        v = nn.tanh(v)
        value = nn.Dense(1, kernel_init=orth(1.0), bias_init=zeros)(v)

        # Actor
        m = nn.Dense(64, kernel_init=orth(jnp.sqrt(2)), bias_init=zeros)(x)
        m = nn.tanh(m)
        m = nn.Dense(64, kernel_init=orth(jnp.sqrt(2)), bias_init=zeros)(m)
        m = nn.tanh(m)
        mean = nn.tanh(nn.Dense(self.action_dim, kernel_init=orth(0.01), bias_init=zeros)(m))

        # Learnable log std: lower for roll/pitch/yaw, higher for thrust
        log_std = self.param("log_std", lambda _: jnp.array([[-1.0, -1.0, -1.0, 1.0]]))

        return mean, log_std, value


def _log_prob(action: Array, mean: Array, log_std: Array) -> Array:
    """Compute log probability of actions under a diagonal Gaussian."""
    std = jnp.exp(log_std)
    return jnp.sum(
        -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def _entropy(log_std: Array, batch_size: int) -> Array:
    """Compute entropy of a diagonal Gaussian, broadcast to batch."""
    per_dim = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + log_std  # (1, action_dim)
    return jnp.broadcast_to(jnp.sum(per_dim, axis=-1), (batch_size,))


# region Train
def train_ppo(
    args: Args, model_path: Path, jax_device: str, wandb_enabled: bool = False
) -> None:
    """Train a PPO agent.

    An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/.
    """
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    train_start_time = time.time()
    set_seeds(args.seed)
    print("Training on device:", jax_device)

    # env setup
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    envs = make_envs(num_envs=args.num_envs, jax_device=jax_device, coefs=r_coefs)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    action_dim = int(np.prod(act_shape))

    # Agent init
    agent = Agent(action_dim=action_dim)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    params = agent.init(init_key, jnp.zeros((1,) + obs_shape))["params"]

    # Optimizer: clip grads then update
    if args.anneal_lr:
        schedule = optax.linear_schedule(args.learning_rate, 0.0, args.num_iterations)
    else:
        schedule = args.learning_rate
    tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adamw(schedule, eps=1e-5))
    opt_state = tx.init(params)

    # ------------------------------------------------------------------ #
    # JIT-compiled inner functions (capture args, agent, tx via closure)  #
    # ------------------------------------------------------------------ #

    @jax.jit
    def policy_step(params: dict, obs: Array, rng_key: Array) -> tuple[Array, Array, Array]:
        mean, log_std, value = agent.apply({"params": params}, obs)
        std = jnp.exp(log_std)
        action = mean + std * jax.random.normal(rng_key, mean.shape)
        logprob = _log_prob(action, mean, log_std)
        return action, logprob, value.squeeze(-1)

    @jax.jit
    def get_value(params: dict, obs: Array) -> Array:
        _, _, value = agent.apply({"params": params}, obs)
        return value.squeeze(-1)

    @jax.jit
    def compute_gae(
        rewards: Array, values: Array, dones: Array, next_value: Array, next_done: Array
    ) -> tuple[Array, Array]:
        """GAE via lax.scan over reversed time steps."""

        def scan_fn(lastgaelam: Array, inputs: tuple) -> tuple[Array, Array]:
            reward, value, next_val, next_d = inputs
            nextnonterminal = 1.0 - next_d
            delta = reward + args.gamma * next_val * nextnonterminal - value
            advantage = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            return advantage, advantage

        # next_val and next_done for each step t: values[t+1] / dones[t+1], last step uses bootstrap
        next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)
        next_dones = jnp.concatenate([dones[1:], next_done[None]], axis=0)

        _, advantages_rev = jax.lax.scan(
            scan_fn,
            jnp.zeros(rewards.shape[1:]),
            (rewards[::-1], values[::-1], next_values[::-1], next_dones[::-1]),
        )
        advantages = advantages_rev[::-1]
        return advantages, advantages + values

    def ppo_loss_fn(
        params: dict,
        obs: Array,
        actions: Array,
        log_probs: Array,
        advantages: Array,
        returns: Array,
        b_values: Array,
    ) -> tuple[Array, tuple]:
        mean, log_std, new_values = agent.apply({"params": params}, obs)
        new_log_probs = _log_prob(actions, mean, log_std)
        entropy = _entropy(log_std, obs.shape[0])

        log_ratio = new_log_probs - log_probs
        ratio = jnp.exp(log_ratio)
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)

        mb_advantages = advantages
        if args.norm_adv:
            mb_advantages = (mb_advantages - jnp.mean(mb_advantages)) / (
                jnp.std(mb_advantages) + 1e-8
            )

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        # Value loss
        new_values_flat = new_values.reshape(-1)
        if args.clip_vloss:
            v_clipped = b_values + jnp.clip(
                new_values_flat - b_values, -args.clip_coef, args.clip_coef
            )
            v_loss = 0.5 * jnp.mean(
                jnp.maximum(
                    (new_values_flat - returns) ** 2,
                    (v_clipped - returns) ** 2,
                )
            )
        else:
            v_loss = 0.5 * jnp.mean((new_values_flat - returns) ** 2)

        entropy_loss = jnp.mean(entropy)
        total_loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
        return total_loss, (pg_loss, v_loss, entropy_loss, approx_kl, ratio)

    @jax.jit
    def update_epoch(
        params: dict,
        opt_state: optax.OptState,
        flat_data: tuple,
        rng_key: Array,
    ) -> tuple[dict, optax.OptState, tuple]:
        """One full epoch of minibatch PPO updates compiled as a single XLA program."""
        shuffled = jax.random.permutation(rng_key, args.batch_size)
        mb_inds = shuffled.reshape(args.num_minibatches, args.minibatch_size)

        def minibatch_step(
            carry: tuple, inds: Array
        ) -> tuple[tuple, tuple]:
            p, os = carry
            mb = tuple(x[inds] for x in flat_data)
            (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(p, *mb)
            updates, new_os = tx.update(grads, os, p)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_os), (loss, aux)

        (params, opt_state), metrics = jax.lax.scan(
            minibatch_step, (params, opt_state), mb_inds
        )
        return params, opt_state, metrics

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = jnp.asarray(next_obs)
    next_done = jnp.zeros(args.num_envs)
    sum_rewards = jnp.zeros(args.num_envs)
    sum_rewards_hist: list[float] = []

    for iteration in range(1, args.num_iterations + 1):
        start_time = time.time()

        # -- Rollout --
        obs_list, act_list, logp_list, val_list, rew_list, done_list = [], [], [], [], [], []

        for _ in range(args.num_steps):
            global_step += args.num_envs
            obs_list.append(next_obs)
            done_list.append(next_done)

            rng, action_key = jax.random.split(rng)
            action, logprob, value = policy_step(params, next_obs, action_key)
            act_list.append(action)
            logp_list.append(logprob)
            val_list.append(value)

            next_obs, reward, terminations, truncations, _ = envs.step(action)
            next_obs = jnp.asarray(next_obs)
            reward = jnp.asarray(reward)
            rew_list.append(reward)

            # Track episode returns; reset done envs before accumulating
            sum_rewards = jnp.where(next_done.astype(bool), 0.0, sum_rewards)
            next_done = jnp.asarray(terminations | truncations).astype(jnp.float32)
            sum_rewards = sum_rewards + reward

            if wandb_enabled and bool(jnp.any(next_done)):
                done_indices = np.where(np.array(next_done, dtype=bool))[0]
                for idx in done_indices:
                    r = float(sum_rewards[idx])
                    wandb.log({"train/reward": r}, step=global_step)
                    sum_rewards_hist.append(r)

        obs_buf = jnp.stack(obs_list)       # (T, E, obs_dim)
        act_buf = jnp.stack(act_list)       # (T, E, act_dim)
        logp_buf = jnp.stack(logp_list)     # (T, E)
        val_buf = jnp.stack(val_list)       # (T, E)
        rew_buf = jnp.stack(rew_list)       # (T, E)
        done_buf = jnp.stack(done_list)     # (T, E)

        # -- GAE --
        next_value = get_value(params, next_obs)
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, next_value, next_done)

        # -- Flatten batch --
        flat_data = (
            obs_buf.reshape((-1,) + obs_shape),
            act_buf.reshape((-1,) + act_shape),
            logp_buf.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
            val_buf.reshape(-1),
        )

        # -- PPO update epochs --
        last_metrics = None
        for _ in range(args.update_epochs):
            rng, epoch_key = jax.random.split(rng)
            params, opt_state, metrics = update_epoch(params, opt_state, flat_data, epoch_key)
            last_metrics = metrics
            if args.target_kl is not None:
                approx_kl = float(jnp.mean(metrics[1][3]))
                if approx_kl > args.target_kl:
                    break

        # -- Logging --
        y_pred = np.array(val_buf.reshape(-1))
        y_true = np.array(returns.reshape(-1))
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        end_time = time.time()
        if wandb_enabled and last_metrics is not None:
            _, (pg_arr, v_arr, ent_arr, kl_arr, ratio_arr) = last_metrics
            wandb.log(
                {
                    "losses/value_loss": float(jnp.mean(v_arr)),
                    "losses/policy_loss": float(jnp.mean(pg_arr)),
                    "losses/entropy": float(jnp.mean(ent_arr)),
                    "losses/approx_kl": float(jnp.mean(kl_arr)),
                    "losses/clipfrac": float(
                        jnp.mean(jnp.abs(ratio_arr - 1.0) > args.clip_coef)
                    ),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(global_step / (end_time - start_time)),
                },
                step=global_step,
            )
        print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")

    train_end_time = time.time()
    print(
        f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds."
    )
    if model_path is not None:
        with open(model_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Model saved to {model_path}")
    envs.close()
    return sum_rewards_hist


# region Evaluate
def evaluate_ppo(args: Args, n_eval: int, model_path: Path) -> tuple[list, list]:
    """Evaluate a trained PPO agent."""
    set_seeds(args.seed)
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    eval_env = make_envs(num_envs=1, coefs=r_coefs)
    action_dim = int(np.prod(eval_env.single_action_space.shape))

    agent = Agent(action_dim=action_dim)
    with open(model_path, "rb") as f:
        params = pickle.load(f)

    @jax.jit
    def deterministic_action(params: dict, obs: Array) -> Array:
        mean, _, _ = agent.apply({"params": params}, obs)
        return mean

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    ep_seed = args.seed

    for episode in range(n_eval):
        obs, _ = eval_env.reset(seed=(ep_seed := ep_seed + 1))
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            obs_jax = jnp.asarray(obs)
            act = deterministic_action(params, obs_jax)
            obs, reward, terminated, truncated, _ = eval_env.step(act)
            eval_env.render()
            done = bool(jnp.any(jnp.asarray(terminated) | jnp.asarray(truncated)))
            episode_reward += float(jnp.asarray(reward)[0])
            steps += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

    print(
        f"Average Reward = {np.mean(episode_rewards):.2f},"
        f" Length = {np.mean(episode_lengths)}"
    )
    eval_env.close()
    return episode_rewards, episode_lengths


# region Main
def main(wandb_enabled: bool = True, train: bool = True, eval: int = 1):
    """Main."""
    args = Args.create()
    model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
    jax_device = args.jax_device

    if train:
        train_ppo(args, model_path, jax_device, wandb_enabled)

    if eval > 0:
        episode_rewards, episode_lengths = evaluate_ppo(args, eval, model_path)
        if wandb_enabled and train:
            wandb.log(
                {
                    "eval/mean_rewards": np.mean(episode_rewards),
                    "eval/mean_steps": np.mean(episode_lengths),
                }
            )
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
