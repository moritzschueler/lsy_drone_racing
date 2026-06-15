"""Random trajectory following task: track a randomly generated spline trajectory."""

from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from jax import Array
from ml_collections import ConfigDict
from scipy.interpolate import CubicSpline

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn, rng_spec2fn
from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.wrappers.observation import FlattenJaxObservation, StackObs
from lsy_drone_racing.rl.wrappers.reward import ActionPenalty, AngleReward
from lsy_drone_racing.utils import load_config

jp = jnp

# Default Args overrides for this task (merged by the CLI before Args.create).
DEFAULTS: dict[str, Any] = {}


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
        """Initialize the environment and create the random trajectory.

        Args:
            n_samples: Number of next trajectory points to sample for observations.
            samples_dt: Time between trajectory sample points in seconds.
            trajectory_time: Total time for completing the trajectory in seconds.
            num_envs: Number of environments to run in parallel.
            max_episode_time: Maximum episode time in seconds.
            physics: Physics backend to use.
            drone_model: Drone model of the environment.
            freq: Frequency of the simulation in iterations per second.
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
        self.num_waypoints = 10  # Number of waypoints that define the trajectory
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


def make_env(args: Args, num_envs: int, jax_device: str = "cpu", config: str = "level0.toml") -> VectorEnv:
    """Build the vectorized, fully-wrapped random-trajectory-following environment."""
    config = load_config(Path(__file__).parents[3] / "config" / config)
    env = RandTrajEnv(
        n_samples=10,
        num_envs=num_envs,
        freq=config.env.freq,
        drone_model=config.sim.drone_model,
        physics=config.sim.physics,
        disturbances=config.env.get("disturbances"),
        device=jax_device,
    )
    env = NormalizeActions(env)
    env = StackObs(env, n_obs=args.n_obs)
    env = AngleReward(env, rpy_coef=args.rpy_coef)
    env = ActionPenalty(
        env,
        act_coef=args.act_coef,
        d_act_th_coef=args.d_act_th_coef,
        d_act_xy_coef=args.d_act_xy_coef,
    )
    env = FlattenJaxObservation(env)
    return env
