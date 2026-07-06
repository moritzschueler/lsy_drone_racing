"""Hovering task: drive the drone to and hold a fixed goal position."""

from pathlib import Path
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.sim.visualize import draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from jax import Array
from ml_collections import ConfigDict

from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn, rng_spec2fn
from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.wrappers.observation import FlattenJaxObservation, StackObs
from lsy_drone_racing.rl.wrappers.reward import ActionPenalty, AngleReward
from lsy_drone_racing.utils import load_config


class HoverEnv(DroneEnv):
    """Drone environment for hovering at a fixed goal position.

    The observation is augmented with the position error to the hover goal, and the reward
    is based on the distance to that goal.
    """

    def __init__(
        self,
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
        """Initialize the hovering environment.

        Args:
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

        # Set takeoff position and build default reset position
        self.takeoff_pos = np.array([-1.5, 1.0, 0.07])
        self.hover_goal = np.array([-1.5, 1.0, 2.0])
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
        spec["goal_error"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Reset."""
        super().reset(seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        self._reset(options=options)  # call jax rest function
        self._marked_for_reset = self._marked_for_reset.at[...].set(False)
        return self.obs(), {}

    def render(self):
        """Render."""
        draw_points(
            self.sim, np.array([self.hover_goal]), rgba=np.array([1.0, 0.0, 0.0, 1.0]), size=0.03
        )
        self.sim.render()

    def obs(self) -> dict[str, Array]:
        """Observations."""
        obs = super().obs()
        goal_error = self.hover_goal - obs["pos"]
        obs["goal_error"] = goal_error
        return obs

    def reward(self) -> Array:
        """Rewards."""
        obs = self.obs()
        pos = obs["pos"]
        goal = self.hover_goal
        # distance to goal
        norm_distance = jnp.linalg.norm(pos - goal, axis=-1)
        reward = 1.0 / (1.0 + norm_distance)
        reward += jnp.exp(-5.0 * norm_distance)
        reward = jnp.where(self.terminated(), -5.0, reward)
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
        lower_bounds = jnp.array([-4.0, -4.0, -0.0])
        upper_bounds = jnp.array([4.0, 4.0, 4.0])
        terminate = jnp.any((pos[:, 0, :] < lower_bounds) | (pos[:, 0, :] > upper_bounds), axis=-1)
        return terminate

    def build_reset_randomization_fn(self, physics: str) -> Callable[[SimData, Array], SimData]:
        """Reset randomization."""

        # Spin up rotors to help takeoff
        def _reset_randomization_so_rpy(data: SimData, mask: Array) -> SimData:
            rotor_vel = 0.05 * jnp.ones(
                (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
            )
            data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
            return data

        def _reset_randomization_first_principles(data: SimData, mask: Array) -> SimData:
            rotor_vel = 10000.0 * jnp.ones(
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


def make_env(
    args: Args, num_envs: int, jax_device: str = "cpu", config: str = "level0.toml"
) -> VectorEnv:
    """Build the vectorized, fully-wrapped hovering environment."""
    config = load_config(Path(__file__).parents[3] / "config" / config)
    env = HoverEnv(
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
