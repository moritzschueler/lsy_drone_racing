"""Self-contained Crazyflow quadrotor environments + wrappers (hovering & figure-8 tracking).

Standalone JAX/Flax module: jittable, vectorised drone envs as frozen PyTreeNodes with pure
reset/step callables that take the env as first arg. Usage::

    env, (obs, info) = env.reset(env, seed=0)
    env, (obs, reward, terminated, truncated, info) = env.step(env, action)
"""

from __future__ import annotations

import os

# Must be set before scipy is imported so scipy Rotations operate on JAX arrays.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import functools
from typing import Any, Callable

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


def create_action_space(control_type: Control | str, drone_model: str) -> spaces.Box:
    """Attitude action space [roll, pitch, yaw, thrust] for the given drone model."""
    if control_type != Control.attitude:
        raise ValueError(f"Only attitude control is supported, got {control_type}")
    params = ForceTorqueParams.load(drone_model)
    thrust_min, thrust_max = params.thrust_min * 4, params.thrust_max * 4
    return spaces.Box(
        np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, thrust_min], dtype=np.float32),
        np.array([np.pi / 2, np.pi / 2, np.pi / 2, thrust_max], dtype=np.float32),
    )


# region Envs


class DroneEnv(struct.PyTreeNode):
    """Base jittable drone environment (frozen PyTreeNode passed into jitted functions)."""

    # Sim object for rendering
    sim: Sim = struct.field(pytree_node=False)
    # Constant environment parameters
    num_envs: int = struct.field(pytree_node=False)
    max_episode_time: float = struct.field(pytree_node=False)
    physics: Physics = struct.field(pytree_node=False)
    control: str = struct.field(pytree_node=False)
    drone_model: str = struct.field(pytree_node=False)
    freq: int = struct.field(pytree_node=False)
    device: str = struct.field(pytree_node=False)
    single_action_space: spaces.Box = struct.field(pytree_node=False)
    action_space: spaces.Box = struct.field(pytree_node=False)
    single_observation_space: spaces.Dict = struct.field(pytree_node=False)
    observation_space: spaces.Dict = struct.field(pytree_node=False)
    n_substeps: int = struct.field(pytree_node=False)

    # Variable simulation data
    data: SimData = struct.field(pytree_node=True)
    steps: Array = struct.field(pytree_node=True)
    _marked_for_reset: Array = struct.field(pytree_node=True)

    #  functions
    reset: Callable = struct.field(pytree_node=False)
    step: Callable = struct.field(pytree_node=False)

    # Non-jittable functions
    def render(self):
        self.sim.data = self.data
        self.sim.render(world=0)

    def close(self):
        self.sim.close()

    @property
    def unwrapped(self) -> struct.PyTreeNode:
        return self


class ReachPosEnv(DroneEnv):
    """Hovering / reach-position task."""

    autoreset_mode: AutoresetMode = struct.field(pytree_node=False)

    # Reach position specific attributes
    goal_pos: Array = struct.field(pytree_node=True)

    # Non-jittable functions
    def render(self, world: int = 0) -> None:
        draw_points(
            self.sim,
            self.goal_pos[None, world],
            rgba=jp.array([1.0, 0, 0, 1.0]),
            size=0.01,
        )
        self.sim.data = self.data
        self.sim.render(world=world)

    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        max_episode_time: float = 5.0,
        drone_model: str = "cf21B_500",
        freq: int = 500,
        sim_freq: int = 500,
        device: str = "cpu",
        pos_min: Array = jp.array([-1.0, -1.0, 1.0]),
        pos_max: Array = jp.array([1.0, 1.0, 2.0]),
        goal_pmin: Array = jp.array([-1.0, -1.0, 0.5]),
        goal_pmax: Array = jp.array([1.0, 1.0, 1.5]),
        ang_vel_min: Array = jp.zeros(3),
        ang_vel_max: Array = jp.zeros(3),
        vel_min: float = -1.0,
        vel_max: float = 1.0,
        reward_dist_scale: float = 6.0,
        reward_angvel_weight: float = 0.0,
    ) -> ReachPosEnv:
        """Create a jittable ReachPosEnv."""
        # Initialize the simulation
        jax_device = jax.devices(device)[0]
        sim = Sim(
            n_worlds=num_envs,
            n_drones=1,
            drone_model=drone_model,
            physics=Physics.first_principles,
            control=Control.attitude,
            device=device,
            freq=sim_freq,
        )

        # Spin up rotors to the cf21B_500 hover RPM on reset.
        def _reset_rotor(data: SimData, mask: Array) -> SimData:
            rotor_vel = 15900.0 * jp.ones(
                (
                    data.core.n_worlds,
                    data.core.n_drones,
                    data.states.rotor_vel.shape[-1],
                )
            )
            return data.replace(
                states=leaf_replace(data.states, mask, rotor_vel=rotor_vel)
            )

        def _reset_randomization(
            data: SimData,
            mask: Array,
            pmin: Array,
            pmax: Array,
            vmin: float,
            vmax: float,
            wmin: Array,
            wmax: Array,
        ) -> SimData:
            shape = (data.core.n_worlds, data.core.n_drones, 3)
            key, pos_key, vel_key, ang_vel_key = jax.random.split(data.core.rng_key, 4)
            data = data.replace(core=data.core.replace(rng_key=key))
            pos = jax.random.uniform(key=pos_key, shape=shape, minval=pmin, maxval=pmax)
            vel = jax.random.uniform(key=vel_key, shape=shape, minval=vmin, maxval=vmax)
            ang_vel = jax.random.uniform(
                key=ang_vel_key, shape=shape, minval=wmin, maxval=wmax
            )
            data = data.replace(
                states=leaf_replace(
                    data.states, mask, pos=pos, vel=vel, ang_vel=ang_vel
                )
            )
            return data

        reset_randomization = functools.partial(
            _reset_randomization,
            pmin=pos_min,
            pmax=pos_max,
            vmin=vel_min,
            vmax=vel_max,
            wmin=ang_vel_min,
            wmax=ang_vel_max,
        )
        sim.reset_pipeline += (reset_randomization, _reset_rotor)
        sim.build_reset_fn()

        # Prepare immutable constants
        single_action_space = create_action_space(Control.attitude, sim.drone_model)
        action_space = batch_space(single_action_space, sim.n_worlds)
        single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        n_substeps = sim.freq // freq

        observation_space = batch_space(single_observation_space, sim.n_worlds)

        # Build jittable functions
        def _sanitize_action_STE(action: Array, low: Array, high: Array) -> Array:
            action_clipped = jp.clip(action, low, high)
            action = action + jax.lax.stop_gradient(action_clipped - action)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _obs(goal_pos: Array, data: SimData) -> dict[str, Array]:
            obs = {
                "pos": data.states.pos[:, 0, :],
                "quat": data.states.quat[:, 0, :],
                "vel": data.states.vel[:, 0, :],
                "ang_vel": data.states.ang_vel[:, 0, :],
            }
            obs["pos"] = (
                data.states.pos[:, 0, :] - goal_pos
            )  # agent only sees relative position
            return obs

        def _sample_goal(key: Array, goal: Array, mask: Array | None) -> Array:
            new_goal = jax.random.uniform(
                key, shape=goal.shape, minval=goal_pmin, maxval=goal_pmax
            )
            if mask is not None:
                new_goal = jp.where(mask[..., None], new_goal, goal)
            return new_goal

        def _reset(
            env: ReachPosEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[tuple[SimData, Array, Array], tuple[dict[str, Array], dict]]:
            data = env.data
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            rng_key, subkey = jax.random.split(data.core.rng_key)
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            goal_pos = _sample_goal(subkey, env.goal_pos, None)
            data = sim._reset(data, sim.default_data, None)
            _marked_for_reset = env._marked_for_reset.at[...].set(False)
            env = env.replace(
                data=data,
                _marked_for_reset=_marked_for_reset,
                goal_pos=goal_pos,
            )
            return env, (
                _obs(goal_pos, data),
                {"success": jp.zeros((num_envs,), dtype=bool)},
            )

        def _reward(
            terminated: Array,
            pos: Array,
            goal: Array,
            success: Array,
            ang_vel: Array,
        ) -> Array:
            norm_distance = jp.linalg.norm(pos - goal, axis=-1)
            reward = jp.exp(-reward_dist_scale * norm_distance)
            # Damp swaying: penalise body angular velocity (rate), not just the tilt angle.
            reward = reward - reward_angvel_weight * jp.linalg.norm(ang_vel, axis=-1)
            reward = jp.where(success, 100.0, reward)
            reward = jp.where(terminated & ~success, -1.0, reward)
            return reward

        def _terminated(env: ReachPosEnv, sim_data: SimData) -> tuple[Array, Array]:
            pos = sim_data.states.pos[:, 0, :]
            lower_bounds = jp.array([-4.0, -4.0, 0.0])
            upper_bounds = jp.array([4.0, 4.0, 4.0])
            out_of_bounds = jp.any((pos < lower_bounds) | (pos > upper_bounds), axis=-1)
            at_goal = jp.linalg.norm(pos - env.goal_pos, axis=-1) < 0.05
            at_rest = jp.linalg.norm(sim_data.states.vel[:, 0, :], axis=-1) < 0.05
            return out_of_bounds | (at_goal & at_rest), (at_goal & at_rest)

        def _truncated(time: Array, max_episode_time: float) -> Array:
            return time >= max_episode_time

        def _done(terminated: Array, truncated: Array) -> Array:
            return terminated | truncated

        def _apply_action(data: SimData, action: Array) -> SimData:
            action = _sanitize_action_STE(action, action_space.low, action_space.high)
            return data.replace(
                controls=data.controls.replace(
                    attitude=data.controls.attitude.replace(staged_cmd=action)
                )
            )

        def _step(
            env: ReachPosEnv, action: Array
        ) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
            data = env.data
            data = _apply_action(data, action)
            data = sim._step(data, n_substeps)
            # termination/reward computed on pre-reset state
            sim_time = data.core.steps / data.core.freq
            terminated, success = _terminated(env, data)
            truncated = _truncated(sim_time[..., 0], max_episode_time)
            done = _done(terminated, truncated)
            # SAME_STEP autoreset resets done envs in-step, so the true final state is stashed in
            # info["final_obs"] for the replay buffer to bootstrap truncated transitions from.
            pos_pre = data.states.pos[:, 0, :]
            final_obs = _obs(env.goal_pos, data)
            final_rotor_vel = data.states.rotor_vel[:, 0, :]
            episode_steps = (data.core.steps[:, 0] // (sim.freq // freq)).astype(
                jp.int32
            )
            ang_vel_pre = data.states.ang_vel[:, 0, :]
            reward = _reward(terminated, pos_pre, env.goal_pos, success, ang_vel_pre)
            # autoreset done envs: resample goal, reset sim
            rng_key, subkey = jax.random.split(data.core.rng_key)
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            goal_pos = _sample_goal(subkey, env.goal_pos, done)
            data = sim._reset(data, sim.default_data, done)
            steps = data.core.steps // (sim.freq // freq)
            obs = _obs(goal_pos, data)
            env = env.replace(
                data=data,
                steps=steps,
                _marked_for_reset=jp.zeros_like(done),
                goal_pos=goal_pos,
            )
            info = {
                "final_obs": final_obs,
                "final_rotor_vel": final_rotor_vel,
                "success": success,
                "episode_steps": episode_steps,
            }
            return env, (obs, reward, terminated, truncated, info)

        # Initialize reset mask and step count
        steps = jp.zeros((num_envs, 1), dtype=jp.int32, device=jax_device)
        _marked_for_reset = jp.zeros((num_envs,), dtype=jp.bool_, device=jax_device)

        return cls(
            sim=sim,
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=Physics.first_principles,
            control=Control.attitude,
            drone_model=drone_model,
            freq=freq,
            device=device,
            single_action_space=single_action_space,
            action_space=action_space,
            single_observation_space=single_observation_space,
            observation_space=observation_space,
            n_substeps=n_substeps,
            autoreset_mode=AutoresetMode.SAME_STEP,
            goal_pos=jp.zeros((sim.n_worlds, 3), dtype=jp.float32, device=jax_device),
            data=sim.data,
            steps=steps,
            _marked_for_reset=_marked_for_reset,
            reset=jax.jit(_reset),
            step=jax.jit(_step),
        )


class FigureEightEnv(DroneEnv):
    """Figure-8 trajectory-tracking task."""

    autoreset_mode: AutoresetMode = struct.field(pytree_node=False)

    # Immutable figure-eight parameters
    trajectories: Array = struct.field(pytree_node=False)
    trajectory_vel: Array = struct.field(pytree_node=False)
    sample_offsets: Array = struct.field(pytree_node=False)

    @property
    def target_pos(self) -> Array:
        """Current trajectory target position for all envs. Shape: (num_envs, 3)."""
        return self.trajectories[
            jp.arange(self.trajectories.shape[0])[:, None], self.steps
        ][:, 0, :]

    @property
    def target_vel(self) -> Array:
        """Current trajectory target velocity for all envs. Shape: (num_envs, 3)."""
        return self.trajectory_vel[
            jp.arange(self.trajectory_vel.shape[0])[:, None], self.steps
        ][:, 0, :]

    # Non-jittable functions
    def render(self, world: int = 0) -> None:
        idx = self.steps + self.sample_offsets[None, ...] % self.trajectories.shape[1]
        next_trajectory = self.trajectories[
            jp.arange(self.trajectories.shape[0])[:, None], idx
        ]
        trajectories = np.array(self.trajectories)
        next_trajectory = np.array(next_trajectory)
        draw_line(
            self.sim,
            trajectories[world, 0:-1:8, :],
            rgba=jp.array([1, 1, 1, 0.4]),
            start_size=2.0,
            end_size=2.0,
        )
        draw_line(
            self.sim,
            next_trajectory[world],
            rgba=jp.array([1, 0, 0, 1]),
            start_size=3.0,
            end_size=3.0,
        )
        draw_points(
            self.sim, next_trajectory[world], rgba=jp.array([1.0, 0, 0, 1]), size=0.01
        )
        current_target = self.trajectories[
            world, self.steps[world] % self.trajectories.shape[1]
        ]
        draw_points(
            self.sim, current_target, rgba=jp.array([0.0, 1.0, 0.0, 0.4]), size=0.02
        )
        self.sim.data = self.data
        self.sim.render(world=world)

    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        drone_model: str = "cf21B_500",
        freq: int = 500,
        sim_freq: int = 500,
        state_freq: int = 100,
        attitude_freq: int = 500,
        force_torque_freq: int = 500,
        device: str = "cpu",
        n_samples: int = 10,
        trajectory_time: float = 10.0,
        samples_dt: float = 0.1,
        reset_randomization: Callable[[SimData, Array], SimData] = None,
    ) -> "FigureEightEnv":
        """Create a jittable FigureEightEnv."""
        # Initialize the simulation
        jax_device = jax.devices(device)[0]
        sim = Sim(
            n_worlds=num_envs,
            n_drones=1,
            drone_model=drone_model,
            physics=Physics.first_principles,
            control=Control.attitude,
            device=device,
            freq=sim_freq,
            state_freq=state_freq,
            attitude_freq=attitude_freq,
            force_torque_freq=force_torque_freq,
        )

        # Create the figure eight trajectory
        sample_offsets = jp.array(jp.arange(n_samples) * freq * samples_dt, dtype=int)
        # enough steps for max episode time plus the future-observation lookahead
        max_sample_idx = sample_offsets[-1] if n_samples > 0 else 0
        n_steps = int(max_episode_time * freq + max_sample_idx)
        sample_time = max_sample_idx / freq
        n_loops = max_episode_time / trajectory_time + sample_time / trajectory_time
        ts = jp.linspace(0, 2 * jp.pi * n_loops, n_steps)[None, :]
        offset = jp.linspace(0, 2 * jp.pi, num_envs, endpoint=False)
        ts += offset[:, None]
        radius = 1
        height_offset = 1.25
        a = radius * jp.sin(ts)
        b = radius / 2 * jp.sin(2 * ts)
        da = radius * jp.cos(ts) * (2 * jp.pi / trajectory_time)
        db = radius * jp.cos(2 * ts) * (2 * jp.pi / trajectory_time)
        trajectories = jp.zeros((ts.shape[0], ts.shape[1], 3))
        trajectories = trajectories.at[..., 0].set(a)
        trajectories = trajectories.at[..., 2].set(b)
        trajectories = trajectories.at[..., 2].add(height_offset)
        trajectory_vel = jp.zeros((ts.shape[0], ts.shape[1], 3))
        trajectory_vel = trajectory_vel.at[..., 0].set(da)
        trajectory_vel = trajectory_vel.at[..., 2].set(db)

        # Spin up rotors to the cf21B_500 hover RPM on reset.
        def _reset_rotor(data: SimData, mask: Array) -> SimData:
            rotor_vel = 15900.0 * jp.ones(
                (
                    data.core.n_worlds,
                    data.core.n_drones,
                    data.states.rotor_vel.shape[-1],
                )
            )
            return data.replace(
                states=leaf_replace(data.states, mask, rotor_vel=rotor_vel)
            )

        def _reset_velocity(data: SimData, mask: Array, ref_vel: Array) -> SimData:
            data = data.replace(states=leaf_replace(data.states, mask, vel=ref_vel))
            return data

        reset_velocity_fn = functools.partial(
            _reset_velocity, ref_vel=trajectory_vel[:, 0:1, :]
        )

        sim.reset_pipeline += (reset_randomization, _reset_rotor, reset_velocity_fn)
        sim.build_reset_fn()

        # Prepare immutable constants
        single_action_space = create_action_space(Control.attitude, sim.drone_model)
        action_space = batch_space(single_action_space, sim.n_worlds)
        single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        n_substeps = sim.freq // freq

        # Set takeoff position and build default reset position
        takeoff_pos = trajectories[:, :1, :]
        sim.data = sim.data.replace(states=sim.data.states.replace(pos=takeoff_pos))
        sim.build_default_data()

        # Update observation space
        spec = {k: v for k, v in single_observation_space.items()}
        # use Python floats for infinity (compatible with gym spaces)
        spec["local_samples"] = spaces.Box(
            -float("inf"), float("inf"), shape=(3 * n_samples,)
        )
        single_observation_space = spaces.Dict(spec)
        observation_space = batch_space(single_observation_space, sim.n_worlds)

        # Build jittable functions
        def _sanitize_action(action: Array, low: Array, high: Array) -> Array:
            action = jp.clip(action, low, high)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _aux_obs(
            trajectories: Array, steps: Array, pos: Array, sample_offsets: Array
        ) -> dict[str, Array]:
            idx = (steps + sample_offsets[None, ...]) % trajectories.shape[
                1
            ]  # wrap for cyclic
            dpos = trajectories[jp.arange(trajectories.shape[0])[:, None], idx] - pos
            local_samples = dpos.reshape(dpos.shape[0], dpos.shape[1] * dpos.shape[2])
            return local_samples

        def _obs(data: SimData) -> dict[str, Array]:
            obs = {
                "pos": data.states.pos[:, 0, :],
                "quat": data.states.quat[:, 0, :],
                "vel": data.states.vel[:, 0, :],
                "ang_vel": data.states.ang_vel[:, 0, :],
            }
            steps = data.core.steps // (sim.freq // freq)
            obs["local_samples"] = _aux_obs(
                trajectories, steps, data.states.pos, sample_offsets
            )
            return obs

        def _reset(
            env: FigureEightEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[tuple[SimData, Array, Array], tuple[dict[str, Array], dict]]:
            data = env.data
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            data = sim._reset(data, sim.default_data, None)
            _marked_for_reset = env._marked_for_reset.at[...].set(False)
            return env.replace(data=data, _marked_for_reset=_marked_for_reset), (
                _obs(data),
                {},
            )

        def _reward(terminated: Array, pos: Array, goal: Array) -> Array:
            norm_distance = jp.linalg.norm(pos - goal, axis=-1)
            reward = jp.exp(-2.0 * norm_distance)
            reward += jp.where(terminated, -1.0, 0.0)
            return reward

        def _terminated(pos: Array) -> Array:
            lower_bounds = jp.array([-4.0, -4.0, 0.0])
            upper_bounds = jp.array([4.0, 4.0, 4.0])
            terminate = jp.any(
                (pos[:, 0, :] < lower_bounds) | (pos[:, 0, :] > upper_bounds), axis=-1
            )
            return terminate

        def _truncated(time: Array, max_episode_time: float) -> Array:
            return time >= max_episode_time

        def _done(terminated: Array, truncated: Array) -> Array:
            return terminated | truncated

        def _apply_action(data: SimData, action: Array) -> SimData:
            action = _sanitize_action(action, action_space.low, action_space.high)
            return data.replace(
                controls=data.controls.replace(
                    attitude=data.controls.attitude.replace(staged_cmd=action)
                )
            )

        sim_step_fn = sim.build_step_fn()
        sim_reset_fn = sim.build_reset_fn()
        default_data = sim.build_default_data()

        def _step(
            env: FigureEightEnv, action: Array
        ) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
            data = env.data
            data = _apply_action(data, action)
            data = sim_step_fn(data, n_substeps)
            # termination/reward computed on pre-reset state
            sim_time = data.core.steps / data.core.freq
            terminated = _terminated(data.states.pos)
            truncated = _truncated(sim_time[..., 0], max_episode_time)
            done = _done(terminated, truncated)
            steps_pre = data.core.steps // (sim.freq // freq)
            pos_pre = data.states.pos[:, 0, :]
            goal_pre = trajectories[
                jp.arange(trajectories.shape[0])[:, None], steps_pre
            ][:, 0, :]
            final_obs = _obs(data)
            final_rotor_vel = data.states.rotor_vel[:, 0, :]
            data = sim_reset_fn(data, default_data, done)  # autoreset done envs in-step
            steps = data.core.steps // (sim.freq // freq)
            info = {"final_obs": final_obs, "final_rotor_vel": final_rotor_vel}
            env = env.replace(
                data=data, steps=steps, _marked_for_reset=jp.zeros_like(done)
            )
            reward = _reward(terminated, pos_pre, goal_pre)
            return env, (_obs(data), reward, terminated, truncated, info)

        # Initialize reset mask and step count
        steps = jp.zeros((num_envs, 1), dtype=jp.int32, device=jax_device)
        _marked_for_reset = jp.zeros((num_envs,), dtype=jp.bool_, device=jax_device)

        return cls(
            sim=sim,
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=Physics.first_principles,
            control=Control.attitude,
            drone_model=drone_model,
            freq=freq,
            device=device,
            single_action_space=single_action_space,
            action_space=action_space,
            single_observation_space=single_observation_space,
            observation_space=observation_space,
            n_substeps=n_substeps,
            autoreset_mode=AutoresetMode.SAME_STEP,
            trajectories=trajectories,
            trajectory_vel=trajectory_vel,
            sample_offsets=sample_offsets,
            data=sim.data,
            steps=steps,
            _marked_for_reset=_marked_for_reset,
            reset=jax.jit(_reset),
            step=jax.jit(_step),
        )


# region Wrappers


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
