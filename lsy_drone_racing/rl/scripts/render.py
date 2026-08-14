import pickle
import time
from pathlib import Path
from typing import Any

import fire
import jax.numpy as jnp
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params
from flax import nnx
from jax import Array

from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.rl.agents.ppo_agent import Agent
from lsy_drone_racing.rl.tasks import get_task
from lsy_drone_racing.rl.wrappers.trajectory_opponent import (
    SPAWN_TIME_MARGIN,
    build_trajectory_pid,
    teleport_opponents,
)
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper
from lsy_drone_racing.utils import env_param, load_config, strip_env_randomization

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"

# Trajectory-trail colors: drone 0 (the trainable ego) is green, drones 1.. (opponents) are red --
# matching the green/red drone markers drawn in ``wrappers.racing_env``.
_EGO_TRAJ_RGBA = np.array([0.0, 1.0, 0.0, 1.0])
_OPPONENT_TRAJ_RGBA = np.array([1.0, 0.0, 0.0, 1.0])
# Cap on line segments drawn per drone. A full race is ~900 steps at 50 Hz; drawing every point for
# every drone each frame would blow past ``Sim.max_visual_geom`` and slow the viewer, so the trail
# is decimated to at most this many points (recent detail is preserved by keeping the last point).
_MAX_TRAJ_POINTS = 400


def _latest_checkpoint(task: str) -> Path | None:
    """Return the most recently modified checkpoint for a task, or None if none exist."""
    candidates = list((CHECKPOINT_DIR / task).glob("*.ckpt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _record_positions(
    eval_env: Wrapper, alive: np.ndarray, trails: list[list[np.ndarray]]
) -> None:
    """Append world-0 positions to each still-alive drone's trail.

    ``alive[d]`` is False once drone ``d`` has crashed/finished, so its trail freezes at the crash
    point instead of tracing the drone tumbling or falling to the ground after it's done.
    """
    pos = np.asarray(eval_env.unwrapped.data.sim_data.states.pos[0])  # [n_drones, 3]
    for drone, trail in enumerate(trails):
        if alive[drone]:
            trail.append(pos[drone])


def _draw_trajectories(sim: Sim, trails: list[list[np.ndarray]]) -> None:
    """Draw each drone's flown trail into the viewer, green for ego / red for opponents.

    ``trails[d]`` is the list of ``[3]`` positions collected while drone ``d`` was airborne; it
    stops growing once the drone crashes/finishes, so the frozen trail stays visible without the
    jump to the drone's below-ground warp/reset pose. Markers don't persist in the viewer, so this
    must be called before every ``sim.render()``.
    """
    for drone, trail in enumerate(trails):
        if len(trail) < 2:  # Need at least two points to make a line segment.
            continue
        positions = np.asarray(trail)  # [T, 3]
        if len(positions) > _MAX_TRAJ_POINTS:
            idx = np.unique(np.linspace(0, len(positions) - 1, _MAX_TRAJ_POINTS).astype(int))
            positions = positions[idx]
        rgba = _EGO_TRAJ_RGBA if drone == 0 else _OPPONENT_TRAJ_RGBA
        draw_line(sim, positions, rgba=rgba)


def main(
    task: str = "single_agent_racing",
    config: str = "level0.toml",
    checkpoint: str | None = None,
    opponent: str = "self_play",
    opponent_pid_speed: float = 1.0,
    opponent_pid_start_frac: float = 0.0,
    show_trajectory: bool = True,
    no_randomization: bool = False,
    **kwargs: Any,
):
    """Render the simulation for a single trained PPO agent on the given task.

    Args:
        task: Task name (one of the keys in ``lsy_drone_racing.rl.tasks.TASKS``).
        checkpoint: Path to a specific checkpoint to evaluate.
        opponent: For multi-drone tasks, how drones ``1..`` are controlled. ``"self_play"``
            (default) mirrors the loaded ego checkpoint onto every drone, like training's
            self-play eval. ``"pid"`` instead flies them with the scripted waypoint-following
            trajectory PID (``wrappers.trajectory_opponent``), at a fixed (non-random) speed --
            needs ``control_mode == "attitude"``. Ignored for single-drone tasks.
        opponent_pid_speed: Speed multiplier for the PID opponent (1.0 == the nominal ~18s single
            pass). Only used when ``opponent == "pid"``.
        opponent_pid_start_frac: Deterministic fraction of the trajectory at which the PID
            opponent starts (0.0 == the pad, like training's random mid-track spawns but fixed for
            visual inspection; clamped to stay before the last gate). Only used when
            ``opponent == "pid"``.
        show_trajectory: If True (default), draw each drone's flown trail into the viewer -- the
            ego drone (0) in green and the opponent(s) (1..) in red. Set False to disable.
        no_randomization: If True, strip the config's ``[env.randomizations]`` /
            ``[env.disturbances]`` blocks for a clean, deterministic race -- needed so a scripted
            PID opponent (``opponent="pid"``) doesn't crash into randomized gates.
    """
    assert opponent in ("self_play", "pid"), f"Unknown opponent mode '{opponent}'."
    task_spec = get_task(task)
    # Render a single env: the viewer only ever shows world 0, so building the training default
    # (num_envs=1024) would step 1024 mjx envs eagerly per frame (crippling on a laptop / CPU) and
    # make ``done`` fire as soon as *any* of the 1024 ends -- aborting the shown drone mid-track.
    kwargs.setdefault("num_envs", 1)
    args = task_spec.args_cls.create(**kwargs)

    checkpoint_dir = CHECKPOINT_DIR / task

    if not checkpoint:
        latest_checkpoint = _latest_checkpoint(task)
        if not latest_checkpoint:
            raise FileNotFoundError(
                f"No checkpoints found for task '{task}' in {checkpoint_dir}, can't render. Aborting!"
            )
        else:
            print(f"No checkpoint specified, using latest checkpoint: {latest_checkpoint}")
            model_path = latest_checkpoint
    else:
        model_path = CHECKPOINT_DIR / task / checkpoint

    env_config = load_config(Path(__file__).parents[3] / "config" / config)
    if no_randomization:
        strip_env_randomization(env_config)

    eval_env: Wrapper = task_spec.make_env(args, env_config)
    action_dim = int(np.prod(eval_env.single_action_space.shape))
    obs_dim = int(np.prod(eval_env.single_observation_space.shape))

    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(0))
    with open(model_path, "rb") as f:
        nnx.update(agent, pickle.load(f))

    n_drones = eval_env.unwrapped.n_drones
    traj_pid = None
    if opponent == "pid":
        assert n_drones > 1, "opponent='pid' needs a multi-drone task/config."
        assert env_param(env_config, "control_mode") == "attitude", (
            f"opponent='pid' needs an attitude-control track config, got control_mode="
            f"'{env_param(env_config, 'control_mode')}'."
        )
        drone_mass = load_params(env_config.sim.physics, env_config.sim.drone_model)["mass"]
        action_space = build_action_space(
            env_param(env_config, "control_mode"), env_config.sim.drone_model
        )
        traj_pid = build_trajectory_pid(
            start_pos=np.asarray(env_config.env.track.drones[1]["pos"]),
            drone_mass=drone_mass,
            freq=env_param(env_config, "freq"),
            control_mode=env_param(env_config, "control_mode"),
            action_low=np.asarray(action_space.low),
            action_high=np.asarray(action_space.high),
            gates=env_config.env.track.gates,
        )
        print(f"Opponent(s): scripted PID trajectory-follower, speed {opponent_pid_speed}x.")

    @nnx.jit
    def policy_step(agent: Agent, env: Wrapper, obs: Array) -> tuple[Wrapper, Array, Array, Array]:
        """One deterministic (mean-action) env step, compiled so the full wrapper stack fuses.

        ``render()`` stays outside jit -- it mutates the host-side MuJoCo sim/viewer -- but the
        physics + wrapper chain run as a single compiled kernel instead of eager op dispatch.
        """
        mean, _, _ = agent(obs)
        env, (obs, _, terminated, truncated, _) = env.step(env, mean)
        return env, obs, terminated, truncated

    @nnx.jit
    def policy_step_pid(
        agent: Agent, env: Wrapper, obs: Array, traj_t: Array, i_error: Array
    ) -> tuple[Wrapper, Array, Array, Array, Array, Array]:
        """Like ``policy_step``, but drones ``1..`` are driven by ``traj_pid`` instead of ``agent``.

        Reads the opponents' raw ``pos``/``vel``/``quat`` off the unwrapped sim state (bypassing
        the flattened/relative observation, which drops absolute position) -- the same pattern
        ``ippo.py``'s rollout uses to mix in scripted-PID opponents during training.
        """
        mean, _, _ = agent(obs)
        states = env.unwrapped.data.sim_data.states
        opp_pos, opp_vel, opp_quat = states.pos[:, 1:], states.vel[:, 1:], states.quat[:, 1:]
        action_phys, i_error = traj_pid.action(
            opp_pos, opp_vel, opp_quat, traj_t, opponent_pid_speed, i_error
        )
        action = mean.at[:, 1:].set(traj_pid.normalize(action_phys))
        env, (obs, _, terminated, truncated, _) = env.step(env, action)
        traj_t = traj_t + opponent_pid_speed / env_param(env_config, "freq")
        return env, obs, terminated, truncated, traj_t, i_error

    eval_env, (obs, _) = eval_env.reset(eval_env, seed=args.seed)
    sim = eval_env.unwrapped.base_env.sim if show_trajectory else None
    trails: list[list[np.ndarray]] = [[] for _ in range(n_drones)]  # world-0 path per drone
    alive = np.ones(n_drones, dtype=bool)  # a drone's trail freezes once it crashes/finishes
    if sim is not None:
        # The viewer is built lazily with ``max_geom=sim.max_visual_geom`` on the first render, so
        # raise the budget now to fit the decimated trails (plus the per-drone markers) for the
        # whole race -- otherwise ``draw_line`` raises once the trail grows past the default 1000.
        sim.max_visual_geom = max(sim.max_visual_geom, _MAX_TRAJ_POINTS * n_drones + 100)
    traj_t = jnp.zeros((1, n_drones - 1))
    i_error = jnp.zeros((1, n_drones - 1, 3))
    if traj_pid is not None and opponent_pid_start_frac > 0.0:
        # Place the PID opponent(s) mid-track at the requested trajectory fraction, mirroring
        # training's random mid-track spawns but deterministic for visual inspection.
        spawn_t_max = float(traj_pid.gate_times[-1]) - SPAWN_TIME_MARGIN
        traj_t = jnp.full_like(traj_t, min(opponent_pid_start_frac * traj_pid.t_total, spawn_t_max))
        eval_env = teleport_opponents(
            eval_env,
            traj_pid,
            traj_t,
            jnp.full_like(traj_t, opponent_pid_speed),
            jnp.ones_like(traj_t, dtype=bool),
        )
        print(f"PID opponent starts mid-track at t0 = {float(traj_t[0, 0]):.2f}s.")
    if sim is not None:
        _record_positions(eval_env, alive, trails)
        _draw_trajectories(sim, trails)
    eval_env.render()
    done = False

    while not done:
        if traj_pid is not None:
            eval_env, obs, terminated, truncated, traj_t, i_error = policy_step_pid(
                agent, eval_env, jnp.asarray(obs), traj_t, i_error
            )
        else:
            eval_env, obs, terminated, truncated = policy_step(agent, eval_env, jnp.asarray(obs))
        # Per-drone done flags for world 0 (a scalar for single-drone tasks, [n_drones] for multi).
        done_per_drone = np.atleast_1d(np.asarray(terminated | truncated)[0]).reshape(-1)
        # Freeze a drone's trail *before* recording this step: a crashed drone is disabled and
        # warped below ground (pos -> [-1, -1, -1]) on the very step it's marked done, so recording
        # it would draw a big jump from the crash point down to that warp/reset pose.
        alive &= ~done_per_drone
        if sim is not None:
            _record_positions(eval_env, alive, trails)
            _draw_trajectories(sim, trails)
        eval_env.render()
        # For multi-drone tasks, terminated/truncated are per-drone -- wait for the whole race
        # (every drone finished/crashed/timed out) instead of stopping as soon as any one drone
        # (e.g. the ego) is done, matching the env's own NEXT_STEP autoreset semantics: the world
        # doesn't actually reset until all drones are settled.
        done = bool(jnp.all(terminated | truncated))
    # Hold the final frame for a moment so the crash/finish (and trajectory trails) stays visible
    # instead of the window vanishing the instant the race ends. Keep calling render() so the viewer
    # stays responsive and re-draws the (non-persistent) trail markers.
    hold_end = time.monotonic() + 2.0
    while time.monotonic() < hold_end:
        if sim is not None:
            _draw_trajectories(sim, trails)
        eval_env.render()
    eval_env.close()


if __name__ == "__main__":
    fire.Fire(main)
