import pickle
from pathlib import Path
from typing import Any

import fire
import jax.numpy as jnp
import numpy as np
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
from lsy_drone_racing.utils import load_config

CHECKPOINT_DIR = Path(__file__).parents[1] / "checkpoints"


def _latest_checkpoint(task: str) -> Path | None:
    """Return the most recently modified checkpoint for a task, or None if none exist."""
    candidates = list((CHECKPOINT_DIR / task).glob("*.ckpt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main(
    task: str = "single_agent_racing",
    config: str = "level0.toml",
    checkpoint: str | None = None,
    opponent: str = "self_play",
    opponent_pid_speed: float = 1.0,
    opponent_pid_start_frac: float = 0.0,
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
        assert env_config.env.control_mode == "attitude", (
            f"opponent='pid' needs an attitude-control track config, got control_mode="
            f"'{env_config.env.control_mode}'."
        )
        drone_mass = load_params(env_config.sim.physics, env_config.sim.drone_model)["mass"]
        action_space = build_action_space(env_config.env.control_mode, env_config.sim.drone_model)
        traj_pid = build_trajectory_pid(
            start_pos=np.asarray(env_config.env.track.drones[1]["pos"]),
            drone_mass=drone_mass,
            freq=env_config.env.freq,
            control_mode=env_config.env.control_mode,
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
        traj_t = traj_t + opponent_pid_speed / env_config.env.freq
        return env, obs, terminated, truncated, traj_t, i_error

    eval_env, (obs, _) = eval_env.reset(eval_env, seed=args.seed)
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
    eval_env.render()
    done = False

    while not done:
        if traj_pid is not None:
            eval_env, obs, terminated, truncated, traj_t, i_error = policy_step_pid(
                agent, eval_env, jnp.asarray(obs), traj_t, i_error
            )
        else:
            eval_env, obs, terminated, truncated = policy_step(agent, eval_env, jnp.asarray(obs))
        eval_env.render()
        # For multi-drone tasks, terminated/truncated are per-drone -- wait for the whole race
        # (every drone finished/crashed/timed out) instead of stopping as soon as any one drone
        # (e.g. the ego) is done, matching the env's own NEXT_STEP autoreset semantics: the world
        # doesn't actually reset until all drones are settled.
        done = bool(jnp.all(terminated | truncated))
    eval_env.close()


if __name__ == "__main__":
    fire.Fire(main)
