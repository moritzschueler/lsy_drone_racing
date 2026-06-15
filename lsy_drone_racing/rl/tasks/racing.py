"""Single-agent drone racing task: env factory + dense in-step reward."""

from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.vector import VectorEnv
from jax import Array

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.wrappers.observation import FlattenJaxObservation, StackObs
from lsy_drone_racing.rl.wrappers.reward import ActionPenalty, AngleReward
from lsy_drone_racing.utils import load_config

jp = jnp

# Default Args overrides for this task (merged by the CLI before Args.create).
DEFAULTS: dict[str, Any] = {"total_timesteps": 50_000_000}


def build_racing_reward(
    progress_coef: float = 1.0,
    gate_bonus: float = 2.0,
    finish_bonus: float = 10.0,
    crash_penalty: float = 5.0,
) -> Callable[[Any, Any], Array]:
    """Build a dense racing reward to be compiled into the env step.

    The reward is computed inside the env (via ``reward_fn``) so it can use the *true* gate
    positions, which the observation only reveals once a gate is sensed. It combines:

    * progress: reduction in Euclidean distance to the current target gate's center,
      measured against the gate that was the target at the *start* of the step, so passing a
      gate does not cause a spurious negative jump,
    * gate_bonus: a one-off bonus each time the target gate advances,
    * finish_bonus: a large one-off bonus when the final gate is passed (target_gate -> -1),
    * crash_penalty: a penalty when the drone is disabled without finishing (out of bounds
      or collision). Timeouts (truncation) are not penalized.

    Reward is zeroed on auto-reset transition steps, where prev/post state straddle an
    episode boundary.

    Args:
        progress_coef: Weight on the per-step distance-progress term.
        gate_bonus: Bonus added when the target gate advances.
        finish_bonus: Bonus added when the whole track is completed.
        crash_penalty: Penalty subtracted on a crash (collision / out of bounds).
    """

    def reward(data: Any, prev_data: Any) -> Array:
        n_gates = data.gates_pos.shape[1]
        env_idx = jp.arange(data.gates_pos.shape[0])[:, None]
        # Distance to the gate that was the target during this step (true positions).
        gate_pos = data.gates_pos[env_idx, prev_data.target_gate % n_gates]
        dist_prev = jp.linalg.norm(prev_data.sim_data.states.pos - gate_pos, axis=-1)
        dist_curr = jp.linalg.norm(data.sim_data.states.pos - gate_pos, axis=-1)
        progress = dist_prev - dist_curr

        active = prev_data.target_gate != -1  # episode was not already finished
        passed_gate = (data.target_gate != prev_data.target_gate) & active
        finished = (data.target_gate == -1) & active
        crashed = data.disabled_drones & ~prev_data.disabled_drones & (data.target_gate != -1)

        r = progress_coef * progress
        r += gate_bonus * passed_gate
        r += finish_bonus * finished
        r -= crash_penalty * crashed
        # Auto-reset transition steps compare across an episode boundary -> zero them out.
        return jp.where(prev_data.marked_for_reset[:, None], 0.0, r)

    return reward


def make_env(args: Args, num_envs: int, jax_device: str = "cpu", config: str = "level0.toml") -> VectorEnv:
    """Build the vectorized, fully-wrapped racing environment."""
    config = load_config(Path(__file__).parents[3] / "config" / config)
    reward_fn = build_racing_reward(
        progress_coef=args.progress_coef,
        gate_bonus=args.gate_bonus,
        finish_bonus=args.finish_bonus,
        crash_penalty=args.crash_penalty,
    )
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
        reward_fn=reward_fn,
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
