"""Single-agent drone racing task: env factory + dense in-step reward."""

from pathlib import Path
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from crazyflow.sim.visualize import draw_points
from gymnasium.vector import VectorEnv, VectorWrapper
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.wrappers.observation import FlattenJaxObservation, RelativeRacingObs
from lsy_drone_racing.rl.wrappers.reward import ActionPenalty, AngleReward, ZeroYaw
from lsy_drone_racing.rl.wrappers.takeoff import SpinUpRotors
from lsy_drone_racing.utils import load_config

jp = jnp

# Distance (m) past the gate center, along its +x traversal axis, at which the progress target is
# placed. Shared by the reward and the render visualization so they always agree.
GATE_OFFSET = 0.3


def progress_target(gates_pos: Array, gates_quat: Array, target_gate: Array, gate_offset: float) -> Array:
    """Point just past each env's target gate, along the gate's +x traversal axis.

    Gates are crossed along their local +x axis (see ``gate_passed``), so offsetting the gate
    center along that axis places the target on the *far* side of the gate plane. Aiming progress
    here -- rather than at the gate center -- makes flying through the gate the only way to keep
    closing distance, so the policy cannot park at the gate face.

    Args:
        gates_pos: Gate centers, shape (n_envs, n_gates, 3).
        gates_quat: Gate orientations (xyzw), shape (n_envs, n_gates, 4).
        target_gate: Current target gate index per drone, shape (n_envs, n_drones). -1 (finished)
            wraps to the last gate.
        gate_offset: Offset distance in meters along the gate's +x axis.

    Returns:
        Target points, shape (n_envs, n_drones, 3).
    """
    n_gates = gates_pos.shape[1]
    env_idx = jp.arange(gates_pos.shape[0])[:, None]
    idx = target_gate % n_gates
    gate_pos = gates_pos[env_idx, idx]
    gate_quat = gates_quat[env_idx, idx]
    normal = R.from_quat(gate_quat).as_matrix()[..., :, 0]  # first column = gate's +x in world
    return gate_pos + gate_offset * normal


# Default Args overrides for this task (merged by the CLI before Args.create).
DEFAULTS: dict[str, Any] = {
    "total_timesteps": 50_000_000,
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "target_kl": 0.03,
    "update_epochs": 4,
    "clip_coef": 0.2,
    "ent_coef": 0.025,
    "progress_coef": 4.0,
    "rpy_coef": 0.01,
    "d_act_xy_coef": 0.1,
    "d_act_th_coef": 0.1,
    "act_coef": 0.005,
    "gate_bonus": 20.0,
    "finish_bonus": 30.0,
    "crash_penalty": 2.0,
}


def build_racing_reward(
    progress_coef: float = 1.0,
    gate_bonus: float = 2.0,
    finish_bonus: float = 10.0,
    crash_penalty: float = 5.0,
    gate_offset: float = GATE_OFFSET,
) -> Callable[[Any, Any], Array]:
    """Build a dense racing reward to be compiled into the env step.

    The reward is computed inside the env (via ``reward_fn``) so it can use the *true* gate
    positions, which the observation only reveals once a gate is sensed. It combines:

    * progress: reduction in Euclidean distance to a point just *past* the current target gate
      (the gate center pushed ``gate_offset`` along the gate's traversal axis), measured against
      the gate that was the target at the *start* of the step. Aiming past the gate -- rather than
      at its center -- means flying through it is the only way to keep collecting progress, so the
      policy cannot park at the gate face for ~zero reward,
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
        gate_offset: Distance (m) past the gate center, along its +x traversal axis, at which the
            progress target is placed. Gates are crossed along their local +x axis (see
            ``gate_passed``), so this pulls the drone through the gate plane.
    """

    def reward(data: Any, prev_data: Any) -> Array:
        # Progress target sits just past the gate that was the target at the start of the step.
        target = progress_target(data.gates_pos, data.gates_quat, prev_data.target_gate, gate_offset)
        dist_prev = jp.linalg.norm(prev_data.sim_data.states.pos - target, axis=-1)
        dist_curr = jp.linalg.norm(data.sim_data.states.pos - target, axis=-1)
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


class DrawProgressTarget(VectorWrapper):
    """Draw each env's current progress target (the point the reward pulls the drone toward).

    Debug visualization only: on ``render`` it marks the through-gate target for the current
    target gate of every env, so the target side relative to the gate can be confirmed visually.
    """

    def __init__(self, env: VectorEnv, gate_offset: float = GATE_OFFSET):
        """Init."""
        super().__init__(env)
        self.gate_offset = gate_offset

    def render(self):
        """Draw the progress target(s) then delegate to the underlying render."""
        base = self.env.unwrapped
        targets = progress_target(
            base.data.gates_pos, base.data.gates_quat, base.data.target_gate, self.gate_offset
        )
        points = np.asarray(targets).reshape(-1, 3)
        draw_points(base.sim, points, rgba=np.array([1.0, 0.0, 1.0, 1.0]), size=0.04)
        return self.env.render()


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
        device=jax_device,
        reward_fn=reward_fn,
    )
    # Seed warm rotors on every (auto)reset so the grounded drone can actually take off.
    env = SpinUpRotors(env)
    env = NormalizeActions(env)
    env = AngleReward(env, rpy_coef=args.rpy_coef)
    env = ActionPenalty(
        env,
        act_coef=args.act_coef,
        d_act_th_coef=args.d_act_th_coef,
        d_act_xy_coef=args.d_act_xy_coef,
    )
    # Zero yaw outside ActionPenalty so the redundant yaw DOF is excluded from the action
    # penalty and last_action (yaw is unused for this yaw-symmetric racing task).
    env = ZeroYaw(env)
    # Relative geometry + next-2-gates + rotation matrices (must come after ActionPenalty so
    # last_action is present in the dict it transforms).
    env = RelativeRacingObs(env)
    env = FlattenJaxObservation(env)
    # Debug visualization of the through-gate progress target (drawn only when rendering).
    env = DrawProgressTarget(env, gate_offset=GATE_OFFSET)
    return env
