"""Single-agent drone racing task: env factory + dense in-step reward."""

from pathlib import Path
from typing import Any, Callable

import jax
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
from lsy_drone_racing.rl.wrappers.segment_spawn import DrawSpawnPoints, SegmentSpawn
from lsy_drone_racing.rl.wrappers.takeoff import SpinUpRotors
from lsy_drone_racing.utils import load_config

jp = jnp

# Half-extent (m) of the square gate opening, used by the dense progress reward as the cuboid
# corridor through which *any* crossing point is equally good. Matches the gate_size passed to
# gate_passed in race_core's _update_target_gates ((0.45, 0.45) -> 0.225 half-extent); keep the two
# in sync so the dense reward's notion of "inside the opening" agrees with the env's pass detection.
GATE_HALF_EXTENT = 0.225

# Default Args overrides for this task (merged by the CLI before Args.create).
RACING_CONFIG: dict[str, Any] = {
    "total_timesteps": 50_000_000,
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "target_kl": 0.03,
    "update_epochs": 4,
    "clip_coef": 0.2,
    "ent_coef": 0.01,
    "anneal_ent_coef": True,  # decay entropy bonus to 0 so the policy sharpens late (was growing more stochastic)
    "progress_coef": 3,
    "rpy_coef": 0.001,
    "d_act_xy_coef": 0.001,
    "d_act_th_coef": 0.001,
    "act_coef": 0.001,
    "gate_bonus": 15.0,
    "finish_bonus": 30.0,
    "crash_penalty": 2.0,
    "timeout_penalty": 5.0, # Penalize timeout instead of finishing
    "num_steps": 128,
    # Early stopping params. Patience floors (absolute iters) below; the *_frac scale them to the
    # horizon (fraction of num_iterations, whichever is larger) so behavior is consistent whether the
    # run is 10M or 50M. arm_frac keeps both kill-switches inert until the cone curriculum (kappa)
    # has finished ramping at tau=0.5, so the run can't be cut off mid-curriculum.
    "early_stop_patience": 20,           # floor; ~38 iters at 50M via the frac below
    "early_stop_patience_frac": 0.10,
    "kl_floor": 5e-4,                    # approx_kl below this (healthy ~2e-3) signals a stalled policy ...
    "kl_floor_patience": 20,             # ... floor; scaled by the frac below
    "kl_floor_patience_frac": 0.10,
    "early_stop_arm_frac": 0.5,          # don't arm kill-switches until the kappa ramp completes
}


def _target_gate_frame(
    drone_pos: Array, gates_pos: Array, gates_quat: Array, target_gate: Array
) -> tuple[Array, Array, Array]:
    """Locate each drone's target gate and express the drone offset in that gate's frame.

    Args:
        drone_pos: Drone positions, shape (n_envs, n_drones, 3).
        gates_pos: Gate centers, shape (n_envs, n_gates, 3).
        gates_quat: Gate orientations (xyzw), shape (n_envs, n_gates, 4).
        target_gate: Current target gate index per drone, shape (n_envs, n_drones). -1 (finished)
            wraps to the last gate.

    Returns:
        ``(gate_pos, rot, local)``: the target gate center (n_envs, n_drones, 3), the gate's
        rotation matrix gate->world (n_envs, n_drones, 3, 3), and the drone position in the gate
        frame (n_envs, n_drones, 3) -- local x is the along-axis (traversal) coordinate, local y/z
        span the gate opening.
    """
    n_gates = gates_pos.shape[1]
    env_idx = jp.arange(gates_pos.shape[0])[:, None]
    idx = target_gate % n_gates  # -1 (finished) indicates the last gate
    gate_pos = gates_pos[env_idx, idx]  # (E, D, 3)
    gate_quat = gates_quat[env_idx, idx]  # (E, D, 4)
    n_envs, n_drones = gate_pos.shape[:2]
    rot = R.from_quat(gate_quat.reshape(-1, 4)).as_matrix().reshape(n_envs, n_drones, 3, 3)
    # rot maps gate-frame vectors to world; its transpose maps the world offset into the gate frame.
    local = jp.einsum("edji,edj->edi", rot, drone_pos - gate_pos)
    return gate_pos, rot, local


def gate_aperture_distance(
    drone_pos: Array, gates_pos: Array, gates_quat: Array, target_gate: Array, half_extent: float
) -> Array:
    """Distance from each drone to the nearest point of its target gate's opening.

    The opening is the square in the gate's y-z plane (the +x column of the gate rotation is the
    traversal normal). We keep the full along-axis (x) gap and clamp the lateral (y, z) offsets to
    the opening, independently per axis -- so the corridor of equally-good crossing points is a
    *cuboid* prism whose square cross-section matches the gate hole. Concretely::

        dist = sqrt(along**2 + max(|y| - h, 0)**2 + max(|z| - h, 0)**2)

    * Laterally inside the opening (``|y|, |z| < h``) this is the pure forward distance to the gate
      plane, so every point inside the hole is an equally good crossing target.
    * Off to the side, the lateral terms pull the drone toward the nearest opening edge and make
      "fly forward beside the frame" earn strictly less than crossing -- and flying *past* beside
      the frame increases the distance again (a penalty), so the term cannot be farmed by skirting
      the gate. It is a potential function, so the per-step progress (its decrease) telescopes to
      net displacement and cannot be looped for free reward.

    Returns distance, shape (n_envs, n_drones).
    """
    _, _, local = _target_gate_frame(drone_pos, gates_pos, gates_quat, target_gate)
    along = local[..., 0]
    dy = jp.maximum(jp.abs(local[..., 1]) - half_extent, 0.0)
    dz = jp.maximum(jp.abs(local[..., 2]) - half_extent, 0.0)
    return jp.sqrt(along**2 + dy**2 + dz**2)


def gate_aperture_target_point(
    drone_pos: Array, gates_pos: Array, gates_quat: Array, target_gate: Array, half_extent: float
) -> Array:
    """World-frame nearest point on the target gate opening to each drone (the reward's pull target).

    Mirrors :func:`gate_aperture_distance` for visualization: projects the drone onto the gate plane
    (along-axis x -> 0) and clamps the in-plane (y, z) coordinates to the square opening.

    Returns points, shape (n_envs, n_drones, 3).
    """
    gate_pos, rot, local = _target_gate_frame(drone_pos, gates_pos, gates_quat, target_gate)
    y = jp.clip(local[..., 1], -half_extent, half_extent)
    z = jp.clip(local[..., 2], -half_extent, half_extent)
    nearest_local = jp.stack([jp.zeros_like(y), y, z], axis=-1)
    return gate_pos + jp.einsum("edij,edj->edi", rot, nearest_local)


def racing_reward_components(
    data: Any,
    prev_data: Any,
    *,
    progress_coef: float,
    gate_bonus: float,
    finish_bonus: float,
    crash_penalty: float,
    timeout_penalty: float,
    gate_half_extent: float,
) -> dict[str, Array]:
    """Per-step racing reward broken into its named, already-weighted+signed terms.

    Each value is a ``(n_envs, n_drones)`` array; their sum is the env-side reward (before the
    wrapper penalties added by ``AngleReward`` / ``ActionPenalty``). The single source of truth for
    both the real reward (``build_racing_reward`` sums these) and the per-component wandb logging
    (``LogRewardComponents`` recomputes them), so the chart can never drift from what's optimized.
    """
    # Progress = decrease in distance to the cuboid opening of the gate that was the target at the
    # start of the step (forward inside the hole, lateral pull toward the opening otherwise).
    dist_prev = gate_aperture_distance(
        prev_data.sim_data.states.pos, data.gates_pos, data.gates_quat, prev_data.target_gate, gate_half_extent
    )
    dist_curr = gate_aperture_distance(
        data.sim_data.states.pos, data.gates_pos, data.gates_quat, prev_data.target_gate, gate_half_extent
    )
    progress = dist_prev - dist_curr

    active = prev_data.target_gate != -1  # episode was not already finished
    passed_gate = (data.target_gate != prev_data.target_gate) & active
    finished = (data.target_gate == -1) & active
    not_finished = data.target_gate != -1
    crashed = data.disabled_drones & ~prev_data.disabled_drones & not_finished
    # Truncation fires at the step steps == max_episode_steps; NEXT_STEP autoreset doesn't reset
    # `steps` until the following step, so it's still readable here. Penalize only if not finished
    # (a drone that already finished and idles to timeout has target_gate == -1 -> exempt).
    timed_out = (data.steps >= data.max_episode_steps)[:, None] & not_finished & active

    # Zero every term on auto-reset transition steps (prev/post straddle an episode boundary).
    keep = (~prev_data.marked_for_reset[:, None]).astype(jp.float32)
    return {
        "progress": progress_coef * progress * keep,
        "gate_bonus": gate_bonus * passed_gate * keep,
        "finish": finish_bonus * finished * keep,
        "crash": -crash_penalty * crashed * keep,
        "timeout": -timeout_penalty * timed_out * keep,
    }


def build_racing_reward(
    progress_coef: float = 1.0,
    gate_bonus: float = 2.0,
    finish_bonus: float = 10.0,
    crash_penalty: float = 5.0,
    timeout_penalty: float = 5.0,
    gate_half_extent: float = GATE_HALF_EXTENT,
) -> Callable[[Any, Any], Array]:
    """Build a dense racing reward to be compiled into the env step.

    The reward is computed inside the env (via ``reward_fn``) so it can use the *true* gate
    positions, which the observation only reveals once a gate is sensed. It combines:

    * progress: reduction in distance to the current target gate's cuboid *opening* (see
      :func:`gate_aperture_distance`), measured against the gate that was the target at the *start*
      of the step. Inside the opening corridor this is pure forward progress toward the plane (any
      crossing point equally good); off to the side it pulls the drone toward the opening and
      penalizes skirting past the frame. Being a potential function it cannot be farmed; the
      gate_bonus confirms the actual crossing,
    * gate_bonus: a one-off bonus each time the target gate advances,
    * finish_bonus: a large one-off bonus when the final gate is passed (target_gate -> -1),
    * crash_penalty: a penalty when the drone is disabled without finishing (out of bounds
      or collision),
    * timeout_penalty: a penalty when the episode truncates (hits ``max_episode_steps``) without
      finishing. Penalizing timeout *symmetrically* with crashing is what stops the policy from
      treating the clock as a safe harbor -- otherwise the dense progress term (fully collected just
      by approaching a gate) plus a free timeout make "approach, bank progress, then idle until
      truncation" dominate actually crossing, so cone_gate_pass_rate peaks then collapses. With both
      failure modes costing the same, the only way to avoid the end-penalty is to finish, and since
      parking banks less progress than flying on (and each gate_bonus already exceeds the penalty),
      forward flight strictly wins.

    Reward is zeroed on auto-reset transition steps, where prev/post state straddle an
    episode boundary.

    Args:
        progress_coef: Weight on the per-step distance-progress term.
        gate_bonus: Bonus added when the target gate advances.
        finish_bonus: Bonus added when the whole track is completed.
        crash_penalty: Penalty subtracted on a crash (collision / out of bounds).
        timeout_penalty: Penalty subtracted on truncation (max_episode_steps) without finishing.
        gate_half_extent: Half-extent (m) of the square gate opening used by the progress term as
            the cuboid corridor of equally-good crossing points.
    """

    def reward(data: Any, prev_data: Any) -> Array:
        terms = racing_reward_components(
            data,
            prev_data,
            progress_coef=progress_coef,
            gate_bonus=gate_bonus,
            finish_bonus=finish_bonus,
            crash_penalty=crash_penalty,
            timeout_penalty=timeout_penalty,
            gate_half_extent=gate_half_extent,
        )
        # Terms are already zeroed on auto-reset transition steps, so the sum is the env reward.
        return sum(terms.values())

    return reward


class LogRewardComponents(VectorWrapper):
    """Innermost monitor that surfaces each env-side reward term in ``info`` for logging.

    Recomputes :func:`racing_reward_components` from the base env data straddling the step -- the
    exact ``(data, prev_data)`` the env's compiled ``reward_fn`` used -- so the logged terms equal
    what is optimized (no drift). It adds one ``rew/<term>`` entry per component; the wrapper
    penalties (``AngleReward`` / ``ActionPenalty``) add their own ``rew/*`` entries higher up, and
    PPO sums all of them per iteration into ``reward/<term>`` charts. Pure monitor: obs, reward, and
    done flags pass through untouched, so it is transparent to ``SegmentSpawn`` above it.
    """

    def __init__(
        self,
        env: VectorEnv,
        *,
        progress_coef: float,
        gate_bonus: float,
        finish_bonus: float,
        crash_penalty: float,
        timeout_penalty: float,
        gate_half_extent: float = GATE_HALF_EXTENT,
    ):
        """Init; jit a closure over the (static) reward coefficients."""
        super().__init__(env)

        @jax.jit
        def components(data: Any, prev_data: Any) -> dict[str, Array]:
            return racing_reward_components(
                data,
                prev_data,
                progress_coef=progress_coef,
                gate_bonus=gate_bonus,
                finish_bonus=finish_bonus,
                crash_penalty=crash_penalty,
                timeout_penalty=timeout_penalty,
                gate_half_extent=gate_half_extent,
            )

        self._components = components

        @jax.jit
        def vel_diag(data: Any) -> tuple[Array, Array]:
            """Velocity projected on the target gate normal (forward) and world-up (vertical)."""
            _, rot, _ = _target_gate_frame(
                data.sim_data.states.pos, data.gates_pos, data.gates_quat, data.target_gate
            )
            normal = rot[..., :, 0]  # (E, D, 3) through-gate (+x) direction in world
            vel = data.sim_data.states.vel  # (E, D, 3)
            along = jp.sum(vel * normal, axis=-1)  # (E, D) forward speed toward the gate
            return along, vel[..., 2]

        self._vel_diag = vel_diag

    def step(self, action: Array) -> tuple[Any, Array, Array, Array, dict]:
        """Step, then stash the per-component env-side reward terms (one drone) into ``info``."""
        prev_data = self.env.unwrapped.data  # pre-step base data == the env reward_fn's prev_data
        obs, reward, terminated, truncated, info = self.env.step(action)
        data = self.env.unwrapped.data
        terms = self._components(data, prev_data)
        info = {**info, **{f"rew/{name}": v[:, 0] for name, v in terms.items()}}
        # Velocity diagnostics: forward speed toward the target gate vs. vertical speed. "Climbs
        # instead of advancing" shows up as vel_up >> vel_along. Logged as diag/* by PPO.
        vel_along, vel_up = self._vel_diag(data)
        info = {**info, "diag/vel_along": vel_along[:, 0], "diag/vel_up": vel_up[:, 0]}
        return obs, reward, terminated, truncated, info


class DrawProgressTarget(VectorWrapper):
    """Draw each env's current progress target (the point the reward pulls the drone toward).

    Debug visualization only: on ``render`` it marks, for every drone, the nearest point on its
    target gate's opening -- the point the dense progress term currently pulls it toward -- so the
    reward's behavior near the gate can be confirmed visually.
    """

    def __init__(self, env: VectorEnv, gate_half_extent: float = GATE_HALF_EXTENT):
        """Init."""
        super().__init__(env)
        self.gate_half_extent = gate_half_extent

    def render(self):
        """Draw the progress target(s) then delegate to the underlying render."""
        base = self.env.unwrapped
        targets = gate_aperture_target_point(
            base.data.sim_data.states.pos,
            base.data.gates_pos,
            base.data.gates_quat,
            base.data.target_gate,
            self.gate_half_extent,
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
        timeout_penalty=args.timeout_penalty,
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
    # Transparent monitor (innermost): surface each env-side reward term in info for per-component
    # wandb charts. Recomputes the same terms reward_fn used; does not alter obs/reward/done.
    env = LogRewardComponents(
        env,
        progress_coef=args.progress_coef,
        gate_bonus=args.gate_bonus,
        finish_bonus=args.finish_bonus,
        crash_penalty=args.crash_penalty,
        timeout_penalty=args.timeout_penalty,
        gate_half_extent=GATE_HALF_EXTENT,
    )
    # Curriculum: respawn drones in per-gate approach cones on (auto)reset. Manages the base data and
    # returns base-format obs (the monitor below it is transparent); inactive until training pushes
    # progress.
    env = SegmentSpawn(env, seed=args.seed)
    # Seed warm rotors on every (auto)reset so the drone starts in hover equilibrium instead of
    # falling with cold rotors. Sits *outside* SegmentSpawn: the cone respawn overrides the pose
    # first (leaving rotor_vel untouched), then this warms rotor_vel for the same just-reset envs
    # (true-start and cone-spawned alike), so no spawn begins with dead rotors.
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
    env = DrawProgressTarget(env, gate_half_extent=GATE_HALF_EXTENT)
    # Debug visualization of the curriculum spawn points (drawn only when rendering).
    env = DrawSpawnPoints(env)
    return env
