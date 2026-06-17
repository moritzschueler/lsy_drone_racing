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
    "progress_reach": 2.0,  # far-field pull length scale (m); >= largest gate-to-gate gap (~2.8 m randomized)
    "progress_sharpness": 0.3,  # directional funnel length scale (m)
    "speed_coef": 0.01,  # exponential speed-barrier weight; tune vs progress
    "max_speed": 3.0,  # speed ceiling (m/s); barrier diverges toward this, drone can't exceed it
    "speed_penalty_slope": 0.15,  # how early/steep the exponential wall rises
    "rpy_coef": 0.001,
    "d_act_xy_coef": 0.001,
    "d_act_th_coef": 0.001,
    "act_coef": 0.001,
    "gate_bonus": 15.0,
    "finish_bonus": 30.0,
    "crash_penalty": 2.0,
    "timeout_penalty": 5.0, # Penalize timeout instead of finishing
    "num_steps": 128,
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


def gate_progress_potential(
    drone_pos: Array,
    gates_pos: Array,
    gates_quat: Array,
    target_gate: Array,
    half_extent: float,
    reach: float,
    sharpness: float,
) -> Array:
    """Directional progress potential toward each drone's target gate (higher == better).

    The per-step progress reward is the *increase* of this potential. Unlike a plain distance, it is
    asymmetric across the gate plane so it builds a through-gate funnel: the env requires gates be
    crossed from -x to +x (see ``gate_passed``), so the entry side (gate-local x < 0) is high and
    the exit side (x > 0) is low, and the potential climbs as the drone lines up and approaches the
    opening from the correct side.

    Geometry (all in the gate frame; the +x column of the gate rotation is the traversal normal):

    * ``distance``: distance to the nearest point of the square opening -- ``along`` (gate-local x)
      kept in full, lateral (y, z) offsets clamped to the opening half-extent per axis, i.e.
      ``sqrt(along**2 + max(|y|-h,0)**2 + max(|z|-h,0)**2)`` (a cuboid prism of equally-good
      crossing points, matching ``gate_passed``'s box test).
    * ``angle_progress = 2*angle/pi - 1`` in [-1, 1], where ``angle`` is between the gate normal and
      the (nearest-opening-point -> drone) vector. In the gate frame the normal is +x, so
      ``cos(angle) = along/distance``: entry side (along < 0) -> +1, dead in the opening -> 0, exit
      side (along > 0) -> -1. This is the directional term.
    * Two **decoupled** length scales blend them::

          reach_term = exp(-distance / reach)      # long-range pull, sized to gate spacing
          w          = exp(-distance / sharpness)  # directional weight, concentrated near the gate
          Phi        = w * angle_progress + (1 - w) * reach_term

      Far out (``w -> 0``) ``Phi -> reach_term``: a smooth, monotone "get closer" field that reaches
      across the largest gate-to-gate gap. Near the gate (``w -> 1``) ``Phi -> angle_progress``: the
      directional funnel dominates, so skirting past the frame or sitting on the exit side reads as
      low potential and the drone is pulled around to the entry side.

    Being a deterministic function of state it is a potential: the per-step progress telescopes and
    cannot be farmed by looping. Note ``Phi`` in (-1, 1], so the worst one-step drop at a crossing
    (entry +1 -> exit -1) is ``2 * progress_coef``; ``gate_bonus`` is held above that (asserted in
    :func:`build_racing_reward`) so passing a gate is never net-penalized on that step.

    Returns potential, shape (n_envs, n_drones).
    """
    _, _, local = _target_gate_frame(drone_pos, gates_pos, gates_quat, target_gate)
    along = local[..., 0]
    oy = jp.maximum(jp.abs(local[..., 1]) - half_extent, 0.0)
    oz = jp.maximum(jp.abs(local[..., 2]) - half_extent, 0.0)
    distance = jp.sqrt(along**2 + oy**2 + oz**2)
    cos_angle = along / (distance + 1e-6)
    angle = jp.arccos(jp.clip(cos_angle, -1.0, 1.0))
    angle_progress = 2.0 * angle / jp.pi - 1.0
    reach_term = jp.exp(-distance / reach)
    w = jp.exp(-distance / sharpness)
    return w * angle_progress + (1.0 - w) * reach_term


def gate_aperture_target_point(
    drone_pos: Array, gates_pos: Array, gates_quat: Array, target_gate: Array, half_extent: float
) -> Array:
    """World-frame nearest point on the target gate opening to each drone (the reward's pull target).

    Mirrors :func:`gate_progress_potential` for visualization: projects the drone onto the gate
    plane (along-axis x -> 0) and clamps the in-plane (y, z) coordinates to the square opening.

    Returns points, shape (n_envs, n_drones, 3).
    """
    gate_pos, rot, local = _target_gate_frame(drone_pos, gates_pos, gates_quat, target_gate)
    y = jp.clip(local[..., 1], -half_extent, half_extent)
    z = jp.clip(local[..., 2], -half_extent, half_extent)
    nearest_local = jp.stack([jp.zeros_like(y), y, z], axis=-1)
    return gate_pos + jp.einsum("edij,edj->edi", rot, nearest_local)


# Cap the barrier's exponent so the (unweighted) penalty stays finite and float32-safe. Without it
# exp() overflows to inf right at the wall and poisons the advantage. exp(_SPEED_ARG_CAP) - 1 is the
# max unweighted penalty (~402); with speed_coef ~ 0.05 that is a firm but bounded wall (~20/step).
_SPEED_ARG_CAP = 6.0


def soft_speed_penalty(speed: Array, max_speed: float, slope: float) -> Array:
    """Unweighted exponential speed barrier (>= 0, 0 at rest, diverging toward ``max_speed``).

    With normalized speed ``u = speed / max_speed`` the penalty is ``expm1(slope * u / (1 - u))``:
    zero at rest and growing without bound as ``u -> 1``, so ``max_speed`` acts as a soft wall the
    drone effectively cannot exceed (rather than the old power-law ramp, which barely bit below the
    limit). ``slope`` sets how early/steep the wall rises.

    The exponent is clamped to ``_SPEED_ARG_CAP`` so the penalty saturates at a large-but-finite
    value for ``u`` near/above 1 (``speed >= max_speed`` is clipped to the wall) -- this keeps the
    reward float32-safe and bounded for stable training. The caller scales it by ``speed_coef`` and
    negates it into a penalty.
    """
    u = jp.clip(speed / max_speed, 0.0, 1.0)
    arg = slope * u / jp.maximum(1.0 - u, 1e-6)  # u/(1-u): 0 at rest, -> large at the wall
    return jp.expm1(jp.minimum(arg, _SPEED_ARG_CAP))


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
    progress_reach: float,
    progress_sharpness: float,
    speed_coef: float,
    max_speed: float,
    speed_penalty_slope: float,
) -> dict[str, Array]:
    """Per-step racing reward broken into its named, already-weighted+signed terms.

    Each value is a ``(n_envs, n_drones)`` array; their sum is the env-side reward (before the
    wrapper penalties added by ``AngleReward`` / ``ActionPenalty``). The single source of truth for
    both the real reward (``build_racing_reward`` sums these) and the per-component wandb logging
    (``LogRewardComponents`` recomputes them), so the chart can never drift from what's optimized.
    """
    # Progress = increase of the directional gate potential (see gate_progress_potential), measured
    # against the gate that was the target at the start of the step. It is NOT masked on the crossing
    # step: there the potential drops (entry high -> exit low) by up to 2*progress_coef, but the
    # gate_bonus (kept >= 2*progress_coef in build_racing_reward) shadows it so the step stays net +.
    # Leaving the term unmasked keeps it a pure potential difference (no bias, no flat step).
    pot_prev = gate_progress_potential(
        prev_data.sim_data.states.pos, data.gates_pos, data.gates_quat, prev_data.target_gate,
        gate_half_extent, progress_reach, progress_sharpness,
    )
    pot_curr = gate_progress_potential(
        data.sim_data.states.pos, data.gates_pos, data.gates_quat, prev_data.target_gate,
        gate_half_extent, progress_reach, progress_sharpness,
    )
    progress = pot_curr - pot_prev

    active = prev_data.target_gate != -1  # episode was not already finished
    passed_gate = (data.target_gate != prev_data.target_gate) & active
    finished = (data.target_gate == -1) & active
    not_finished = data.target_gate != -1
    crashed = data.disabled_drones & ~prev_data.disabled_drones & not_finished
    # Truncation fires at the step steps == max_episode_steps; NEXT_STEP autoreset doesn't reset
    # `steps` until the following step, so it's still readable here. Penalize only if not finished
    # (a drone that already finished and idles to timeout has target_gate == -1 -> exempt).
    timed_out = (data.steps >= data.max_episode_steps)[:, None] & not_finished & active

    # Speed barrier: an exponential penalty that diverges toward max_speed (see soft_speed_penalty),
    # an effective ceiling so the policy races at a controllable pace instead of diving through gates
    # too fast. Disabled when speed_coef == 0.
    speed = jp.linalg.norm(data.sim_data.states.vel, axis=-1)
    speed_term = -speed_coef * soft_speed_penalty(speed, max_speed, speed_penalty_slope)

    # Zero every term on auto-reset transition steps (prev/post straddle an episode boundary).
    keep = (~prev_data.marked_for_reset[:, None]).astype(jp.float32)
    return {
        "progress": progress_coef * progress * keep,
        "gate_bonus": gate_bonus * passed_gate * keep,
        "finish": finish_bonus * finished * keep,
        "crash": -crash_penalty * crashed * keep,
        "timeout": -timeout_penalty * timed_out * keep,
        "speed": speed_term * keep,
    }


def build_racing_reward(
    progress_coef: float = 1.0,
    gate_bonus: float = 2.0,
    finish_bonus: float = 10.0,
    crash_penalty: float = 5.0,
    timeout_penalty: float = 5.0,
    gate_half_extent: float = GATE_HALF_EXTENT,
    progress_reach: float = 2.0,
    progress_sharpness: float = 0.3,
    speed_coef: float = 0.0,
    max_speed: float = 3.0,
    speed_penalty_slope: float = 0.3,
) -> Callable[[Any, Any], Array]:
    """Build a dense racing reward to be compiled into the env step.

    The reward is computed inside the env (via ``reward_fn``) so it can use the *true* gate
    positions, which the observation only reveals once a gate is sensed. It combines:

    * progress: increase of the directional gate potential (see :func:`gate_progress_potential`),
      measured against the gate that was the target at the *start* of the step. The potential is
      high on the gate's entry side and low on its exit side, so approaching the correct side climbs
      it (a through-gate funnel) while skirting past the frame or sitting on the wrong side reads as
      low. Being a potential function it cannot be farmed. The term is left unmasked on the crossing
      step, where the potential drops by up to ``2 * progress_coef``; ``gate_bonus`` is asserted
      ``>= 2 * progress_coef`` so that drop is shadowed and passing is never net-penalized,
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
    * speed: an exponential barrier that diverges toward ``max_speed`` (see
      :func:`soft_speed_penalty`), an effective ceiling the drone cannot exceed; ``speed_penalty_slope``
      sets how early/steep the wall rises. It pushes the policy toward a controllable racing pace.
      Disabled when ``speed_coef == 0``.

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
        progress_reach: Length scale (m) of the progress potential's far field; size to the largest
            gate-to-gate gap so there is no flat dead zone between gates.
        progress_sharpness: Length scale (m) over which the directional (entry-vs-exit) term acts.
        speed_coef: Overall weight of the exponential speed-barrier penalty; 0 = off.
        max_speed: Speed ceiling (m/s) the barrier diverges toward (an effective hard limit).
        speed_penalty_slope: Slope of the barrier; larger = the wall rises earlier/steeper.
    """
    # The directional potential drops by at most its full range (Phi in (-1, 1], so <= 2) across a
    # gate plane, scaled by progress_coef. gate_bonus must cover that one-step drop or crossing a
    # gate is locally punished (the progress term is intentionally left unmasked there; see
    # racing_reward_components). Catch a mis-tuned pair at build time rather than as silent stalling.
    if gate_bonus < 2.0 * progress_coef:
        raise ValueError(
            f"gate_bonus ({gate_bonus}) must be >= 2 * progress_coef ({2.0 * progress_coef}) so the "
            f"one-step potential drop at a gate crossing is shadowed and a pass is not net-penalized."
        )

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
            progress_reach=progress_reach,
            progress_sharpness=progress_sharpness,
            speed_coef=speed_coef,
            max_speed=max_speed,
            speed_penalty_slope=speed_penalty_slope,
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
        progress_reach: float = 2.0,
        progress_sharpness: float = 0.3,
        speed_coef: float = 0.0,
        max_speed: float = 3.0,
        speed_penalty_slope: float = 0.3,
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
                progress_reach=progress_reach,
                progress_sharpness=progress_sharpness,
                speed_coef=speed_coef,
                max_speed=max_speed,
                speed_penalty_slope=speed_penalty_slope,
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
        # instead of advancing" shows up as vel_up >> vel_along. Logged as diagnostics/* by PPO.
        vel_along, vel_up = self._vel_diag(data)
        info = {**info, "diagnostics/vel_along": vel_along[:, 0], "diagnostics/vel_up": vel_up[:, 0]}
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
        progress_reach=args.progress_reach,
        progress_sharpness=args.progress_sharpness,
        speed_coef=args.speed_coef,
        max_speed=args.max_speed,
        speed_penalty_slope=args.speed_penalty_slope,
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
        progress_reach=args.progress_reach,
        progress_sharpness=args.progress_sharpness,
        speed_coef=args.speed_coef,
        max_speed=args.max_speed,
        speed_penalty_slope=args.speed_penalty_slope,
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
