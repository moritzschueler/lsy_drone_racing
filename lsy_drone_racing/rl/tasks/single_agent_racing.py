"""Single-agent drone racing task: env factory + dense in-step reward."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.vector import VectorEnv, VectorWrapper
from jax import Array

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.tasks.progress_variants import (
    GATE_HALF_EXTENT,
    PotentialFn,
    _target_gate_frame,
    build_progress_potential,
    default_progress_params,
    gate_opening_distance,
)
from lsy_drone_racing.rl.wrappers.observation import FlattenJaxObservation, RelativeRacingObs
from lsy_drone_racing.rl.wrappers.reward import ActionPenalty, ActionSmoothnessPenalty, ZeroYaw
from lsy_drone_racing.rl.wrappers.segment_spawn import SegmentSpawn
from lsy_drone_racing.rl.wrappers.takeoff import SpinUpRotors
from lsy_drone_racing.utils import load_config

# Re-exported from progress_variants for back-compat (tests / notebook import these from here).
__all__ = ["GATE_HALF_EXTENT", "gate_opening_distance", "_target_gate_frame"]

@dataclass
class RacingArgs(Args):
    """Task-specific ``Args`` defaults for single-agent racing.

    Overrides the base ``Args`` field defaults with values tuned for this task; the CLI still
    layers any explicit ``--flag`` overrides on top via ``RacingArgs.create(**kwargs)``.
    """

    total_timesteps: int = 50_000_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    target_kl: float = 0.03
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.008
    anneal_ent_coef: bool = False  # decay entropy if True
    # Swappable dense progress reward: (variant_name, coef). The variant selects the per-gate
    # potential Phi (see progress_variants.PROGRESS_VARIANTS); the per-step reward is
    # coef * (Phi(curr) - Phi(prev)). Per-variant shape params live in progress_params (inherited
    # from Args), which always carries every variant's knobs so switching never drops a param.
    progress: tuple[str, float] = ("fancy", 5.0)
    # Per-variant progress shape params, keyed by variant name. Always carries every variant's knobs
    # (see default_progress_params) so switching the active variant never requires dropping a param.
    progress_params: dict = field(default_factory=default_progress_params)
    speed_coef: float = 0.1  # softplus speed-hinge weight (0 disables); starting guess, tune
    speed_threshold: float = 4.0  # speed (m/s) above which the hinge penalizes; free below it
    speed_softness: float = 3.0  # softplus knee sharpness (1/(m/s)); larger = sharper knee
    # Single action-smoothness penalty (champion-style) on the bounded action; replaces the old
    # rpy / act / d_act_xy / d_act_th stack. Whisper-level relative to progress; tune up only once
    # gate-passing is solid (a too-large smoothness penalty rewards "fly calm" over "pass gates").
    d_act_coef: float = 0.000 
    d_act_th_coef: float = 0.0005 # Coefficient for thrust change penalty (thrust smoothness)
    d_act_xy_coef: float = 0.001 # Coefficient for xy action change penalty (attitude smoothness)
    act_coef: float = 0.00 # Coefficient for action penalty (energy smoothness)
    gate_bonus: float = 20.0
    finish_bonus: float = 30.0
    crash_penalty: float = 3.0
    timeout_penalty: float = 0.0  # Terminal penalty if sim truncates without drone finished
    time_alive_penalty: float = 0.03 # Continous penalty for each step alive and not finished
    num_steps: int = 128
    max_episode_length: int = 1500


def softplus_speed_penalty(speed: Array, threshold: float, softness: float) -> Array:
    """Unweighted softplus speed hinge (>= 0, ~0 well below ``threshold``, ~linear above it).

    Smooth one-sided penalty on speed above ``threshold`` (m/s):
    ``softplus(softness * (speed - threshold)) / softness``. This approximates the ReLU hinge
    ``max(0, speed - threshold)`` -- a soft knee at ``threshold`` -- so the drone races freely below
    the threshold and pays a smooth, roughly linear cost for every m/s above it. Unlike a true hinge
    it leaks a small penalty just below ``threshold`` (softplus never reaches exactly 0); larger
    ``softness`` sharpens the knee and shrinks that leak (the transition happens over ~1/softness
    m/s). ``jax.nn.softplus`` is float32-stable for large args (returns ~arg, no exp overflow), and
    the penalty grows only linearly, so no exponent cap is needed. The caller scales it by
    ``speed_coef`` and negates it into a penalty.
    """
    return jax.nn.softplus(softness * (speed - threshold)) / softness


def racing_reward_components(
    data: Any,
    prev_data: Any,
    *,
    progress_potential: PotentialFn,
    progress_coef: float,
    gate_bonus: float,
    finish_bonus: float,
    crash_penalty: float,
    timeout_penalty: float,
    time_penalty: float,
    gate_half_extent: float,
    speed_coef: float,
    speed_threshold: float,
    speed_softness: float,
) -> dict[str, Array]:
    """Per-step racing reward broken into its named, already-weighted+signed terms.

    Each value is a ``(n_envs, n_drones)`` array; their sum is the env-side reward (before the
    wrapper penalty added by ``ActionSmoothnessPenalty``). The single source of truth for
    both the real reward (``build_racing_reward`` sums these) and the per-component wandb logging
    (``LogRewardComponents`` recomputes them), so the chart can never drift from what's optimized.
    """
    # Progress = the per-step INCREASE of the selected progress potential Phi (progress_potential,
    # one of progress_variants.PROGRESS_VARIANTS), measured against the gate that was the target at
    # the start of the step (prev_data.target_gate for both terms, so the gate-advance is handled
    # without an artifact). Every variant is a deterministic function of state, so this difference
    # telescopes and cannot be farmed by looping; the gate-advance resets the reference potential to
    # the next gate (a one-step change bounded by the drone's per-step displacement, not by
    # progress_coef -- so crossing is never meaningfully net-penalized). Left unmasked on the
    # crossing step to keep it a pure potential difference (no bias, no flat step).
    phi_prev = progress_potential(
        prev_data.sim_data.states.pos, data.gates_pos, data.gates_quat, prev_data.target_gate,
        gate_half_extent,
    )
    phi_curr = progress_potential(
        data.sim_data.states.pos, data.gates_pos, data.gates_quat, prev_data.target_gate,
        gate_half_extent,
    )
    progress = phi_curr - phi_prev

    active = prev_data.target_gate != -1  # episode was not already finished
    passed_gate = (data.target_gate != prev_data.target_gate) & active
    finished = (data.target_gate == -1) & active
    not_finished = data.target_gate != -1
    crashed = data.disabled_drones & ~prev_data.disabled_drones & not_finished
    # Truncation fires at the step steps == max_episode_steps; NEXT_STEP autoreset doesn't reset
    # `steps` until the following step, so it's still readable here. Penalize only if not finished
    # (a drone that already finished and idles to timeout has target_gate == -1 -> exempt).
    timed_out = (data.steps >= data.max_episode_steps)[:, None] & not_finished & active

    # Speed hinge: a smooth one-sided penalty on speed above speed_threshold (see
    # softplus_speed_penalty). The drone races freely below the threshold and pays a roughly linear
    # cost per m/s above it, keeping top speed realistic. Disabled when speed_coef == 0.
    speed = jnp.linalg.norm(data.sim_data.states.vel, axis=-1)
    speed_term = -speed_coef * softplus_speed_penalty(speed, speed_threshold, speed_softness)

    # Dense per-step time cost: charged every step the drone is actively racing and not yet finished
    # (the finishing step itself has not_finished == False, so finishing is never time-penalized; the
    # already-finished/idle steps have active == False -> exempt). This is the per-step pace pressure
    # the terminal timeout_penalty cannot provide across a 1500-step / gamma=0.99 horizon.
    racing = not_finished & active

    # Zero every term on auto-reset transition steps (prev/post straddle an episode boundary).
    keep = (~prev_data.marked_for_reset[:, None]).astype(jnp.float32)
    return {
        "progress": progress_coef * progress * keep,
        "gate_bonus": gate_bonus * passed_gate * keep,
        "finish": finish_bonus * finished * keep,
        "crash": -crash_penalty * crashed * keep,
        "timeout": -timeout_penalty * timed_out * keep,
        "time": -time_penalty * racing * keep,
        "speed": speed_term * keep,
    }


def build_racing_reward(
    progress_potential: PotentialFn | None = None,
    progress_coef: float = 1.0,
    gate_bonus: float = 2.0,
    finish_bonus: float = 10.0,
    crash_penalty: float = 5.0,
    timeout_penalty: float = 5.0,
    time_penalty: float = 0.0,
    gate_half_extent: float = GATE_HALF_EXTENT,
    speed_coef: float = 0.0,
    speed_threshold: float = 4.0,
    speed_softness: float = 3.0,
) -> Callable[[Any, Any], Array]:
    """Build a dense racing reward to be compiled into the env step.

    The reward is computed inside the env (via ``reward_fn``) so it can use the *true* gate
    positions, which the observation only reveals once a gate is sensed. It combines:

    * progress: the per-step *increase* of a swappable per-gate potential ``Phi``
      (``progress_potential``, one of :data:`progress_variants.PROGRESS_VARIANTS`; defaults to the
      champion-paper ``Phi = -gate_opening_distance``), measured against the gate that was the target
      at the *start* of the step. Every variant is a deterministic function of state, so it cannot be
      farmed by looping. The gate-advance resets the reference potential to the next gate, a one-step
      change bounded by the drone's per-step displacement (not by ``progress_coef``), so crossing a
      gate is never meaningfully net-penalized,
    * gate_bonus: a one-off bonus each time the target gate advances,
    * finish_bonus: a large one-off bonus when the final gate is passed (target_gate -> -1),
    * crash_penalty: a penalty when the drone is disabled without finishing (out of bounds
      or collision),
    * timeout_penalty: a one-off penalty when the episode truncates (hits ``max_episode_steps``)
      without finishing. Being terminal and discounted over the full horizon it exerts little
      *pace* pressure on its own (the ``time`` term below does that); it mainly discourages the
      pathological "idle out the clock" end-state,
    * time: a dense per-step "living"/time cost charged every step the drone is still actively racing
      (not yet finished). This is the racing pressure the terminal ``timeout_penalty`` cannot
      provide: the progress term telescopes (creeping and sprinting to a gate bank the *same*
      cumulative progress), and a single penalty at step ``max_episode_steps`` is discounted
      (gamma over a 1500-step horizon) to a near-zero gradient when the policy chooses its pace.
      A constant per-step cost gives a non-vanishing gradient toward finishing sooner, so forward
      flight beats the safe-but-slow basin. Disabled when ``time_penalty == 0``.
    * speed: a softplus hinge on speed above ``speed_threshold`` (see
      :func:`softplus_speed_penalty`); the drone races freely below the threshold and pays a smooth,
      roughly linear cost per m/s above it, keeping top speed realistic. ``speed_softness`` sets the
      knee sharpness. Disabled when ``speed_coef == 0``.

    Reward is zeroed on auto-reset transition steps, where prev/post state straddle an
    episode boundary.

    Args:
        progress_coef: Weight on the per-step distance-progress term.
        gate_bonus: Bonus added when the target gate advances.
        finish_bonus: Bonus added when the whole track is completed.
        crash_penalty: Penalty subtracted on a crash (collision / out of bounds).
        timeout_penalty: Penalty subtracted on truncation (max_episode_steps) without finishing.
        time_penalty: Dense per-step cost charged every step the drone is still racing (not finished);
            0 = off. Provides the pace pressure the terminal timeout_penalty cannot.
        gate_half_extent: Half-extent (m) of the square gate opening used by the progress term as
            the cuboid corridor of equally-good crossing points.
        speed_coef: Overall weight of the softplus speed-hinge penalty; 0 = off.
        speed_threshold: Speed (m/s) above which the hinge penalizes; the drone is free below it.
        speed_softness: Sharpness (1/(m/s)) of the softplus knee; larger = a sharper hinge that
            bites closer to the threshold and leaks less below it.
        progress_potential: The per-gate potential ``Phi`` whose per-step increase is the progress
            reward; ``None`` defaults to the champion variant. Build one with
            :func:`progress_variants.build_progress_potential`.
    """
    if progress_potential is None:
        progress_potential = build_progress_potential("champion", default_progress_params())

    def reward(data: Any, prev_data: Any) -> Array:
        terms = racing_reward_components(
            data,
            prev_data,
            progress_potential=progress_potential,
            progress_coef=progress_coef,
            gate_bonus=gate_bonus,
            finish_bonus=finish_bonus,
            crash_penalty=crash_penalty,
            timeout_penalty=timeout_penalty,
            time_penalty=time_penalty,
            gate_half_extent=gate_half_extent,
            speed_coef=speed_coef,
            speed_threshold=speed_threshold,
            speed_softness=speed_softness,
        )
        # Terms are already zeroed on auto-reset transition steps, so the sum is the env reward.
        return sum(terms.values())

    return reward


class LogRewardComponents(VectorWrapper):
    """Innermost monitor that surfaces each env-side reward term in ``info`` for logging.

    Recomputes :func:`racing_reward_components` from the base env data straddling the step -- the
    exact ``(data, prev_data)`` the env's compiled ``reward_fn`` used -- so the logged terms equal
    what is optimized (no drift). It adds one ``rew/<term>`` entry per component; the
    ``ActionSmoothnessPenalty`` wrapper adds its ``rew/d_act`` entry higher up, and
    PPO sums all of them per iteration into ``reward/<term>`` charts. Pure monitor: obs, reward, and
    done flags pass through untouched, so it is transparent to ``SegmentSpawn`` above it.
    """

    def __init__(
        self,
        env: VectorEnv,
        *,
        progress_potential: PotentialFn,
        progress_coef: float,
        gate_bonus: float,
        finish_bonus: float,
        crash_penalty: float,
        timeout_penalty: float,
        time_penalty: float = 0.0,
        gate_half_extent: float = GATE_HALF_EXTENT,
        speed_coef: float = 0.0,
        speed_threshold: float = 4.0,
        speed_softness: float = 3.0,
    ):
        """Init; jit a closure over the (static) progress potential + reward coefficients."""
        super().__init__(env)

        @jax.jit
        def components(data: Any, prev_data: Any) -> dict[str, Array]:
            return racing_reward_components(
                data,
                prev_data,
                progress_potential=progress_potential,
                progress_coef=progress_coef,
                gate_bonus=gate_bonus,
                finish_bonus=finish_bonus,
                crash_penalty=crash_penalty,
                timeout_penalty=timeout_penalty,
                time_penalty=time_penalty,
                gate_half_extent=gate_half_extent,
                speed_coef=speed_coef,
                speed_threshold=speed_threshold,
                speed_softness=speed_softness,
            )

        self._components = components

        @jax.jit
        def vel_diag(data: Any) -> tuple[Array, Array, Array]:
            """Velocity along the target gate normal (forward), world-up, and its magnitude."""
            _, rot, _ = _target_gate_frame(
                data.sim_data.states.pos, data.gates_pos, data.gates_quat, data.target_gate
            )
            normal = rot[..., :, 0]  # (E, D, 3) through-gate (+x) direction in world
            vel = data.sim_data.states.vel  # (E, D, 3)
            along = jnp.sum(vel * normal, axis=-1)  # (E, D) forward speed toward the gate
            speed = jnp.linalg.norm(vel, axis=-1)  # (E, D) speed magnitude
            return along, vel[..., 2], speed

        self._vel_diag = vel_diag

    def step(self, action: Array) -> tuple[Any, Array, Array, Array, dict]:
        """Step, then stash the per-component env-side reward terms (one drone) into ``info``."""
        prev_data = self.env.unwrapped.data  # pre-step base data == the env reward_fn's prev_data
        obs, reward, terminated, truncated, info = self.env.step(action)
        data = self.env.unwrapped.data
        terms = self._components(data, prev_data)
        info = {**info, **{f"rew/{name}": v[:, 0] for name, v in terms.items()}}
        # Velocity diagnostics: forward speed toward the target gate vs. vertical speed. "Climbs
        # instead of advancing" shows up as vel_up >> vel_along. Plus speed magnitude, logged both
        # mean-per-step (diagnostics/vel_mean) and as the iteration peak (max/vel -> diagnostics/
        # vel_max). Logged as diagnostics/* by PPO.
        vel_along, vel_up, speed = self._vel_diag(data)
        info = {
            **info,
            "diagnostics/vel_along": vel_along[:, 0],
            "diagnostics/vel_up": vel_up[:, 0],
            "diagnostics/vel_mean": speed[:, 0],
            "max/vel": speed[:, 0],
        }
        return obs, reward, terminated, truncated, info


def make_env(
    args: Args, num_envs: int, jax_device: str = "cpu", config: str = "level0.toml"
) -> VectorEnv:
    """Build the vectorized, fully-wrapped racing environment."""
    config = load_config(Path(__file__).parents[3] / "config" / config)
    # Resolve the swappable progress potential once and share it between the env reward and the
    # logging monitor so the logged components can never drift from what is optimized.
    progress_variant, progress_coef = args.progress
    progress_potential = build_progress_potential(progress_variant, args.progress_params)
    reward_fn = build_racing_reward(
        progress_potential=progress_potential,
        progress_coef=progress_coef,
        gate_bonus=args.gate_bonus,
        finish_bonus=args.finish_bonus,
        crash_penalty=args.crash_penalty,
        timeout_penalty=args.timeout_penalty,
        time_penalty=args.time_alive_penalty,
        speed_coef=args.speed_coef,
        speed_threshold=args.speed_threshold,
        speed_softness=args.speed_softness,
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
        max_episode_steps=args.max_episode_length
    )
    # Transparent monitor (innermost): surface each env-side reward term in info for per-component
    # wandb charts. Recomputes the same terms reward_fn used; does not alter obs/reward/done.
    env = LogRewardComponents(
        env,
        progress_potential=progress_potential,
        progress_coef=progress_coef,
        gate_bonus=args.gate_bonus,
        finish_bonus=args.finish_bonus,
        crash_penalty=args.crash_penalty,
        timeout_penalty=args.timeout_penalty,
        time_penalty=args.time_alive_penalty,
        gate_half_extent=GATE_HALF_EXTENT,
        speed_coef=args.speed_coef,
        speed_threshold=args.speed_threshold,
        speed_softness=args.speed_softness,
    )
    # Curriculum: respawn drones in per-gate approach cones on (auto)reset. Manages the base data
    # and returns base-format obs (the monitor below it is transparent); inactive until training
    # pushes progress.
    env = SegmentSpawn(env, seed=args.seed)
    # Seed warm rotors on every (auto)reset so the drone starts in hover equilibrium instead of
    # falling with cold rotors. Sits *outside* SegmentSpawn: the cone respawn overrides the pose
    # first (leaving rotor_vel untouched), then this warms rotor_vel for the same just-reset envs
    # (true-start and cone-spawned alike), so no spawn begins with dead rotors.
    env = SpinUpRotors(env)
    env = ActionPenalty(env, act_coef=args.act_coef, d_act_th_coef = args.d_act_th_coef, d_act_xy_coef = args.d_act_xy_coef)
    env = NormalizeActions(env)
    env = ZeroYaw(env)
    # Relative geometry + next-2-gates + rotation matrices (must come after
    # ActionSmoothnessPenalty so last_action is present in the dict it transforms).
    env = RelativeRacingObs(env)
    env = FlattenJaxObservation(env)
    return env
