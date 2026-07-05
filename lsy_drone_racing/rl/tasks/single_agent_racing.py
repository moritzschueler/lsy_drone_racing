"""Single-agent drone racing task: env factory + dense in-step reward."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import flax.struct as struct
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.tasks.progress_variants import (
    GATE_HALF_EXTENT,
    PotentialFn,
    _target_gate_frame,
    build_progress_potential,
    default_progress_params,
)
from lsy_drone_racing.rl.wrappers.observation import FlattenJaxObservation, RelativeRacingObs
from lsy_drone_racing.rl.wrappers.racing_env import RacingEnv
from lsy_drone_racing.rl.wrappers.reward import ActionPenalty, NormalizeActions, ZeroYaw
from lsy_drone_racing.rl.wrappers.takeoff import SpinUpRotors
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper


@dataclass
class RacingArgs(Args):
    """Task-specific ``Args`` defaults for single-agent racing.

    Overrides the base ``Args`` field defaults with values tuned for this task; the CLI still
    layers any explicit ``--flag`` overrides on top via ``RacingArgs.create(**kwargs)``.
    """

    total_timesteps: int = 70_000_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    target_kl: float = 0.03
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.008
    anneal_ent_coef: bool = False  # decay entropy if True
    progress: tuple[str, float] = ("fancy", 5.0) # Tuple of variant and coefficent
    progress_params: dict = field(default_factory=default_progress_params)
    speed_coef: float = 0.05  # quadratic speed-hinge weight (0 disables); starting guess, tune
    speed_threshold: float = 4.0  # speed (m/s) above which the hinge penalizes; free below it
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


def quadratic_speed_penalty(speed: Array, threshold: float) -> Array:
    """Unweighted quadratic speed hinge (0 below ``threshold``, growing quadratically above it).

    One-sided penalty on speed above ``threshold`` (m/s): ``max(0, speed - threshold) ** 2``. It is
    *exactly* zero below the threshold -- a true free zone, no penalty and no gradient, so the drone
    races freely up to it -- and rises quadratically above it, so the marginal cost grows with the
    overshoot: a little over the threshold is cheap, blowing well past it is expensive. It is C^1 at
    the knee (both value and gradient are zero at ``threshold``), so it introduces no gradient
    discontinuity. The caller scales it by ``speed_coef`` and negates it into a penalty.
    """
    return jnp.square(jnp.maximum(speed - threshold, 0.0))


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

    # Speed hinge: a one-sided penalty on speed above speed_threshold (see quadratic_speed_penalty).
    # Exactly zero below the threshold (the drone races freely up to it) and growing quadratically
    # above it, keeping top speed realistic. Disabled when speed_coef == 0.
    speed = jnp.linalg.norm(data.sim_data.states.vel, axis=-1)
    speed_term = -speed_coef * quadratic_speed_penalty(speed, speed_threshold)

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
    * speed: a quadratic hinge on speed above ``speed_threshold`` (see
      :func:`quadratic_speed_penalty`); exactly zero below the threshold (the drone races freely up
      to it) and growing quadratically above it, keeping top speed realistic. Disabled when
      ``speed_coef == 0``.

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
        speed_coef: Overall weight of the quadratic speed-hinge penalty; 0 = off.
        speed_threshold: Speed (m/s) above which the hinge penalizes; the drone is free below it.
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
        )
        # Terms are already zeroed on auto-reset transition steps, so the sum is the env reward.
        return sum(terms.values())

    return reward


@struct.dataclass
class LogRewardComponents(Wrapper):
    """Innermost functional monitor that surfaces each env-side reward term in ``info`` for logging.

    Recomputes :func:`racing_reward_components` from the base ``EnvData`` straddling the step -- the
    exact ``(data, prev_data)`` the env's compiled ``reward_fn`` used (``prev_data`` is the wrapped
    env's pre-step ``data``, ``data`` its post-step ``data``) -- so the logged terms equal what is
    optimized (no drift). It adds one ``rew/<term>`` entry per component (``ActionPenalty`` adds its
    own ``rew/act``/``rew/d_act_*`` higher up), plus velocity diagnostics; PPO sums/peaks them per
    iteration into ``reward/<term>`` and ``diagnostics/<name>`` charts. Pure monitor: obs, reward
    and done flags pass through untouched. Wrap *directly* over ``RacingEnv`` so ``env.base.data``
    is the raw ``EnvData``.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode,
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
    ) -> LogRewardComponents:
        """Create a LogRewardComponents monitor around the (RacingEnv) base environment."""

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
            )

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

        def reset(
            env: LogRewardComponents, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[LogRewardComponents, tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            return env.replace(base=base_env), (obs, info)

        def step(
            env: LogRewardComponents, action: Array
        ) -> tuple[LogRewardComponents, tuple[Any, ...]]:
            prev_data = env.base.data  # pre-step EnvData == the env reward_fn's prev_data
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            data = base_env.data  # post-step EnvData
            terms = components(data, prev_data)
            info = {**info, **{f"rew/{name}": v[:, 0] for name, v in terms.items()}}
            # Velocity diagnostics: forward speed toward the target gate vs. vertical speed. "Climbs
            # instead of advancing" shows up as vel_up >> vel_along. Speed magnitude is logged both
            # mean-per-step (diagnostics/vel_mean) and as the iteration peak (max/vel ->
            # diagnostics/vel_max). Logged as diagnostics/* by PPO.
            vel_along, vel_up, speed = vel_diag(data)
            info = {
                **info,
                "diagnostics/vel_along": vel_along[:, 0],
                "diagnostics/vel_up": vel_up[:, 0],
                "diagnostics/vel_mean": speed[:, 0],
                "max/vel": speed[:, 0],
            }
            return env.replace(base=base_env), (obs, reward, terminated, truncated, info)

        return cls(base=base, step=step, reset=reset)


def make_env(args: Args, config: dict = None) -> Any:
    """Build the vectorized, fully-wrapped racing environment."""
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
    )
    env = VecDroneRaceEnv(
        num_envs=args.num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
        device=args.jax_device,
        reward_fn=reward_fn,
        max_episode_steps=args.max_episode_length
    )
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
    )
    # Seed warm rotors on every (auto)reset so the drone starts in hover equilibrium instead of
    # falling with cold rotors.
    env = SpinUpRotors(env)
    env = ActionPenalty(env, act_coef=args.act_coef, d_act_th_coef = args.d_act_th_coef, d_act_xy_coef = args.d_act_xy_coef)
    env = NormalizeActions(env)
    env = ZeroYaw(env)
    # Relative geometry + next-2-gates + rotation matrices
    env = RelativeRacingObs(env)
    env = FlattenJaxObservation(env)
    return env

def make_functional_env(args: Args, config: dict = None) -> Wrapper:
    """Build the fully-wrapped functional (scannable) racing environment.

    Mirrors :func:`make_env` but composes the functional ``.create()`` wrapper chain over a
    :class:`RacingEnv` adapter instead of the gym wrapper stack, so the whole env step is a pure
    pytree-threading function that can be rolled through ``lax.scan``.
    """
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
    )

    env = VecDroneRaceEnv(
    num_envs=args.num_envs,
    freq=config.env.freq,
    sim_config=config.sim,
    sensor_range=config.env.sensor_range,
    control_mode=config.env.control_mode,
    track=config.env.track,
    disturbances=config.env.get("disturbances"),
    randomizations=config.env.get("randomizations"),
    seed=config.env.seed,
    device=args.jax_device,
    reward_fn=reward_fn,
    max_episode_steps=args.max_episode_length
    )

    env = RacingEnv.create(env)
    # Innermost monitor: surface per-component env-side reward terms + velocity diagnostics in info
    # (recomputed from the raw EnvData straddling the step). Wraps RacingEnv directly so
    # env.base.data is the raw EnvData. Pure pass-through for obs/reward/done.
    env = LogRewardComponents.create(
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
    )
    env = SpinUpRotors.create(env)
    env = ActionPenalty.create(env, act_coef=args.act_coef, d_act_th_coef = args.d_act_th_coef, d_act_xy_coef = args.d_act_xy_coef)
    env = NormalizeActions.create(env)
    env = ZeroYaw.create(env)
    env = RelativeRacingObs.create(env)
    env = FlattenJaxObservation.create(env)
    return env
