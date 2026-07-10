"""Multi-agent (self-play) drone racing task: args + env factory.

Extends the single-agent racing task with a *self-play* opponent: the ego drone (drone 0,
trainable) races against one or more frozen opponent drones (drones ``1..n_opponents``) whose
parameters are past snapshots of the ego policy. Because a self-play opponent is just a frozen
copy of the ego's own network, the opponent pool lives entirely on-device as a fixed-size ring
buffer threaded through the rollout scan carry (no host round-trips): every
``opponent_snapshot_interval`` steps the current ego params are written into a rotating slot, and
each episode samples a slot as that env's opponent. See the ``[[multi-agent-integration-plan]]``.

The ego observes the nearest opponent's ego-relative, body-frame position and velocity
(``include_opponent_obs``, on by default; see ``RelativeRacingObs``/``_opponent_body_frame``), and
additionally receives the ``CompetitionReward`` shaping (rank / segment-lead / proximity / downwash
/ victory), ported from the host-side ``OpponentWrapper``.

Opponents are a per-episode mix of two behaviors, both compiled into the same ``lax.scan`` rollout
(see ``ippo.py``):

* self-play -- a frozen snapshot of the ego's own policy, drawn from the on-device pool below.
* scripted PID -- a cascaded position/velocity PID (ported from
  ``lsy_drone_racing.control.attitude_controller.AttitudeController``, see
  ``wrappers.trajectory_opponent``) that flies the same fixed waypoint spline once, at a
  per-episode randomized speed, then holds the final waypoint.

Each env resamples which behavior its opponent(s) use at every episode boundary, with the PID
fraction annealing from ``opponent_pid_prob_start`` to ``opponent_pid_prob_end`` over
``opponent_pid_decay_steps``, so self-play comes to dominate as the pool matures. PID opponents
require ``control_mode == "attitude"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lsy_drone_racing.envs.multi_drone_race import VecMultiDroneRaceEnv
from lsy_drone_racing.envs.race_core import build_action_space, build_observation_space
from lsy_drone_racing.rl.tasks.progress_variants import GATE_HALF_EXTENT, build_progress_potential
from lsy_drone_racing.rl.tasks.single_agent_racing import (
    LogRewardComponents,
    RacingArgs,
    build_racing_reward,
)
from lsy_drone_racing.rl.wrappers.competition_reward import CompetitionReward
from lsy_drone_racing.rl.wrappers.observation import FlattenJaxObservation, RelativeRacingObs
from lsy_drone_racing.rl.wrappers.racing_env import MultiRacingEnv
from lsy_drone_racing.rl.wrappers.reward import ActionPenalty, NormalizeActions, ZeroYaw
from lsy_drone_racing.rl.wrappers.takeoff import SpinUpRotors

if TYPE_CHECKING:
    from lsy_drone_racing.rl.config import Args
    from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper


@dataclass
class MultiAgentRacingArgs(RacingArgs):
    """``Args`` defaults for self-play multi-agent racing.

    Inherits every tuned single-agent racing default (reward coefs, PPO knobs, episode length)
    from :class:`~lsy_drone_racing.rl.tasks.single_agent_racing.RacingArgs` and layers on the
    self-play opponent knobs. The CLI still layers explicit ``--flag`` overrides on top via
    ``MultiAgentRacingArgs.create(**kwargs)``.
    """

    # -- Multi-drone layout --
    # Number of frozen opponent drones sharing the track with the ego. The underlying functional
    # env is built with ``n_drones = 1 + n_opponents``
    n_opponents: int = 1

    # Observe the opponent: multi-agent racing needs the ego to see the other drone, so default the
    # nearest-opponent obs slots on (single-agent leaves this off). See Args.include_opponent_obs.
    include_opponent_obs: bool = True

    # -- On-device self-play opponent pool (ring buffer) --
    # Number of past-ego snapshots kept on device
    opponent_pool_size: int = 8
    # How often (in global steps) to snapshot the current ego params into the next ring-buffer slot
    opponent_snapshot_interval: int = 5_000_000
    # Global step at which opponents are activated
    opponent_start_step: int = 0
    # Recency-weighted sampling over the self-play pool: 0 = uniform over filled slots,
    # 1 = almost always the most recent.
    opponent_recency_bias: float = 0.7

    # -- Competition reward (ego vs. opponent shaping; see CompetitionReward) --
    # Master switch: set False to fall back to plain progress-based racing reward only.
    use_competition_reward: bool = True
    competition_rank_coef: float = 0.0001
    competition_segment_lead_coef: float = 0.0001
    competition_proximity_coef: float = 1.0
    competition_proximity_threshold: float = 0.15
    competition_victory_coef: float = 100.0
    competition_downwash_coef: float = 0.005
    competition_downwash_base_radius: float = 0.2
    competition_downwash_expansion: float = 0.5
    competition_downwash_vertical_softening: float = 0.5

    # -- Warm start --
    # Optional path to a trained single-agent checkpoint to initialize the ego AND seed the whole
    # opponent pool from, so self-play begins with a competent racer on both drones instead of from
    # scratch. Accepts an absolute/relative path, or a bare filename resolved against the
    # single_agent_racing checkpoint dir.
    init_checkpoint: str | None = None

    # -- Scripted PID trajectory-follower opponents (mixed in with the self-play pool) --
    # Fraction of envs whose opponent(s) are PID/spline-controlled rather than drawn from the
    # self-play pool, resampled every episode. Requires control_mode == "attitude"; set both
    # opponent_pid_prob_start and opponent_pid_prob_end to 0 to disable entirely.
    opponent_pid_prob_start: float = 0.8
    # Target fraction once the anneal (below) completes -- lower than the start so self-play comes
    # to dominate opponent behavior as the pool matures.
    opponent_pid_prob_end: float = 0.0
    # Global steps over which opponent_pid_prob linearly anneals from *_start to *_end (held
    # constant at *_end afterwards).
    opponent_pid_decay_steps: int = 80_000_000
    # Per-episode PID speed multiplier range, sampled uniformly; 1.0 == the nominal
    # opponent_pid_t_total-second single pass.
    opponent_pid_speed_min: float = 0.6
    opponent_pid_speed_max: float = 1.6
    # Nominal (speed multiplier == 1.0) time to fly the waypoint spline once, in seconds.
    opponent_pid_t_total: float = 18.0
    # Position-PID gains (kp, ki, kd) and integral-error clamp, identical defaults to
    # lsy_drone_racing.control.attitude_controller.AttitudeController.
    opponent_pid_kp: tuple[float, float, float] = (0.4, 0.4, 1.25)
    opponent_pid_ki: tuple[float, float, float] = (0.05, 0.05, 0.05)
    opponent_pid_kd: tuple[float, float, float] = (0.2, 0.2, 0.4)
    opponent_pid_ki_range: tuple[float, float, float] = (2.0, 2.0, 0.4)


def make_multi_agent_env(args: Args, config: Any = None) -> Wrapper:
    """Build the fully-wrapped functional (scannable) multi-drone racing environment.

    Mirrors :func:`single_agent_racing.make_functional_env` but over a multi-drone
    :class:`~lsy_drone_racing.rl.wrappers.racing_env.MultiRacingEnv` base: the ``n_drones`` axis is
    kept through the whole wrapper chain (drone 0 = trainable ego, drones ``1..`` = opponents), and
    the obs/action wrappers are told ``n_drones`` so their per-drone transforms map over that axis.

    ``CompetitionReward`` is inserted right after ``LogRewardComponents`` and before
    ``SpinUpRotors``/``RelativeRacingObs``: it needs the raw per-drone ``pos``/``target_gate`` and
    the env-shared ``gates_pos`` that ``RelativeRacingObs`` later discards/transforms away, so it
    must run upstream of it. It adds its shaped reward to the ego drone's (index 0) reward only.

    ``n_drones`` is determined by the loaded track config (``len(config.env.track.drones)``), so a
    multi-drone config must be selected on the CLI, e.g.
    ``--task multi_agent_racing --config multi_level0.toml`` (a single-drone config yields
    ``n_drones == 1`` and the assertion below fires). The opponent policies themselves are *not*
    wired here -- the rollout supplies the opponent-drone actions from the self-play pool; this
    factory only builds the environment that consumes a full ``(n_envs, n_drones, act_dim)`` action.
    """
    n_drones = len(config.env.track.drones)
    assert n_drones >= 2, (
        f"multi_agent_racing needs a multi-drone track config (>=2 drones), but "
        f"'{getattr(config.env, 'track', None) and 'track'}' has {n_drones}. Pass e.g. "
        f"--config multi_level0.toml."
    )
    assert n_drones == 1 + args.n_opponents, (
        f"config track has {n_drones} drones (=> {n_drones - 1} opponents) but args.n_opponents="
        f"{args.n_opponents}. Set --n_opponents {n_drones - 1} or pick a matching config."
    )
    n_gates, n_obstacles = len(config.env.track.gates), len(config.env.track.obstacles)

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

    base = VecMultiDroneRaceEnv(
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
        max_episode_steps=args.max_episode_length,
    )

    # Per-drone spaces (identical to single-agent): the MultiRacingEnv arrays carry the extra
    # n_drones axis, but the spaces describe a single drone so the wrappers reuse their space logic.
    env = MultiRacingEnv.create(
        base,
        single_observation_space=build_observation_space(n_gates, n_obstacles),
        single_action_space=build_action_space(config.env.control_mode, config.sim.drone_model),
    )
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
    if args.use_competition_reward:
        # Must run before RelativeRacingObs -- see CompetitionReward's module docstring.
        env = CompetitionReward.create(
            env,
            n_drones=n_drones,
            rank_coef=args.competition_rank_coef,
            segment_lead_coef=args.competition_segment_lead_coef,
            proximity_coef=args.competition_proximity_coef,
            proximity_threshold=args.competition_proximity_threshold,
            victory_coef=args.competition_victory_coef,
            downwash_coef=args.competition_downwash_coef,
            downwash_base_radius=args.competition_downwash_base_radius,
            downwash_expansion=args.competition_downwash_expansion,
            downwash_vertical_softening=args.competition_downwash_vertical_softening,
        )
    env = SpinUpRotors.create(env, n_drones=n_drones)
    env = ActionPenalty.create(
        env,
        act_coef=args.act_coef,
        d_act_th_coef=args.d_act_th_coef,
        d_act_xy_coef=args.d_act_xy_coef,
        n_drones=n_drones,
    )
    env = NormalizeActions.create(env)
    env = ZeroYaw.create(env)
    env = RelativeRacingObs.create(
        env, n_drones=n_drones, include_opponent_obs=args.include_opponent_obs
    )
    env = FlattenJaxObservation.create(env, n_drones=n_drones)
    return env
