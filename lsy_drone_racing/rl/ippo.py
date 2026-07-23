"""Independent PPO (IPPO) for multi-agent self-play drone racing.

A multi-agent sibling of :mod:`lsy_drone_racing.rl.ppo`, kept as a *separate* module so the
single-agent pipeline in ``ppo.py`` stays completely untouched. One trainable ego drone (drone 0)
is optimized with exactly the same PPO update as the single-agent trainer; the remaining drones
(1..``n_opponents``) are *frozen* opponents whose parameters come from an on-device self-play pool.

Because only drone 0's transitions are stored for the update (ego reward/done/obs/action slices),
the GAE, minibatch update, metric and checkpoint machinery is identical to ``ppo.py`` -- all the
multi-agent logic is confined to the rollout step (run the frozen opponents forward, concatenate
their actions onto the ego's, step the multi-drone env) and to the on-device opponent pool threaded
through the scan carry.

Self-play pool (fully on device, no host round-trips): a fixed-size ring buffer of past ego
parameter snapshots (``opponent_pool_size`` slots). Every ``opponent_snapshot_interval`` steps the
current ego params are written into the next slot; each env samples a slot per episode as its
opponent. The whole run is still one ``lax.scan`` over iterations.

``opponent_start_step`` gates two things, both via a single traced ``opponent_active`` boolean
recomputed once per training iteration (``iter_idx * batch_size >= opponent_start_step``):
  * the opponent's action -- before the threshold, drone 1 gets a frozen no-op/zero action instead
    of a forward pass through the still-largely-random self-play pool (mirrors the host-side
    ``OpponentWrapper``'s behavior for an inactive opponent);
  * the competition-reward shaping (rank / segment-lead / proximity / downwash / victory) --
    ``CompetitionReward`` always computes and adds these into ``reward[:, 0]`` (it has no notion of
    training progress), so before the threshold the rollout strips them back out via the
    ``rew/comp_*`` breakdown it logs in ``info``, leaving plain solo-racing reward. Meaningless
    opponent-relative terms (e.g. "victory" against a stationary drone) would otherwise leak free
    reward into the ego's early training.

The self-play pool itself is *not* gated: it is seeded with the initial ego params and keeps
snapshotting/filling from step 0 regardless, so a real (if still early) opponent is ready to go the
moment ``opponent_active`` flips.

Pool sampling is recency-weighted (``opponent_recency_bias``): a filled slot's age is its distance
from ``write_ptr`` going backwards around the ring, and its sampling weight is
``(1 - opponent_recency_bias) ** age`` -- 0 reduces to uniform over the filled slots, 1 collapses to
always the single most-recently-written slot.
"""

import dataclasses
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from drone_models.core import load_params
from flax import nnx
from jax import Array

import wandb
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.rl.agents.ppo_agent import Agent, _entropy, _log_prob
from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.git_provenance import pin_run_to_branch
from lsy_drone_racing.rl.wrappers.trajectory_opponent import (
    SPAWN_TIME_MARGIN,
    TrajectoryPID,
    build_trajectory_pid,
    teleport_opponents,
)
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper
from lsy_drone_racing.utils.utils import env_param, set_seeds

# Env factory: (args, config) -> fully-wrapped functional multi-drone env (a ``Wrapper``).
MakeEnv = Callable[[Args, Any], Wrapper]


def _resolve_checkpoint(path_str: str, checkpoint_dir: Path | None) -> Path:
    """Resolve a warm-start checkpoint path.

    Accepts an absolute/relative path as given, or a bare filename resolved against the sibling
    ``single_agent_racing`` checkpoint directory (``checkpoint_dir`` is ``.../multi_agent_racing``).
    """
    p = Path(path_str)
    if p.exists():
        return p
    if checkpoint_dir is not None:
        cand = checkpoint_dir.parent / "single_agent_racing" / path_str
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"init_checkpoint '{path_str}' not found (also tried under the single_agent_racing "
        f"checkpoint dir). Pass an existing path."
    )


def train_ippo(
    args: Args,
    make_env: MakeEnv,
    config: Any,
    checkpoint_dir: Path | None,
    run_name: str,
    wandb_enabled: bool = False,
) -> Path | None:
    """Train the ego drone with PPO against a frozen self-play opponent pool.

    Mirrors :func:`lsy_drone_racing.rl.ppo.train_ppo` (one whole-run ``lax.scan``, metrics as scan
    outputs, best checkpoint in the carry) but over a multi-drone env: drone 0 is the trainable ego,
    drones ``1..`` are opponents sampled from an on-device ring buffer of past ego snapshots.
    """
    if wandb_enabled and wandb.run is None:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            group="multi_agent_racing",
        )
        prov = pin_run_to_branch(wandb.run.name, wandb.run.id)
        wandb.config.update(prov, allow_val_change=True)
        if prov.get("wandb_branch"):
            backup = "pushed to remote" if prov.get("wandb_branch_pushed") else "local only"
            print(
                f"[git_provenance] pinned code to branch: {prov['wandb_branch']} "
                f"({prov['git_sha'][:8]}, {backup})"
            )
    train_start_time = time.time()
    set_seeds(args.seed)
    print("Training (IPPO / self-play) on device:", args.jax_device)
    print("--- Hyperparameters ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-----------------------")

    # -- Functional multi-drone env setup --
    envs = make_env(args, config)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    num_envs = args.num_envs
    n_drones = envs.unwrapped.n_drones
    n_opponents = n_drones - 1
    assert n_opponents >= 1, "IPPO needs >=2 drones (>=1 opponent); use ppo.py for single-agent."
    obs_shape = envs.single_observation_space.shape  # per-drone, e.g. (52,)
    act_shape = envs.single_action_space.shape  # per-drone, e.g. (4,)
    action_dim = int(np.prod(act_shape))
    obs_dim = int(np.prod(obs_shape))
    print(f"n_drones={n_drones} (ego=1, opponents={n_opponents})")
    print(f"Shape of (per-drone) observation space: {obs_shape}")
    print(f"Shape of (per-drone) action space: {act_shape}")

    base_space = envs.unwrapped.single_observation_space
    n_gates = (
        base_space["gates_pos"].shape[0]
        if hasattr(base_space, "spaces") and "gates_pos" in base_space.spaces
        else None
    )
    # Length of the configured gate order (track.gate_order) -- the number of gates a drone must
    # pass to finish. Not necessarily equal to n_gates (the physical gate count) under a repeating
    # gate_order, so this (not n_gates) is what "finished"/"gates passed" must be compared against.
    n_gate_passes = (
        base_space["gate_sequence"].shape[0]
        if hasattr(base_space, "spaces") and "gate_sequence" in base_space.spaces
        else None
    )

    # -- Scripted PID trajectory-follower opponents, mixed in with the self-play pool --
    use_pid_opponents = (
        getattr(args, "opponent_pid_prob_start", 0.0) > 0.0
        or getattr(args, "opponent_pid_prob_end", 0.0) > 0.0
    )
    traj_pid: TrajectoryPID | None = None
    if use_pid_opponents:
        assert env_param(config, "control_mode") == "attitude", (
            "opponent_pid_prob_start/opponent_pid_prob_end mixing needs an attitude-control track "
            f"config (physical roll/pitch/yaw/thrust setpoints), got control_mode="
            f"'{env_param(config, 'control_mode')}'."
        )
        # The scripted PID opponent flies a fixed, single-pass, non-looping spline that crosses
        # the physical gates in config.env.track.gates' raw list order (see
        # trajectory_opponent._compute_gate_pass_times / teleport_opponents). Its progress
        # bookkeeping only lines up with n_gates_passed under the plain identity gate_order --
        # a permutation desyncs it from the very first gate, and a repeat leaves it permanently
        # stuck one (or more) gate-order entries short of finishing, since the spline can't fly
        # back around. Fail fast rather than silently degrade self-play against PID opponents.
        assert list(config.env.track.gate_order) == list(range(1, (n_gates or 0) + 1)), (
            "scripted PID opponents require track.gate_order to be the plain forward sequence "
            f"[1..n_gates] ({list(range(1, (n_gates or 0) + 1))}), got "
            f"{list(config.env.track.gate_order)}. Disable PID opponents "
            "(opponent_pid_prob_start=0, opponent_pid_prob_end=0) to train on a permuted or "
            "repeating gate order."
        )
        drone_mass = load_params(config.sim.physics, config.sim.drone_model)["mass"]
        action_space = build_action_space(env_param(config, "control_mode"), config.sim.drone_model)
        traj_pid = build_trajectory_pid(
            start_pos=np.asarray(config.env.track.drones[1]["pos"]),
            drone_mass=drone_mass,
            freq=env_param(config, "freq"),
            control_mode=env_param(config, "control_mode"),
            action_low=np.asarray(action_space.low),
            action_high=np.asarray(action_space.high),
            t_total=args.opponent_pid_t_total,
            kp=args.opponent_pid_kp,
            ki=args.opponent_pid_ki,
            kd=args.opponent_pid_kd,
            ki_range=args.opponent_pid_ki_range,
            gates=config.env.track.gates,
        )
        print(
            f"Scripted PID opponents: prob {args.opponent_pid_prob_start:.2f} -> "
            f"{args.opponent_pid_prob_end:.2f} over {args.opponent_pid_decay_steps} steps, speed "
            f"in [{args.opponent_pid_speed_min}, {args.opponent_pid_speed_max}]x "
            f"({args.opponent_pid_t_total}s nominal single pass)."
        )

    # -- Random mid-track spawn for PID opponents (see teleport_opponents) --
    # Static (python-level) gate: when disabled (frac_max == 0), no teleport code is traced and
    # the traj_t reset below keeps the literal 0.0 -- byte-identical rollout to the pad-start
    # behavior.
    use_random_pid_start = (
        use_pid_opponents and getattr(args, "opponent_pid_start_frac_max", 0.0) > 0.0
    )
    if use_random_pid_start:
        assert 0.0 <= args.opponent_pid_start_frac_min <= args.opponent_pid_start_frac_max <= 1.0, (
            "opponent_pid_start_frac_min/max must satisfy 0 <= min <= max <= 1, got "
            f"[{args.opponent_pid_start_frac_min}, {args.opponent_pid_start_frac_max}]."
        )
        # Never spawn past (or within a margin of) the last gate: the opponent must always have at
        # least one gate left, so it can't start the episode already "finished" (which would break
        # victory/win-rate semantics).
        spawn_t_max = float(traj_pid.gate_times[-1]) - SPAWN_TIME_MARGIN
        print(
            f"PID opponents spawn mid-track: t0 ~ U({args.opponent_pid_start_frac_min:.2f}, "
            f"{args.opponent_pid_start_frac_max:.2f}) * {args.opponent_pid_t_total}s, clamped to "
            f"<= {spawn_t_max:.2f}s (last gate pass at {float(traj_pid.gate_times[-1]):.2f}s)."
        )

        def sample_spawn_t(key: Array, shape: tuple[int, ...]) -> Array:
            """Sample per-slot virtual spawn times along the opponent trajectory."""
            frac = jax.random.uniform(
                key,
                shape,
                minval=args.opponent_pid_start_frac_min,
                maxval=args.opponent_pid_start_frac_max,
            )
            return jnp.minimum(frac * traj_pid.t_total, spawn_t_max)

    # -- Agent + optimizer (functional) -- identical to ppo.py: the ego Agent is per-drone. --
    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(args.seed))
    # Optional warm start: load a trained (single-agent) checkpoint into the ego. The seeded
    # opponent pool below is built from these params, so both drones start competent. Optimizer
    # state still starts fresh (a fine-tune, not a resume).
    init_ckpt = getattr(args, "init_checkpoint", None)
    if init_ckpt:
        ckpt_path = _resolve_checkpoint(init_ckpt, checkpoint_dir)
        with open(ckpt_path, "rb") as f:
            nnx.update(agent, pickle.load(f))
        print(f"Warm-starting ego + opponent pool from checkpoint: {ckpt_path}")
    graphdef, params = nnx.split(agent)
    rng = jax.random.PRNGKey(args.seed)

    if args.anneal_lr:
        schedule = optax.cosine_decay_schedule(
            args.learning_rate, args.num_iterations * args.update_epochs * args.num_minibatches
        )
    else:
        schedule = args.learning_rate
    tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adamw(schedule, eps=1e-5))
    opt_state = tx.init(params)

    # -- Self-play pool config --
    pool_size = int(args.opponent_pool_size)
    batch_size = args.batch_size
    opponent_start_step = int(args.opponent_start_step)
    # Snapshot cadence in iterations (each iteration advances the global step by batch_size).
    snapshot_every = max(1, round(args.opponent_snapshot_interval / batch_size))
    recency_bias = float(args.opponent_recency_bias)
    print(
        f"Self-play pool: {pool_size} slots, snapshot every {snapshot_every} iterations "
        f"(~{snapshot_every * batch_size} steps), recency_bias={recency_bias:.2f} "
        f"({'uniform' if recency_bias == 0.0 else 'recency-weighted'} sampling over filled slots)."
    )
    print(f"Opponent activates at global step {opponent_start_step} (action + competition reward).")

    def forward(params: nnx.State, obs: Array) -> tuple[Array, Array, Array]:
        """Param-pure forward pass: (action_mean, log_std, value)."""
        return nnx.merge(graphdef, params)(obs)

    def policy_step(params: nnx.State, obs: Array, key: Array) -> tuple[Array, Array, Array]:
        mean, log_std, value = forward(params, obs)
        std = jnp.exp(log_std)
        action = mean + std * jax.random.normal(key, mean.shape)
        logprob = _log_prob(action, mean, log_std)
        return action, logprob, value.squeeze(-1)

    def get_value(params: nnx.State, obs: Array) -> Array:
        _, _, value = forward(params, obs)
        return value.squeeze(-1)

    def opponent_actions(pool: nnx.State, opp_obs: Array, opp_idx: Array) -> Array:
        """Frozen (deterministic, mean) opponent actions for drones 1...

        Args:
            pool: ring buffer of ego param snapshots; each leaf is ``(pool_size, *leaf)``.
            opp_obs: opponent observations, ``(num_envs, n_opponents, obs_dim)``.
            opp_idx: per-(env, opponent) pool slot index, ``(num_envs, n_opponents)``.

        Returns:
            ``(num_envs, n_opponents, action_dim)`` opponent actions.
        """
        e, k = opp_idx.shape
        # Gather each (env, opponent)'s params from its sampled slot, then vmap the forward pass
        # over the flattened (env * opponent) axis so every opponent uses its own frozen params.
        gathered = jax.tree.map(lambda leaf: leaf[opp_idx], pool)  # each leaf (e, k, *leaf)
        flat_obs = opp_obs.reshape(e * k, obs_dim)
        flat_params = jax.tree.map(lambda leaf: leaf.reshape((e * k,) + leaf.shape[2:]), gathered)
        fwd = lambda p, o: forward(jax.lax.stop_gradient(p), o)[0]  # noqa: E731
        means = jax.vmap(fwd)(flat_params, flat_obs)
        return means.reshape(e, k, action_dim)

    def compute_gae(
        rewards: Array, values: Array, dones: Array, next_value: Array, next_done: Array
    ) -> tuple[Array, Array]:
        """GAE via lax.scan over reversed time steps (cleanrl semantics)."""

        def scan_fn(lastgaelam: Array, inputs: tuple) -> tuple[Array, Array]:
            reward, value, next_val, next_d = inputs
            nextnonterminal = 1.0 - next_d
            delta = reward + args.gamma * next_val * nextnonterminal - value
            advantage = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            return advantage, advantage

        next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)
        next_dones = jnp.concatenate([dones[1:], next_done[None]], axis=0)
        _, advantages_rev = jax.lax.scan(
            scan_fn,
            jnp.zeros(rewards.shape[1:]),
            (rewards[::-1], values[::-1], next_values[::-1], next_dones[::-1]),
        )
        advantages = advantages_rev[::-1]
        return advantages, advantages + values

    def ppo_loss_fn(
        params: nnx.State,
        obs: Array,
        actions: Array,
        log_probs: Array,
        advantages: Array,
        returns: Array,
        b_values: Array,
        ent_coef: Array,
    ) -> tuple[Array, tuple]:
        mean, log_std, new_values = forward(params, obs)
        new_log_probs = _log_prob(actions, mean, log_std)
        entropy = _entropy(log_std, obs.shape[0])

        log_ratio = new_log_probs - log_probs
        ratio = jnp.exp(log_ratio)
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)

        mb_advantages = advantages
        if args.norm_adv:
            mb_advantages = (mb_advantages - jnp.mean(mb_advantages)) / (
                jnp.std(mb_advantages) + 1e-8
            )

        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        new_values_flat = new_values.reshape(-1)
        if args.clip_vloss:
            v_clipped = b_values + jnp.clip(
                new_values_flat - b_values, -args.clip_coef, args.clip_coef
            )
            v_loss = 0.5 * jnp.mean(
                jnp.maximum((new_values_flat - returns) ** 2, (v_clipped - returns) ** 2)
            )
        else:
            v_loss = 0.5 * jnp.mean((new_values_flat - returns) ** 2)

        entropy_loss = jnp.mean(entropy)
        total_loss = pg_loss - ent_coef * entropy_loss + args.vf_coef * v_loss
        return total_loss, (pg_loss, v_loss, entropy_loss, approx_kl, ratio)

    # -- Discover the static per-step metric keys the wrapper stack surfaces in ``info`` --
    # The multi-drone wrappers already slice ego (drone 0) reward-component / diagnostic entries to
    # (num_envs,), so the metric handling is identical to single-agent. Probe one full-action step.
    _, (_, _, _, _, _probe_info) = envs.step(envs, jnp.zeros((num_envs, n_drones, action_dim)))
    metric_keys = sorted(
        k
        for k in _probe_info
        if isinstance(k, str) and (k.startswith("rew/") or k.startswith("diagnostics/"))
    )
    max_keys = sorted(k for k in _probe_info if isinstance(k, str) and k.startswith("max/"))
    denom = float(args.num_steps * num_envs)
    # Static (python-level) list of the competition-reward breakdown keys CompetitionReward logs,
    # used to strip its shaped reward back out of the ego reward before opponent_start_step. Empty
    # if CompetitionReward isn't in the wrapper stack (use_competition_reward=False) -- gating then
    # becomes a no-op on the reward side but the opponent-action gate below still applies.
    comp_keys = [k for k in metric_keys if k.startswith("rew/comp_")]

    def pid_actions(
        env: Wrapper, traj_t: Array, traj_speed: Array, i_error: Array
    ) -> tuple[Array, Array]:
        """Normalized PID actions for every opponent slot, from the env's raw (unwrapped) state.

        Reads world ``pos``/``vel``/``quat`` directly off the innermost sim state (bypassing the
        flattened/relative observation the policy sees, which drops absolute position) -- the same
        pattern ``SpinUpRotors`` uses. Returns ``(action_norm, new_i_error)``, both ``(E, k, ...)``.
        """
        states = env.unwrapped.data.sim_data.states
        opp_pos, opp_vel, opp_quat = states.pos[:, 1:], states.vel[:, 1:], states.quat[:, 1:]
        action_phys, i_error = traj_pid.action(
            opp_pos, opp_vel, opp_quat, traj_t, traj_speed, i_error
        )
        return traj_pid.normalize(action_phys), i_error

    def rollout(
        params: nnx.State,
        pool: nnx.State,
        filled: Array,
        write_ptr: Array,
        env: Wrapper,
        obs: Array,
        rng: Array,
        ep_ret: Array,
        ep_len: Array,
        opp_idx: Array,
        ep_step: Array,
        finish_step: Array,
        opponent_active: Array,
        traj_state: tuple,
        pid_prob_now: Array | None,
    ) -> tuple[tuple, dict]:
        """One rollout of ``num_steps``; only ego (drone 0) transitions are stored for the update.

        ``obs`` is the full ``(num_envs, n_drones, obs_dim)`` observation. Each step runs the ego
        policy on drone 0 and, for each opponent slot, either the frozen self-play pool or the
        scripted PID trajectory-follower (see ``pid_actions``), concatenates the actions, and steps
        the multi-drone env. Opponent slot indices ``opp_idx`` and, when ``use_pid_opponents``, the
        per-slot PID phase/speed/integral-error/behavior-choice in ``traj_state`` are resampled per
        env whenever the ego episode ends, weighted by recency (see ``opponent_recency_bias`` on
        ``Args``) via ``write_ptr``, the pool's next-write ring position.

        ``opponent_active`` (scalar bool, constant for the duration of this rollout) gates both the
        opponent's action -- a frozen no-op action is substituted while inactive, for *either*
        opponent behavior -- and the competition-reward shaping added into ``reward[:, 0]`` by
        ``CompetitionReward``, which is stripped back out via the ``rew/comp_*`` breakdown in
        ``info`` while inactive.
        """
        # Recency-weighted sampling distribution over pool slots, fixed for this whole rollout
        # (write_ptr/filled only change between rollouts, via snapshot_pool). A filled slot's age is
        # its distance from write_ptr going backwards around the ring (0 = most recently written);
        # weight (1 - opponent_recency_bias) ** age reduces to uniform at bias 0 and collapses onto
        # the single newest slot at bias 1 (0 ** 0 == 1 by convention, so that slot keeps weight 1).
        pool_slot = jnp.arange(pool_size)
        slot_age = (write_ptr - 1 - pool_slot) % pool_size
        slot_weight = jnp.where(
            pool_slot < filled, jnp.power(1.0 - recency_bias, slot_age.astype(jnp.float32)), 0.0
        )
        sample_probs = slot_weight / jnp.sum(slot_weight)

        def step(carry: tuple, _: Any) -> tuple[tuple, dict]:
            env, obs, rng, ep_ret, ep_len, opp_idx, ep_step, finish_step, traj_state = carry
            ego_obs = obs[:, 0]  # (E, obs_dim)
            opp_obs = obs[:, 1:]  # (E, k, obs_dim)

            rng, akey = jax.random.split(rng)
            ego_action, logprob, value = policy_step(params, ego_obs, akey)  # (E, act_dim)
            opp_action_pool = opponent_actions(pool, opp_obs, opp_idx)  # (E, k, act_dim)
            if use_pid_opponents:
                traj_t, traj_speed, traj_i_error, is_pid_opp = traj_state
                pid_action, traj_i_error = pid_actions(env, traj_t, traj_speed, traj_i_error)
                opp_action_selfplay = jnp.where(
                    opponent_active, opp_action_pool, jnp.zeros_like(opp_action_pool)
                )
                use_pid = is_pid_opp & opponent_active
                opp_action = jnp.where(use_pid[..., None], pid_action, opp_action_selfplay)
                traj_t = traj_t + traj_speed / env_param(config, "freq")  # advance virtual traj time
            else:
                # Before opponent_start_step, freeze the opponent to a no-op action instead of
                # racing against a still-largely-random self-play snapshot (mirrors
                # OpponentWrapper's inactive-opponent behavior).
                opp_action = jnp.where(
                    opponent_active, opp_action_pool, jnp.zeros_like(opp_action_pool)
                )
            action = jnp.concatenate([ego_action[:, None], opp_action], axis=1)  # (E, D, act_dim)

            if use_random_pid_start:
                # The env uses NEXT_STEP autoreset keyed on the *pre-step* marked_for_reset flags
                # (race_core.build_step_fn), so these are exactly the envs whose reset fires inside
                # the env.step below.
                will_reset = env.unwrapped.data.marked_for_reset  # (E,)
            env, (next_obs, reward, term, trunc, info) = env.step(env, action)
            if use_random_pid_start:
                # Move freshly reset PID opponents from the pad onto the spline at their (just
                # resampled) random start time. traj_t/traj_speed/is_pid_opp are the new episode's
                # draws: resampled at the previous step's ego-done, which necessarily preceded this
                # reset (a marked env has all drones settled, ego included). traj_t is post-advance
                # here, matching the PID's next-step target exactly. Self-play slots keep the pad
                # spawn. next_obs still shows the opponent on the pad for this one frame -- see
                # teleport_opponents for why that is accepted.
                env = teleport_opponents(
                    env, traj_pid, traj_t, traj_speed, will_reset[:, None] & is_pid_opp
                )
            # CompetitionReward (if present) always adds its shaped components into reward[:, 0];
            # strip them back out before opponent_start_step so the ego trains on plain solo-racing
            # reward until the opponent is actually active (opponent-relative terms are meaningless
            # against a frozen no-op drone, and would otherwise leak free/spurious reward).
            comp_reward = sum((info[k] for k in comp_keys), jnp.zeros_like(reward[:, 0]))
            ego_reward_solo = reward[:, 0] - comp_reward
            ego_reward = jnp.where(opponent_active, reward[:, 0], ego_reward_solo)
            done = jnp.logical_or(term[:, 0], trunc[:, 0])  # ego episode boundary
            donef = done.astype(jnp.float32)

            new_ret = ep_ret + ego_reward
            new_len = ep_len + 1.0
            # n_gates_passed is already the gates-passed count (including at finish -- no -1
            # sentinel to translate), unlike the old target_gate index.
            gates_passed = info["n_gates_passed"][:, 0].astype(jnp.float32)

            # -- Ego win rate (decided at the env-episode boundary: all drones settled) --
            # The multi-drone env only resets once every drone has finished/crashed/timed out, so the
            # winner is decided when ``all(term | trunc)`` fires (true only on that terminal step).
            # Drones are ranked lexicographically: more gates passed wins; if tied on gates and both
            # finished the whole track (n_gates_passed reached n_gate_passes), the earlier finish
            # step wins. The ego (drone 0) wins the episode only if it strictly beats every opponent
            # (ties are not wins). Note: while opponent_active is False the opponent is frozen in
            # place, so ego "wins" trivially -- win_rate/gates_passed metrics are informative only
            # once activated.
            all_gates_passed = info["n_gates_passed"]  # (E, D)
            finished = all_gates_passed >= (n_gate_passes or 0)  # completed the whole track
            new_ep_step = ep_step + 1  # within-episode step counter (per env)
            newly_finished = finished & (finish_step < 0)  # first step a drone reaches the finish
            finish_step = jnp.where(newly_finished, new_ep_step[:, None], finish_step)  # (E, D)
            gates_all = all_gates_passed  # (E, D) -- already the full count once finished
            env_done = jnp.all(term | trunc, axis=1)  # (E,) true only on the terminal step
            ego_gates = gates_all[:, :1]  # (E, 1)
            ego_finished = finished[:, :1]  # (E, 1)
            ego_fstep = finish_step[:, :1]  # (E, 1)
            ego_ahead = (ego_gates > gates_all[:, 1:]) | (
                (ego_gates == gates_all[:, 1:]) & ego_finished & (ego_fstep < finish_step[:, 1:])
            )  # (E, k): ego beats each opponent
            win = jnp.all(ego_ahead, axis=1)  # (E,) ego strictly beats every opponent

            out = {
                "obs": ego_obs,
                "action": ego_action,
                "logprob": logprob,
                "value": value,
                "reward": ego_reward,
                "done": donef,
                "ret_done": jnp.where(done, new_ret, 0.0),
                "len_done": jnp.where(done, new_len, 0.0),
                "gates_done": jnp.where(done, gates_passed, 0.0),
                "completed_done": jnp.where(
                    done, (gates_passed == (n_gate_passes or -1)).astype(jnp.float32), 0.0
                ),
                "win_done": jnp.where(env_done, win.astype(jnp.float32), 0.0),
                "ep_done": jnp.where(env_done, 1.0, 0.0),
                "metrics": {k: jnp.sum(info[k]) for k in metric_keys},
                "metrics_max": {k: jnp.max(info[k]) for k in max_keys},
            }
            ep_ret = jnp.where(done, 0.0, new_ret)
            ep_len = jnp.where(done, 0.0, new_len)
            # Advance the win-tracking state; a finished env-episode restarts the step counter and
            # clears recorded finish steps so the next episode starts fresh.
            ep_step = jnp.where(env_done, 0, new_ep_step)
            finish_step = jnp.where(env_done[:, None], -1, finish_step)
            # Resample opponents for envs whose ego episode just ended (recency-weighted over the
            # filled slots, see sample_probs above).
            rng, skey = jax.random.split(rng)
            resampled = jax.random.choice(
                skey, pool_size, shape=opp_idx.shape, p=sample_probs
            ).astype(opp_idx.dtype)
            opp_idx = jnp.where(done[:, None], resampled, opp_idx)
            if use_pid_opponents:
                # Resample the PID phase/speed/behavior-choice at the same episode boundary; the
                # integral error always resets there too (it's meaningless across episodes).
                reset = done[:, None]  # (E, 1), broadcasts over the opponent-slot axis
                rng, speed_key, pid_key = jax.random.split(rng, 3)
                resampled_speed = jax.random.uniform(
                    speed_key,
                    traj_speed.shape,
                    minval=args.opponent_pid_speed_min,
                    maxval=args.opponent_pid_speed_max,
                )
                resampled_is_pid = jax.random.bernoulli(pid_key, pid_prob_now, is_pid_opp.shape)
                # Fresh episodes start at a random point along the trajectory when mid-track
                # spawning is enabled (the teleport above places the drone there at the actual env
                # reset); otherwise at the pad (t0 = 0). The extra rng split only exists when
                # enabled, keeping the disabled path's random stream identical to before.
                if use_random_pid_start:
                    rng, spawn_key = jax.random.split(rng)
                    resampled_t = sample_spawn_t(spawn_key, traj_t.shape)
                else:
                    resampled_t = 0.0
                traj_t = jnp.where(reset, resampled_t, traj_t)
                traj_speed = jnp.where(reset, resampled_speed, traj_speed)
                traj_i_error = jnp.where(reset[..., None], 0.0, traj_i_error)
                is_pid_opp = jnp.where(reset, resampled_is_pid, is_pid_opp)
                traj_state = (traj_t, traj_speed, traj_i_error, is_pid_opp)
            return (
                env,
                next_obs,
                rng,
                ep_ret,
                ep_len,
                opp_idx,
                ep_step,
                finish_step,
                traj_state,
            ), out

        carry0 = (env, obs, rng, ep_ret, ep_len, opp_idx, ep_step, finish_step, traj_state)
        (env, last_obs, rng, ep_ret, ep_len, opp_idx, ep_step, finish_step, traj_state), outs = (
            jax.lax.scan(step, carry0, None, length=args.num_steps)
        )
        return (env, last_obs, rng, ep_ret, ep_len, opp_idx, ep_step, finish_step, traj_state), outs

    def update_epochs(
        params: nnx.State, opt_state: Any, flat_data: tuple, rng: Array, ent_coef: Array
    ) -> tuple[nnx.State, Any, tuple]:
        """All PPO update epochs as a nested scan; ``target_kl`` freezes updates via masking."""

        def epoch(carry: tuple, _: Any) -> tuple[tuple, tuple]:
            params, opt_state, frozen, rng = carry
            rng, pkey = jax.random.split(rng)
            mb_inds = jax.random.permutation(pkey, args.batch_size).reshape(
                args.num_minibatches, args.minibatch_size
            )

            def minibatch(mb_carry: tuple, inds: Array) -> tuple[tuple, tuple]:
                params, opt_state = mb_carry
                batch = tuple(x[inds] for x in flat_data)
                (_, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
                    params, *batch, ent_coef
                )
                updates, new_opt_state = tx.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                keep = lambda old, new: jnp.where(frozen, old, new)  # noqa: E731
                params = jax.tree.map(keep, params, new_params)
                opt_state = jax.tree.map(keep, opt_state, new_opt_state)
                return (params, opt_state), aux

            (params, opt_state), auxs = jax.lax.scan(minibatch, (params, opt_state), mb_inds)
            mean_kl = jnp.mean(auxs[3])
            if args.target_kl is not None:
                frozen = frozen | (mean_kl > args.target_kl)
            return (params, opt_state, frozen, rng), auxs

        (params, opt_state, _, _), auxs = jax.lax.scan(
            epoch, (params, opt_state, jnp.bool_(False), rng), None, length=args.update_epochs
        )
        return params, opt_state, auxs

    best_key = "charts/gates_passed" if n_gates is not None else "charts/best_reward"
    tag = "g" if n_gates is not None else "r"
    metric = "gates-passed" if n_gates is not None else "return"

    console_live = args.console_log_interval > 0
    wandb_live = wandb_enabled and args.wandb_log_interval > 0
    if console_live:
        print(f"Console progress every {args.console_log_interval} iterations")
    if wandb_live:
        print(f"Publishing metrics to wandb every {args.wandb_log_interval} iterations (live)")

    # Periodic checkpoint saving (in addition to the best checkpoint written at the end): disabled
    # unless both a checkpoint dir is given and checkpoint_save_interval > 0.
    checkpoint_live = checkpoint_dir is not None and args.checkpoint_save_interval > 0
    if checkpoint_live:
        save_every = max(1, round(args.checkpoint_save_interval / args.batch_size))
        print(
            f"Saving periodic checkpoints every {save_every} iterations "
            f"(~{args.checkpoint_save_interval} steps)"
        )

    def _log_iter(
        iter_idx: Array, reward: Array, gates: Array, value_loss: Array, approx_kl: Array
    ) -> None:
        elapsed = time.time() - train_start_time
        print(
            f"Iteration {int(iter_idx) + 1}/{args.num_iterations} | "
            f"reward {float(reward):+.2f} | gates {float(gates):.2f} | "
            f"v_loss {float(value_loss):.3f} | kl {float(approx_kl):.4f} | {elapsed:.1f}s"
        )

    def _wandb_log(step: Array, metrics: dict) -> None:
        wandb.log({k: float(v) for k, v in metrics.items()}, step=int(step))

    def _save_checkpoint(step: Array, params_state: nnx.State, score: Array) -> None:
        """Host-side periodic checkpoint write; called by an ordered jax.debug.callback."""
        nnx.update(agent, params_state)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{run_name}_{timestamp}_{tag}{float(score):.2f}_step{int(step)}.ckpt"
        path = checkpoint_dir / name
        with open(path, "wb") as f:
            pickle.dump(nnx.state(agent, nnx.Param), f)
        print(
            f"Periodic checkpoint (step {int(step)}, {metric} {float(score):.2f}) saved to {path}"
        )

    def snapshot_pool(
        pool: nnx.State, write_ptr: Array, filled: Array, params: nnx.State
    ) -> tuple[nnx.State, Array, Array]:
        """Write the current ego params into ring-buffer slot ``write_ptr``; advance ptr + fill."""
        pool = jax.tree.map(lambda buf, cur: buf.at[write_ptr].set(cur), pool, params)
        write_ptr = (write_ptr + 1) % pool_size
        filled = jnp.minimum(filled + 1, pool_size)
        return pool, write_ptr, filled

    def train_iteration(carry: tuple, iter_idx: Array) -> tuple[tuple, dict]:
        (
            params,
            opt_state,
            env,
            obs,
            prev_done,
            rng,
            ep_ret,
            ep_len,
            pool,
            write_ptr,
            filled,
            opp_idx,
            best_params,
            best_score,
            ep_step,
            finish_step,
            traj_state,
        ) = carry

        # -- Opponent activation gate: static threshold, traced comparison (iter_idx is the scan
        # loop var) so it stays a single compiled program across the whole run. --
        global_step_at_iter = iter_idx * args.batch_size
        opponent_active = global_step_at_iter >= opponent_start_step

        # -- Scripted-PID mixing fraction: linear anneal, held constant at *_end afterwards. --
        pid_prob_now = None
        if use_pid_opponents:
            frac = jnp.clip(global_step_at_iter / max(args.opponent_pid_decay_steps, 1), 0.0, 1.0)
            pid_prob_now = args.opponent_pid_prob_start + frac * (
                args.opponent_pid_prob_end - args.opponent_pid_prob_start
            )

        # -- Rollout (ego trained, opponents frozen from the pool / scripted-PID once active) --
        rng, roll_rng = jax.random.split(rng)
        (env, last_obs, _, ep_ret, ep_len, opp_idx, ep_step, finish_step, traj_state), outs = (
            rollout(
                params,
                pool,
                filled,
                write_ptr,
                env,
                obs,
                roll_rng,
                ep_ret,
                ep_len,
                opp_idx,
                ep_step,
                finish_step,
                opponent_active,
                traj_state,
                pid_prob_now,
            )
        )

        # -- GAE (ego only) --
        d = outs["done"]
        dones_buf = jnp.concatenate([prev_done[None], d[:-1]], axis=0)
        next_done = d[-1]
        next_value = get_value(params, last_obs[:, 0])  # ego value bootstrap
        advantages, returns = compute_gae(
            outs["reward"], outs["value"], dones_buf, next_value, next_done
        )

        flat_data = (
            outs["obs"].reshape((-1,) + obs_shape),
            outs["action"].reshape((-1,) + act_shape),
            outs["logprob"].reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
            outs["value"].reshape(-1),
        )

        if args.anneal_ent_coef:
            ent_coef_now = args.ent_coef * (1.0 - iter_idx / args.num_iterations)
        else:
            ent_coef_now = jnp.asarray(args.ent_coef, dtype=jnp.float32)

        # -- PPO update (ego) --
        rng, upd_rng = jax.random.split(rng)
        params, opt_state, auxs = update_epochs(params, opt_state, flat_data, upd_rng, ent_coef_now)

        # -- Self-play snapshot: write current ego params into the pool on the snapshot cadence --
        # Unconditional on opponent_active -- the pool keeps filling from step 0 so a real (if
        # still-early) opponent is ready the moment activation flips.
        is_snapshot = ((iter_idx + 1) % snapshot_every) == 0
        pool, write_ptr, filled = jax.lax.cond(
            is_snapshot,
            lambda: snapshot_pool(pool, write_ptr, filled, params),
            lambda: (pool, write_ptr, filled),
        )

        # -- Per-iteration metrics (ego) --
        last = jax.tree.map(lambda x: x[-1], auxs)
        pg_arr, v_arr, ent_arr, kl_arr, ratio_arr = last
        y_pred = outs["value"].reshape(-1)
        y_true = returns.reshape(-1)
        var_y = jnp.var(y_true)
        explained_var = jnp.where(var_y == 0, jnp.nan, 1.0 - jnp.var(y_true - y_pred) / var_y)

        n_done = jnp.maximum(jnp.sum(outs["done"]), 1.0)
        ep_return = jnp.sum(outs["ret_done"]) / n_done
        ep_length = jnp.sum(outs["len_done"]) / n_done
        gates_passed = jnp.sum(outs["gates_done"]) / n_done
        completed = jnp.sum(outs["completed_done"]) / n_done
        # Win rate is per env-episode (all drones settled), so it uses its own episode count.
        n_ep = jnp.maximum(jnp.sum(outs["ep_done"]), 1.0)
        win_rate = jnp.sum(outs["win_done"]) / n_ep
        score = gates_passed if n_gates is not None else ep_return

        improved = score > best_score
        best_params = jax.tree.map(lambda b, p: jnp.where(improved, p, b), best_params, params)
        best_score = jnp.where(improved, score, best_score)

        value_loss = jnp.mean(v_arr)
        approx_kl = jnp.mean(kl_arr)
        metrics = {
            "loss/value_loss": value_loss,
            "loss/policy_loss": jnp.mean(pg_arr),
            "loss/entropy": jnp.mean(ent_arr),
            "loss/approx_kl": approx_kl,
            "loss/clipfrac": jnp.mean(
                (jnp.abs(ratio_arr - 1.0) > args.clip_coef).astype(jnp.float32)
            ),
            "loss/explained_variance": explained_var,
            "charts/ent_coef": ent_coef_now,
            "charts/pool_filled": filled.astype(jnp.float32),
            "charts/opponent_active": opponent_active.astype(jnp.float32),
            "train/reward": ep_return,
            "train/episode_length": ep_length,
            "train/gates_passed": gates_passed,
            "train/completed": completed,
            "train/win_rate": win_rate,
            best_key: best_score,
        }
        if use_pid_opponents:
            metrics["charts/opponent_pid_prob"] = pid_prob_now
        for k in metric_keys:
            name = f"reward/{k[len('rew/') :]}" if k.startswith("rew/") else k
            metrics[name] = jnp.sum(outs["metrics"][k]) / denom
        for k in max_keys:
            metrics[f"diagnostics/{k[len('max/') :]}_max"] = jnp.max(outs["metrics_max"][k])

        last_iter = iter_idx == args.num_iterations - 1
        if console_live:
            should_log = ((iter_idx + 1) % args.console_log_interval == 0) | last_iter
            jax.lax.cond(
                should_log,
                lambda: jax.debug.callback(
                    _log_iter,
                    iter_idx,
                    ep_return,
                    gates_passed,
                    value_loss,
                    approx_kl,
                    ordered=True,
                ),
                lambda: None,
            )
        if wandb_live:
            should_pub = ((iter_idx + 1) % args.wandb_log_interval == 0) | last_iter
            jax.lax.cond(
                should_pub,
                lambda: jax.debug.callback(
                    _wandb_log, (iter_idx + 1) * args.batch_size, metrics, ordered=True
                ),
                lambda: None,
            )
        if checkpoint_live:
            should_save = ((iter_idx + 1) % save_every) == 0
            jax.lax.cond(
                should_save,
                lambda: jax.debug.callback(
                    _save_checkpoint, (iter_idx + 1) * args.batch_size, params, score, ordered=True
                ),
                lambda: None,
            )

        new_carry = (
            params,
            opt_state,
            env,
            last_obs,
            next_done,
            rng,
            ep_ret,
            ep_len,
            pool,
            write_ptr,
            filled,
            opp_idx,
            best_params,
            best_score,
            ep_step,
            finish_step,
            traj_state,
        )
        return new_carry, metrics

    # ------------------------------------------------------------------ #
    # Whole-run scan                                                     #
    # ------------------------------------------------------------------ #
    env0, (next_obs, _) = envs.reset(envs, seed=args.seed)
    # Pool seeded with the initial ego params in every slot (slot 0 counts as the first snapshot).
    pool0 = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (pool_size,) + leaf.shape).copy(), params
    )
    rng, idx_rng = jax.random.split(rng)
    opp_idx0 = jnp.zeros((num_envs, n_opponents), dtype=jnp.int32)  # all use slot 0 initially
    traj_state0 = ()
    if use_pid_opponents:
        rng, speed_key, pid_key = jax.random.split(rng, 3)
        traj_t0 = jnp.zeros((num_envs, n_opponents))  # virtual elapsed trajectory time
        if use_random_pid_start:
            rng, spawn_key = jax.random.split(rng)
            traj_t0 = sample_spawn_t(spawn_key, (num_envs, n_opponents))
        traj_state0 = (
            traj_t0,
            jax.random.uniform(
                speed_key,
                (num_envs, n_opponents),
                minval=args.opponent_pid_speed_min,
                maxval=args.opponent_pid_speed_max,
            ),  # traj_speed
            jnp.zeros((num_envs, n_opponents, 3)),  # traj_i_error
            jax.random.bernoulli(
                pid_key, args.opponent_pid_prob_start, (num_envs, n_opponents)
            ),  # is_pid_opp
        )
        if use_random_pid_start:
            # The very first episodes should be randomized too: envs.reset above spawned every
            # drone on the pad, so move the PID slots onto the spline at their sampled t0 (the
            # in-rollout teleport only fires on autoresets). next_obs keeps the pad position for
            # frame 0, same accepted one-frame staleness as in the rollout.
            traj_t0, traj_speed0, _, is_pid_opp0 = traj_state0
            env0 = teleport_opponents(env0, traj_pid, traj_t0, traj_speed0, is_pid_opp0)
    init_carry = (
        params,
        opt_state,
        env0,
        next_obs,
        jnp.zeros(num_envs),  # prev_done (ego)
        rng,
        jnp.zeros(num_envs),  # ep_ret
        jnp.zeros(num_envs),  # ep_len
        pool0,
        jnp.asarray(1 % pool_size, dtype=jnp.int32),  # write_ptr (next slot after seed)
        jnp.asarray(1, dtype=jnp.int32),  # filled (seed counts as one)
        opp_idx0,
        params,  # best_params
        jnp.asarray(-jnp.inf, dtype=jnp.float32),  # best_score
        jnp.zeros(num_envs, dtype=jnp.int32),  # ep_step (within-episode step counter)
        -jnp.ones(
            (num_envs, n_drones), dtype=jnp.int32
        ),  # finish_step (per drone; -1 = unfinished)
        traj_state0,
    )
    print(f"Compiling and running {args.num_iterations} iterations as one scan...")
    final_carry, metrics_stack = jax.lax.scan(
        train_iteration, init_carry, jnp.arange(args.num_iterations)
    )
    jax.block_until_ready(metrics_stack)
    best_params = final_carry[12]
    best_score = float(final_carry[13])
    global_step = args.num_iterations * args.batch_size
    train_end_time = time.time()
    sps = int(global_step / (train_end_time - train_start_time)) if global_step else 0
    print(
        f"Training for {global_step} steps took {train_end_time - train_start_time:.2f}s "
        f"({sps} steps/s)."
    )

    if wandb_enabled and not wandb_live:
        metrics_host = jax.device_get(metrics_stack)
        for it in range(args.num_iterations):
            log = {k: float(v[it]) for k, v in metrics_host.items()}
            wandb.log(log, step=(it + 1) * args.batch_size)

    model_path = None
    if checkpoint_dir is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = checkpoint_dir / f"{run_name}_{timestamp}_{tag}{best_score:.2f}_best.ckpt"
        nnx.update(agent, best_params)
        with open(model_path, "wb") as f:
            pickle.dump(nnx.state(agent, nnx.Param), f)
        print(f"Best model (mean {metric} {best_score:.2f}) saved to {model_path}")
    envs.close()
    return model_path


def evaluate_ippo(
    args: Args, make_env: MakeEnv, config: Any, n_eval: int, model_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a trained ego policy headless in the multi-drone env, opponents = the same policy.

    Each of ``n_eval`` parallel envs runs one episode; the ego (drone 0) plays its deterministic
    (mean) policy, and the opponents use the *same* loaded policy (self-play against the final ego).
    Reward/length are masked after each env's first ego ``done`` to isolate a single episode.
    """
    set_seeds(args.seed)
    eval_env = make_env(dataclasses.replace(args, num_envs=n_eval), config)
    n_drones = eval_env.unwrapped.n_drones
    action_dim = int(np.prod(eval_env.single_action_space.shape))
    obs_dim = int(np.prod(eval_env.single_observation_space.shape))
    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(0))
    with open(model_path, "rb") as f:
        nnx.update(agent, pickle.load(f))
    graphdef, params = nnx.split(agent)

    def act(obs: Array) -> Array:
        """Deterministic mean action for every drone (ego + opponents = the loaded policy)."""
        flat = obs.reshape(n_eval * n_drones, obs_dim)
        mean, _, _ = nnx.merge(graphdef, params)(flat)
        return mean.reshape(n_eval, n_drones, action_dim)

    max_steps = int(getattr(args, "max_episode_length", 2000))

    def eval_scan(carry: tuple, _: Any) -> tuple[tuple, None]:
        env, obs, done_so_far, ret, length = carry
        action = act(obs)
        env, (next_obs, reward, term, trunc, _) = env.step(env, action)
        alive = 1.0 - done_so_far.astype(jnp.float32)
        ret = ret + reward[:, 0] * alive  # ego reward
        length = length + alive
        done_so_far = done_so_far | jnp.logical_or(term[:, 0], trunc[:, 0])
        return (env, next_obs, done_so_far, ret, length), None

    env0, (obs0, _) = eval_env.reset(eval_env, seed=args.seed)
    carry0 = (env0, obs0, jnp.zeros(n_eval, dtype=bool), jnp.zeros(n_eval), jnp.zeros(n_eval))
    (_, _, _, ret, length), _ = jax.lax.scan(eval_scan, carry0, None, length=max_steps)
    eval_env.close()
    return np.array(ret), np.array(length)
