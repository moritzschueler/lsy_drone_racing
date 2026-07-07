"""Generic PPO training and evaluation, shared across all drone RL tasks.

The algorithm is task-agnostic: it is parameterized by a ``make_env`` factory (which builds the
vectorized, fully-wrapped *functional* environment for a given task) and the shared ``Agent``
network. An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/.

The whole training run is a single ``lax.scan`` over iterations: each iteration is a nested scan
(rollout) feeding a nested scan (PPO update epochs), and every metric is a fixed-shape scan output.
Nothing syncs to the host inside the scan -- all wandb metrics are flushed once, after the run, and
the best checkpoint is tracked in the scan carry (no mid-run pickling). This trades live curves,
per-iteration checkpoints and graceful Ctrl-C for a single compiled XLA program that keeps the GPU
saturated. See the functional env in ``rl/wrappers/`` and ``make_functional_env``.
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
from flax import nnx
from jax import Array

import wandb
from lsy_drone_racing.rl.agents.ppo_agent import Agent, _entropy, _log_prob
from lsy_drone_racing.rl.config import Args
from lsy_drone_racing.rl.git_provenance import pin_run_to_branch
from lsy_drone_racing.rl.wrappers.wrapper_base import Wrapper
from lsy_drone_racing.utils.utils import set_seeds

# Env factory: (args, config) -> fully-wrapped functional env (a ``Wrapper``/``struct.PyTreeNode``).
MakeEnv = Callable[[Args, Any], Wrapper]


def train_ppo(
    args: Args,
    make_env: MakeEnv,
    config: Any,
    checkpoint_dir: Path | None,
    run_name: str,
    wandb_enabled: bool = False,
) -> Path | None:
    """Train a PPO agent on the functional environment produced by ``make_env``.

    The entire run is compiled as one ``lax.scan`` over ``args.num_iterations``; metrics are
    accumulated as scan outputs and flushed to wandb after the scan returns.
    The best checkpoint is tracked in the scan carry and written at the end.
    """
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), group="single_agent_racing")
        # Pin the exact code behind this run to a branch wandb-runs/<name>-<id> so the chart
        # legend maps straight to reproducible code. Best-effort; never aborts training.
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
    print("Training on device:", args.jax_device)
    print("--- Hyperparameters ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-----------------------")

    # -- Functional env setup --
    envs = make_env(args, config)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    num_envs = args.num_envs
    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    action_dim = int(np.prod(act_shape))
    obs_dim = int(np.prod(obs_shape))
    print(f"Shape of observation space: {obs_shape}")
    print(f"Shape of action space: {act_shape}")

    base_space = envs.unwrapped.single_observation_space
    n_gates = (
        base_space["gates_pos"].shape[0]
        if hasattr(base_space, "spaces") and "gates_pos" in base_space.spaces
        else None
    )

    # -- Agent + optimizer (functional) --
    # NNX owns the weights statefully; split once into a static graphdef + a param ``State`` pytree,
    # then thread ``params`` (and the optax state) through the scan carry. ``nnx.merge`` rebuilds
    # the module for the forward pass and is differentiable w.r.t. ``params``.
    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(args.seed))
    graphdef, params = nnx.split(agent)
    rng = jax.random.PRNGKey(args.seed)

    if args.anneal_lr:
        schedule = optax.cosine_decay_schedule(args.learning_rate, args.num_iterations * args.update_epochs * args.num_minibatches)
    else:
        schedule = args.learning_rate
    tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adamw(schedule, eps=1e-5))
    opt_state = tx.init(params)

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
    # The functional wrappers stash per-env reward components ("rew/...") and behavioural
    # diagnostics ("diagnostics/...") in info, mean-reduced over the rollout, plus "max/..." keys
    # reduced with a running max (logged as the iteration peak). The key set is static, so probe one
    # (functional, non-mutating) step to capture it for the fixed-shape scan outputs.
    _, (_, _, _, _, _probe_info) = envs.step(envs, jnp.zeros((num_envs, action_dim)))
    metric_keys = sorted(
        k
        for k in _probe_info
        if isinstance(k, str) and (k.startswith("rew/") or k.startswith("diagnostics/"))
    )
    max_keys = sorted(k for k in _probe_info if isinstance(k, str) and k.startswith("max/"))
    denom = float(args.num_steps * num_envs)

    def rollout(
        params: nnx.State, env: Wrapper, obs: Array, rng: Array, ep_ret: Array, ep_len: Array
    ) -> tuple[tuple, dict]:
        """One rollout of ``num_steps`` as an inner scan; episode stats tracked inline in carry."""

        def step(carry: tuple, _: Any) -> tuple[tuple, dict]:
            env, obs, rng, ep_ret, ep_len = carry
            rng, akey = jax.random.split(rng)
            action, logprob, value = policy_step(params, obs, akey)
            env, (next_obs, reward, term, trunc, info) = env.step(env, action)
            done = jnp.logical_or(term, trunc)
            donef = done.astype(jnp.float32)

            new_ret = ep_ret + reward
            new_len = ep_len + 1.0
            gates_passed = jnp.where(info["target_gate"] == -1, n_gates or 0, info["target_gate"])
            gates_passed = gates_passed.astype(jnp.float32)
            out = {
                "obs": obs,
                "action": action,
                "logprob": logprob,
                "value": value,
                "reward": reward,
                "done": donef,
                # Episode-end masked quantities (0 except on the step an env finishes).
                "ret_done": jnp.where(done, new_ret, 0.0),
                "len_done": jnp.where(done, new_len, 0.0),
                "gates_done": jnp.where(done, gates_passed, 0.0),
                "completed_done": jnp.where(
                    done, (gates_passed == (n_gates or -1)).astype(jnp.float32), 0.0
                ),
                # Per-step reward-component / diagnostic sums over envs (scalar each).
                "metrics": {k: jnp.sum(info[k]) for k in metric_keys},
                # Per-step env-max of each "max/..." key (peak-reduced over the rollout later).
                "metrics_max": {k: jnp.max(info[k]) for k in max_keys},
            }
            ep_ret = jnp.where(done, 0.0, new_ret)
            ep_len = jnp.where(done, 0.0, new_len)
            return (env, next_obs, rng, ep_ret, ep_len), out

        carry0 = (env, obs, rng, ep_ret, ep_len)
        (env, last_obs, rng, ep_ret, ep_len), outs = jax.lax.scan(
            step, carry0, None, length=args.num_steps
        )
        return (env, last_obs, rng, ep_ret, ep_len), outs

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
                # Freeze: once this epoch's predecessor crossed target_kl, keep params/opt_state
                # unchanged (the grads above are computed but discarded). Mirrors the old per-epoch
                # ``break`` -- the epoch that crosses still applies, later epochs are no-ops.
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

    # Which metric key holds the best-checkpoint score, and the chart it logs to.
    best_key = "charts/gates_passed" if n_gates is not None else "charts/best_reward"

    # Periodic in-scan logging. Both are Python (compile-time) flags: when disabled (<=0) the
    # corresponding callback is never emitted into the traced graph, so it adds zero overhead.
    console_live = args.console_log_interval > 0  # console progress line
    wandb_live = wandb_enabled and args.wandb_log_interval > 0  # live wandb publishing mid-run
    if console_live:
        print(f"Console progress every {args.console_log_interval} iterations")
    if wandb_live:
        print(f"Publishing metrics to wandb every {args.wandb_log_interval} iterations (live)")

    def _log_iter(
        iter_idx: Array, reward: Array, gates: Array, value_loss: Array, approx_kl: Array
    ) -> None:
        """Host-side progress line; called by jax.debug.callback with materialized arrays."""
        elapsed = time.time() - train_start_time
        print(
            f"Iteration {int(iter_idx) + 1}/{args.num_iterations} | "
            f"reward {float(reward):+.2f} | gates {float(gates):.2f} | "
            f"v_loss {float(value_loss):.3f} | kl {float(approx_kl):.4f} | {elapsed:.1f}s"
        )

    def _wandb_log(step: Array, metrics: dict) -> None:
        """Host-side wandb.log of one iteration's metrics; called via an ordered callback."""
        wandb.log({k: float(v) for k, v in metrics.items()}, step=int(step))

    def train_iteration(carry: tuple, iter_idx: Array) -> tuple[tuple, dict]:
        params, opt_state, env, obs, prev_done, rng, ep_ret, ep_len, best_params, best_score = carry

        # -- Rollout --
        rng, roll_rng = jax.random.split(rng)
        (env, last_obs, _, ep_ret, ep_len), outs = rollout(
            params, env, obs, roll_rng, ep_ret, ep_len
        )

        # -- GAE -- 
        d = outs["done"]  # (T, E) post-step done
        dones_buf = jnp.concatenate([prev_done[None], d[:-1]], axis=0)
        next_done = d[-1]
        next_value = get_value(params, last_obs)
        advantages, returns = compute_gae(
            outs["reward"], outs["value"], dones_buf, next_value, next_done
        )

        # -- Flatten batch --
        flat_data = (
            outs["obs"].reshape((-1,) + obs_shape),
            outs["action"].reshape((-1,) + act_shape),
            outs["logprob"].reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
            outs["value"].reshape(-1),
        )

        # Anneal entropy bonus linearly to 0 over training when enabled.
        if args.anneal_ent_coef:
            ent_coef_now = args.ent_coef * (1.0 - iter_idx / args.num_iterations)
        else:
            ent_coef_now = jnp.asarray(args.ent_coef, dtype=jnp.float32)

        # -- PPO update --
        rng, upd_rng = jax.random.split(rng)
        params, opt_state, auxs = update_epochs(params, opt_state, flat_data, upd_rng, ent_coef_now)

        # -- Per-iteration metrics --
        # Report the last epoch's per-minibatch losses (post-update state).
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
        score = gates_passed if n_gates is not None else ep_return

        # -- Best-checkpoint snapshot (tracked in carry; no host pickling mid-scan) --
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
            "train/reward": ep_return,
            "train/episode_length": ep_length,
            "train/gates_passed": gates_passed,
            "train/completed": completed,
            best_key: best_score,
        }
        # Reward components -> reward/<name>; diagnostics keep their prefix. Mean per step.
        for k in metric_keys:
            name = f"reward/{k[len('rew/'):]}" if k.startswith("rew/") else k
            metrics[name] = jnp.sum(outs["metrics"][k]) / denom
        # Max-reduced metrics -> diagnostics/<name>_max: the rollout peak (e.g. max/vel).
        for k in max_keys:
            metrics[f"diagnostics/{k[len('max/'):]}_max"] = jnp.max(outs["metrics_max"][k])

        # Periodic in-scan side effects via ordered external callbacks. Each is gated by lax.cond so
        # the host round-trip happens only on the logged iterations (plus the last) -- the rest of
        # the run stays host-sync-free. The enclosing `if` flags are compile-time, so a disabled
        # channel emits no callback at all (zero overhead). ordered=True keeps them ordered.
        last_iter = iter_idx == args.num_iterations - 1
        if console_live:
            should_log = ((iter_idx + 1) % args.console_log_interval == 0) | last_iter
            jax.lax.cond(
                should_log,
                lambda: jax.debug.callback(
                    _log_iter, iter_idx, ep_return, gates_passed, value_loss, approx_kl,
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

        new_carry = (
            params, opt_state, env, last_obs, next_done, rng, ep_ret, ep_len,
            best_params, best_score,
        )
        return new_carry, metrics

    # ------------------------------------------------------------------ #
    # Whole-run scan                                                     #
    # ------------------------------------------------------------------ #
    env0, (next_obs, _) = envs.reset(envs, seed=args.seed)
    init_carry = (
        params,
        opt_state,
        env0,
        next_obs,
        jnp.zeros(num_envs),  # prev_done
        rng,
        jnp.zeros(num_envs),  # ep_ret
        jnp.zeros(num_envs),  # ep_len
        params,  # best_params (init to current; replaced on first improvement)
        jnp.asarray(-jnp.inf, dtype=jnp.float32),  # best_score
    )
    print(f"Compiling and running {args.num_iterations} iterations as one scan...")
    final_carry, metrics_stack = jax.lax.scan(
        train_iteration, init_carry, jnp.arange(args.num_iterations)
    )
    jax.block_until_ready(metrics_stack)
    best_params = final_carry[8]
    best_score = float(final_carry[9])
    global_step = args.num_iterations * args.batch_size
    train_end_time = time.time()
    sps = int(global_step / (train_end_time - train_start_time)) if global_step else 0
    print(
        f"Training for {global_step} steps took {train_end_time - train_start_time:.2f}s "
        f"({sps} steps/s)."
    )

    # -- Flush all per-iteration metrics to wandb in one batch (after training) --
    # Skipped when live publishing is on: the in-scan callback already logged the (down)sampled
    # points, and re-flushing 1..N here would violate wandb's monotonically-increasing step order.
    if wandb_enabled and not wandb_live:
        metrics_host = jax.device_get(metrics_stack)
        for it in range(args.num_iterations):
            log = {k: float(v[it]) for k, v in metrics_host.items()}
            wandb.log(log, step=(it + 1) * args.batch_size)

    # -- Save the best checkpoint --
    model_path = None
    if checkpoint_dir is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        tag = "g" if n_gates is not None else "r"
        metric = "gates-passed" if n_gates is not None else "return"
        model_path = checkpoint_dir / f"{run_name}_{timestamp}_{tag}{best_score:.2f}_best.ckpt"
        nnx.update(agent, best_params)
        with open(model_path, "wb") as f:
            pickle.dump(nnx.state(agent, nnx.Param), f)
        print(f"Best model (mean {metric} {best_score:.2f}) saved to {model_path}")
    envs.close()
    return model_path


def evaluate_ppo(
    args: Args, make_env: MakeEnv, config: Any, n_eval: int, model_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a trained PPO agent headless: ``n_eval`` parallel envs, one episode each.

    Rendering lives in ``rl/scripts/render.py`` and is intentionally not handled here. Each env
    runs the deterministic (mean) policy for one episode; the env autoresets internally, so reward
    and length are masked after each env's first ``done`` to isolate a single episode.
    """
    set_seeds(args.seed)
    eval_env = make_env(dataclasses.replace(args, num_envs=n_eval), config)
    action_dim = int(np.prod(eval_env.single_action_space.shape))

    obs_dim = int(np.prod(eval_env.single_observation_space.shape))
    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(0))
    with open(model_path, "rb") as f:
        nnx.update(agent, pickle.load(f))
    graphdef, params = nnx.split(agent)

    max_steps = int(getattr(args, "max_episode_length", 2000))

    def eval_scan(carry: tuple, _: Any) -> tuple[tuple, None]:
        env, obs, done_so_far, ret, length = carry
        mean, _, _ = nnx.merge(graphdef, params)(obs)
        env, (next_obs, reward, term, trunc, _) = env.step(env, mean)
        alive = 1.0 - done_so_far.astype(jnp.float32)
        ret = ret + reward * alive
        length = length + alive
        done_so_far = done_so_far | jnp.logical_or(term, trunc)
        return (env, next_obs, done_so_far, ret, length), None

    env0, (obs0, _) = eval_env.reset(eval_env, seed=args.seed)
    carry0 = (
        env0,
        obs0,
        jnp.zeros(n_eval, dtype=bool),
        jnp.zeros(n_eval),
        jnp.zeros(n_eval),
    )
    (_, _, _, ret, length), _ = jax.lax.scan(eval_scan, carry0, None, length=max_steps)
    episode_rewards = np.array(ret)
    episode_lengths = np.array(length)

    eval_env.close()
    return episode_rewards, episode_lengths
