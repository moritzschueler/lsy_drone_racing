"""Generic PPO training and evaluation, shared across all drone RL tasks.

The algorithm is task-agnostic: it is parameterized by a ``make_env`` factory (which builds
the vectorized, fully-wrapped environment for a given task) and the shared ``Agent`` network.
An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/.
"""

import pickle
import random
import signal
import time
import warnings
from pathlib import Path
from typing import Callable

import glfw
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import nnx
from gymnasium.vector import VectorEnv
from jax import Array

from lsy_drone_racing.rl.agents.ppo_agent import Agent, _entropy, _log_prob
from lsy_drone_racing.rl.config import Args

MakeEnv = Callable[[Args, int, str], VectorEnv]


def set_seeds(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)


def train_ppo(
    args: Args,
    make_env: MakeEnv,
    checkpoint_dir: Path | None,
    run_name: str,
    jax_device: str,
    wandb_enabled: bool = False,
) -> Path | None:
    """Train a PPO agent on the environment produced by ``make_env``.

    The checkpoint is saved into ``checkpoint_dir`` under
    ``{run_name}_{timestamp}_r{reward}.ckpt`` so multiple runs are distinguishable, and the
    saved path is returned (None if ``checkpoint_dir`` is None). ``reward`` is the mean of the
    most recent episode returns, or ``NA`` if no episodes finished/were tracked (wandb off).
    """
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    train_start_time = time.time()
    set_seeds(args.seed)
    print("Training on device:", jax_device)
    # Log the full configuration so the exact values used are recoverable from the run logs
    # (wandb.init also records these under the run's config).
    print("--- Hyperparameters ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-----------------------")

    # env setup
    envs = make_env(args, args.num_envs, jax_device)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    # Curriculum hook: any wrapper in the chain exposing ``set_progress`` is fed training progress
    # (global_step / total_timesteps) each iteration. Check the class (not the instance) so the
    # gymnasium attribute proxy doesn't report inner wrappers' methods on outer ones.
    progress_sinks = []
    _e = envs
    while _e is not None:
        if getattr(type(_e), "set_progress", None) is not None:
            progress_sinks.append(_e)
        _e = getattr(_e, "env", None)

    def set_progress(step: int) -> None:
        frac = min(step / args.total_timesteps, 1.0)
        for sink in progress_sinks:
            sink.set_progress(frac)

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    action_dim = int(np.prod(act_shape))

    print(f"Shape of observation shape: {obs_shape}")
    print(f"Shape of action space: {act_shape}")

    # Number of gates, for the racing completion metric (None for tasks without gates).
    base_space = envs.unwrapped.single_observation_space
    n_gates = (
        base_space["gates_pos"].shape[0]
        if hasattr(base_space, "spaces") and "gates_pos" in base_space.spaces
        else None
    )

    # Agent init. NNX owns the weights statefully (PyTorch-style); the model instance is
    # passed directly into the jitted functions below.
    obs_dim = int(np.prod(obs_shape))
    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(args.seed))
    rng = jax.random.PRNGKey(args.seed)

    # Optimizer: clip grads then update. nnx.Optimizer wraps optax and manages the
    # optimizer state for us, mutating the model in place on optimizer.update(...).
    if args.anneal_lr:
        schedule = optax.linear_schedule(
            args.learning_rate, 0.0, args.num_iterations * args.update_epochs * args.num_minibatches
        )
    else:
        schedule = args.learning_rate
    tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adamw(schedule, eps=1e-5))
    optimizer = nnx.Optimizer(agent, tx, wrt=nnx.Param)

    # ------------------------------------------------------------------ #
    # JIT-compiled inner functions (capture args, tx via closure)         #
    # nnx.jit threads the model's state through the compiled program.      #
    # ------------------------------------------------------------------ #

    @nnx.jit
    def policy_step(agent: Agent, obs: Array, rng_key: Array) -> tuple[Array, Array, Array]:
        mean, log_std, value = agent(obs)
        std = jnp.exp(log_std)
        action = mean + std * jax.random.normal(rng_key, mean.shape)
        logprob = _log_prob(action, mean, log_std)
        return action, logprob, value.squeeze(-1)

    @nnx.jit
    def get_value(agent: Agent, obs: Array) -> Array:
        _, _, value = agent(obs)
        return value.squeeze(-1)

    @jax.jit
    def compute_gae(
        rewards: Array, values: Array, dones: Array, next_value: Array, next_done: Array
    ) -> tuple[Array, Array]:
        """GAE via lax.scan over reversed time steps."""

        def scan_fn(lastgaelam: Array, inputs: tuple) -> tuple[Array, Array]:
            reward, value, next_val, next_d = inputs
            nextnonterminal = 1.0 - next_d
            delta = reward + args.gamma * next_val * nextnonterminal - value
            advantage = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            return advantage, advantage

        # next_val and next_done for each step t: values[t+1] / dones[t+1], last step uses bootstrap
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
        agent: Agent,
        obs: Array,
        actions: Array,
        log_probs: Array,
        advantages: Array,
        returns: Array,
        b_values: Array,
        ent_coef: Array,
    ) -> tuple[Array, tuple]:
        mean, log_std, new_values = agent(obs)
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

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        # Value loss
        new_values_flat = new_values.reshape(-1)
        if args.clip_vloss:
            v_clipped = b_values + jnp.clip(
                new_values_flat - b_values, -args.clip_coef, args.clip_coef
            )
            v_loss = 0.5 * jnp.mean(
                jnp.maximum(
                    (new_values_flat - returns) ** 2,
                    (v_clipped - returns) ** 2,
                )
            )
        else:
            v_loss = 0.5 * jnp.mean((new_values_flat - returns) ** 2)

        entropy_loss = jnp.mean(entropy)
        total_loss = pg_loss - ent_coef * entropy_loss + args.vf_coef * v_loss
        return total_loss, (pg_loss, v_loss, entropy_loss, approx_kl, ratio)

    @nnx.jit
    def update_epoch(
        agent: Agent,
        optimizer: nnx.Optimizer,
        flat_data: tuple,
        rng_key: Array,
        ent_coef: Array,
    ) -> tuple:
        """One full epoch of minibatch PPO updates compiled as a single XLA program.

        nnx.Optimizer mutates ``agent`` in place each minibatch; the loop over
        minibatches is a plain Python loop (it unrolls into the trace), so we collect
        per-minibatch metrics in lists and stack them to mirror the old lax.scan output.
        """
        shuffled = jax.random.permutation(rng_key, args.batch_size)
        mb_inds = shuffled.reshape(args.num_minibatches, args.minibatch_size)

        losses, pgs, vs, ents, kls, ratios = [], [], [], [], [], []
        for i in range(args.num_minibatches):
            mb = tuple(x[mb_inds[i]] for x in flat_data)
            (loss, aux), grads = nnx.value_and_grad(ppo_loss_fn, has_aux=True)(agent, *mb, ent_coef)
            optimizer.update(agent, grads)
            pg, v_loss, ent, kl, ratio = aux
            losses.append(loss)
            pgs.append(pg)
            vs.append(v_loss)
            ents.append(ent)
            kls.append(kl)
            ratios.append(ratio)

        metrics = (
            jnp.stack(losses),
            (jnp.stack(pgs), jnp.stack(vs), jnp.stack(ents), jnp.stack(kls), jnp.stack(ratios)),
        )
        return metrics

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #
    global_step = 0
    set_progress(global_step)  # activate the curriculum before the initial reset
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = jnp.asarray(next_obs)
    next_done = jnp.zeros(args.num_envs)
    sum_rewards = jnp.zeros(args.num_envs)
    ep_lengths = jnp.zeros(args.num_envs)  # per-env step counter; reset on done (survival time)
    sum_rewards_hist: list[float] = []
    # Returns of true-start (deployment-like) episodes; with the curriculum, mid-track cone spawns
    # are excluded so "best" reflects full-track performance. Falls back to all episodes when the
    # episode_true_start flag is absent (other tasks / no curriculum).
    true_start_hist: list[float] = []
    # Per true-start episode racing *progress* = gates passed (n_gates if the track was completed).
    # Skill measure on the fixed deployment distribution, unaffected by reward shaping (crash/timeout/
    # smoothness penalties), so unlike return it rises with competence and peaks late rather than
    # early. Drives best-checkpoint + early-stop; empty for non-racing tasks (then we fall back to
    # return). See best_score selection below.
    true_start_progress_hist: list[float] = []
    # Best checkpoint tracking: snapshot the params whenever the recent mean true-start *score*
    # (gates-passed for racing, else return) improves, then write that snapshot (not the final
    # state) at the end.
    best_score = -float("inf")
    best_is_progress = False  # True once gates-passed history exists (racing); else return-based
    best_state_bytes: bytes | None = None
    best_window, best_min_episodes = 100, 20

    # -- Graceful early stop --
    # Ctrl-C requests a stop: the current iteration finishes, then we break to the best-checkpoint
    # save (evaluation/render then runs on it). A second Ctrl-C forces an immediate abort.
    stop_requested = {"flag": False}

    def _handle_sigint(signum, frame):
        if stop_requested["flag"]:
            raise KeyboardInterrupt  # second press -> hard abort
        stop_requested["flag"] = True
        print(
            "\nStop requested (Ctrl-C): finishing current iteration, then saving the best "
            "checkpoint. Press Ctrl-C again to force quit."
        )

    prev_sigint = signal.signal(signal.SIGINT, _handle_sigint)

    for iteration in range(1, args.num_iterations + 1):
        start_time = time.time()
        set_progress(global_step)  # advance the curriculum schedules (kappa, p_start_position)

        # -- Rollout --
        obs_list, act_list, logp_list, val_list, rew_list, done_list = [], [], [], [], [], []
        # Cone-spawn gate-pass tally for this iteration: of curriculum (non-true-start) episodes that
        # finish, how many advanced past the gate they were spawned in front of.
        cone_passed, cone_total = 0, 0
        # Start-mix tally for this iteration: how many finished episodes started from the true race
        # start vs a cone spawn. The realized true-start fraction should track p_start_schedule(tau).
        true_start_count = 0
        # Per-step metric accumulators: each wrapper stashes a per-env value under a "rew/<name>"
        # (reward component) or "diagnostics/<name>" (diagnostic, e.g. commanded thrust / velocity)
        # info key; we sum them over the rollout (lazily, on-device) and log mean-per-step values so
        # the reward composition and the policy's behavior over training are visible.
        step_metric_sums: dict[str, Array] = {}

        for _ in range(args.num_steps):
            global_step += args.num_envs
            obs_list.append(next_obs)
            done_list.append(next_done)

            rng, action_key = jax.random.split(rng)
            action, logprob, value = policy_step(agent, next_obs, action_key)
            act_list.append(action)
            logp_list.append(logprob)
            val_list.append(value)

            next_obs, reward, terminations, truncations, info = envs.step(action)
            next_obs = jnp.asarray(next_obs)
            reward = jnp.asarray(reward)
            rew_list.append(reward)

            # Accumulate per-step metrics (reward components + diagnostics; kept on-device,
            # summed across steps & envs, divided by step*env count at log time -> mean per step).
            for key, val in info.items():
                if isinstance(key, str) and (key.startswith("rew/") or key.startswith("diagnostics/")):
                    step_metric_sums[key] = step_metric_sums.get(key, jnp.zeros(())) + jnp.sum(
                        jnp.asarray(val)
                    )

            # Track episode returns and lengths; reset done envs before accumulating
            done_mask = next_done.astype(bool)
            sum_rewards = jnp.where(done_mask, 0.0, sum_rewards)
            ep_lengths = jnp.where(done_mask, 0.0, ep_lengths)
            next_done = jnp.asarray(terminations | truncations).astype(jnp.float32)
            sum_rewards = sum_rewards + reward
            ep_lengths = ep_lengths + 1.0

            if bool(jnp.any(next_done)):
                done_indices = np.where(np.array(next_done, dtype=bool))[0]
                # target_gate at the terminal step = next gate to pass, or -1 if the whole track
                # was completed. Present only for the racing task.
                target_gate = np.array(info["target_gate"]) if "target_gate" in info else None
                # Curriculum spawns mid-track episodes that trivially "complete"; gate-progress
                # metrics and the best-checkpoint criterion are only meaningful for episodes that
                # started from the true race start. Absent (other tasks / no curriculum) -> treat
                # every episode as a true start.
                true_start = (
                    np.array(info["episode_true_start"], dtype=bool)
                    if "episode_true_start" in info
                    else None
                )
                # Start gate of each finishing episode (cone spawns); paired with the terminal
                # target_gate to detect whether the gate the drone was spawned in front of was passed.
                start_gate = (
                    np.array(info["episode_start_gate"]) if "episode_start_gate" in info else None
                )
                for idx in done_indices:
                    r = float(sum_rewards[idx])
                    sum_rewards_hist.append(r)
                    full_track = true_start is None or bool(true_start[idx])
                    # Gates passed this episode (n_gates if completed, g==-1); None when the task
                    # exposes no gate index (non-racing).
                    gates_passed = None
                    if target_gate is not None and n_gates is not None:
                        g = int(target_gate[idx])
                        gates_passed = float(n_gates if g == -1 else g)
                    if full_track:
                        true_start_hist.append(r)
                        if gates_passed is not None:
                            true_start_progress_hist.append(gates_passed)
                        # Count true-start episodes only when the curriculum is active (true_start
                        # present), so the start-mix fraction is meaningful for the racing task.
                        true_start_count += int(true_start is not None)
                    elif target_gate is not None and start_gate is not None:
                        # Cone-spawned episode: it passed its gate iff the target advanced past the
                        # one it started on (forward-only; terminal -1 = finished also counts).
                        cone_total += 1
                        cone_passed += int(int(target_gate[idx]) != int(start_gate[idx]))
                    if wandb_enabled:
                        log = {"train/reward": r, "train/episode_length": float(ep_lengths[idx])}
                        if full_track and gates_passed is not None:
                            log["train/gates_passed"] = gates_passed
                            log["train/completed"] = float(gates_passed == n_gates)
                        wandb.log(log, step=global_step)

        obs_buf = jnp.stack(obs_list)       # (T, E, obs_dim)
        act_buf = jnp.stack(act_list)       # (T, E, act_dim)
        logp_buf = jnp.stack(logp_list)     # (T, E)
        val_buf = jnp.stack(val_list)       # (T, E)
        rew_buf = jnp.stack(rew_list)       # (T, E)
        done_buf = jnp.stack(done_list)     # (T, E)

        # -- GAE --
        next_value = get_value(agent, next_obs)
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, next_value, next_done)

        # -- Flatten batch --
        flat_data = (
            obs_buf.reshape((-1,) + obs_shape),
            act_buf.reshape((-1,) + act_shape),
            logp_buf.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
            val_buf.reshape(-1),
        )

        # Anneal the entropy bonus linearly to 0 over training (mirrors anneal_lr) so the policy
        # explores early and sharpens late; a constant bonus was inflating the action spread as
        # advantages flattened on the hard cones, eroding precision. Passed as a traced scalar so
        # update_epoch isn't recompiled each iteration.
        ent_coef_now = (
            args.ent_coef * (1.0 - (iteration - 1) / args.num_iterations)
            if args.anneal_ent_coef
            else args.ent_coef
        )
        ent_coef_arr = jnp.asarray(ent_coef_now, dtype=jnp.float32)

        # -- PPO update epochs --
        last_metrics = None
        for _ in range(args.update_epochs):
            rng, epoch_key = jax.random.split(rng)
            metrics = update_epoch(agent, optimizer, flat_data, epoch_key, ent_coef_arr)
            last_metrics = metrics
            if args.target_kl is not None:
                approx_kl = float(jnp.mean(metrics[1][3]))
                if approx_kl > args.target_kl:
                    break

        # -- Logging --
        y_pred = np.array(val_buf.reshape(-1))
        y_true = np.array(returns.reshape(-1))
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        end_time = time.time()
        if wandb_enabled and last_metrics is not None:
            _, (pg_arr, v_arr, ent_arr, kl_arr, ratio_arr) = last_metrics
            log_metrics = {
                "losses/value_loss": float(jnp.mean(v_arr)),
                "losses/policy_loss": float(jnp.mean(pg_arr)),
                "losses/entropy": float(jnp.mean(ent_arr)),
                "losses/approx_kl": float(jnp.mean(kl_arr)),
                "losses/clipfrac": float(jnp.mean(jnp.abs(ratio_arr - 1.0) > args.clip_coef)),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (end_time - start_time)),
                "charts/ent_coef": float(ent_coef_now),
            }
            # Fraction (0-1) of cone-spawned episodes this iteration that passed their spawn gate.
            # Isolates curriculum progress from full-track skill; only logged when any cone episode
            # finished (i.e. the curriculum is active and producing cone spawns).
            if cone_total > 0:
                log_metrics["train/cone_gate_pass_rate"] = cone_passed / cone_total
            # Realized true-start fraction among finished episodes this iteration (cone fraction is
            # 1 - this); should track p_start_schedule(tau). n_starts > 0 only when the curriculum is
            # active (both counters increment solely when episode_true_start is present).
            n_starts = true_start_count + cone_total
            if n_starts > 0:
                log_metrics["train/true_start_frac"] = true_start_count / n_starts
            # Mean per-step value of each accumulated metric this iteration: reward components go
            # under reward/<name>, behavioral diagnostics keep their diagnostics/<name> prefix.
            denom = args.num_steps * args.num_envs
            for key, total in step_metric_sums.items():
                name = f"reward/{key[len('rew/'):]}" if key.startswith("rew/") else key
                log_metrics[name] = float(total) / denom
            wandb.log(log_metrics, step=global_step)

        # -- Best-checkpoint snapshot --
        # Snapshot params when the recent mean true-start *score* improves. The score is gates-passed
        # (skill on the fixed deployment distribution, unshaped) for racing, falling back to return
        # when no gate progress is available (other tasks). pickle.dumps takes an immutable snapshot
        # now (the optimizer mutates `agent` in place each update, so a live reference would not
        # preserve this iteration's weights).
        best_hist = true_start_progress_hist if true_start_progress_hist else true_start_hist
        best_is_progress = bool(true_start_progress_hist)
        if checkpoint_dir is not None and len(best_hist) >= best_min_episodes:
            mean_score = float(np.mean(best_hist[-best_window:]))
            if mean_score > best_score:
                best_score = mean_score
                best_state_bytes = pickle.dumps(nnx.state(agent, nnx.Param))
                if wandb_enabled:
                    key = "charts/best_true_start_gates" if best_is_progress else "charts/best_reward"
                    wandb.log({key: best_score}, step=global_step)

        print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")

        # Graceful manual stop: Ctrl-C finishes the current iteration (best snapshot already taken
        # above, so the saved checkpoint is current), then breaks to write it out. There is no
        # rule-based early stopping; the run otherwise trains for the full horizon.
        if stop_requested["flag"]:
            print("Stopping on user request (Ctrl-C); saving best checkpoint.")
            break

    signal.signal(signal.SIGINT, prev_sigint)  # restore so evaluation/render handles Ctrl-C itself
    train_end_time = time.time()
    print(
        f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds."
    )
    model_path = None
    if checkpoint_dir is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if best_state_bytes is not None:
            # Save the best snapshot (highest recent mean true-start score), not the final state.
            tag = "g" if best_is_progress else "r"  # gates-passed vs return
            metric = "gates-passed" if best_is_progress else "return"
            model_path = checkpoint_dir / f"{run_name}_{timestamp}_{tag}{best_score:.2f}_best.ckpt"
            with open(model_path, "wb") as f:
                f.write(best_state_bytes)
            print(f"Best model (mean true-start {metric} {best_score:.2f}) saved to {model_path}")
        else:
            # No best was recorded (too few completed episodes) -> fall back to the final state.
            recent = sum_rewards_hist[-100:]
            reward_tag = f"r{np.mean(recent):.2f}" if recent else "rNA"
            model_path = checkpoint_dir / f"{run_name}_{timestamp}_{reward_tag}.ckpt"
            with open(model_path, "wb") as f:
                pickle.dump(nnx.state(agent, nnx.Param), f)
            print(f"Final model saved to {model_path} (no best snapshot recorded)")
    envs.close()
    return model_path


def evaluate_ppo(
    args: Args, make_env: MakeEnv, n_eval: int, model_path: Path
) -> tuple[list, list]:
    """Evaluate a trained PPO agent."""
    set_seeds(args.seed)
    eval_env = make_env(args, 1, args.jax_device)
    action_dim = int(np.prod(eval_env.single_action_space.shape))
    obs_dim = int(np.prod(eval_env.single_observation_space.shape))

    # Build a fresh model and load the saved parameters into it (like load_state_dict).
    agent = Agent(obs_dim, action_dim, rngs=nnx.Rngs(0))
    with open(model_path, "rb") as f:
        nnx.update(agent, pickle.load(f))

    @nnx.jit
    def deterministic_action(agent: Agent, obs: Array) -> Array:
        mean, _, _ = agent(obs)
        return mean

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    ep_seed = args.seed

    for episode in range(n_eval):
        obs, _ = eval_env.reset(seed=(ep_seed := ep_seed + 1))
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            obs_jax = jnp.asarray(obs)
            act = deterministic_action(agent, obs_jax)
            obs, reward, terminated, truncated, _ = eval_env.step(act)
            eval_env.render()
            done = bool(jnp.any(jnp.asarray(terminated) | jnp.asarray(truncated)))
            episode_reward += float(jnp.asarray(reward)[0])
            steps += 1
        sim = eval_env.unwrapped.sim
        while (
            sim.viewer is not None
            and sim.viewer.viewer is not None
            and sim.viewer.viewer.window is not None
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=glfw.GLFWError)
                try:
                    eval_env.render()
                except glfw.GLFWError:
                    # ESC called glfw.terminate() without setting window=None; null it out so
                    # WindowViewer.__del__ -> free() skips GLFW calls on garbage collection.
                    if sim.viewer is not None and sim.viewer.viewer is not None:
                        sim.viewer.viewer.window = None
                    sim.viewer = None
                    break
            time.sleep(1 / 60)

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

    print(
        f"Average Reward = {np.mean(episode_rewards):.2f},"
        f" Length = {np.mean(episode_lengths)}"
    )
    eval_env.close()
    return episode_rewards, episode_lengths
