"""Generic PPO training and evaluation, shared across all drone RL tasks.

The algorithm is task-agnostic: it is parameterized by a ``make_env`` factory (which builds
the vectorized, fully-wrapped environment for a given task) and the shared ``Agent`` network.
An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/.
"""

import pickle
import random
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
    model_path: Path,
    jax_device: str,
    wandb_enabled: bool = False,
) -> list[float]:
    """Train a PPO agent on the environment produced by ``make_env``."""
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    train_start_time = time.time()
    set_seeds(args.seed)
    print("Training on device:", jax_device)

    # env setup
    envs = make_env(args, args.num_envs, jax_device)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    action_dim = int(np.prod(act_shape))

    print(f"Shape of observation shape: {obs_shape}")
    print(f"Shape of action space: {act_shape}")

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
        total_loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
        return total_loss, (pg_loss, v_loss, entropy_loss, approx_kl, ratio)

    @nnx.jit
    def update_epoch(
        agent: Agent,
        optimizer: nnx.Optimizer,
        flat_data: tuple,
        rng_key: Array,
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
            (loss, aux), grads = nnx.value_and_grad(ppo_loss_fn, has_aux=True)(agent, *mb)
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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = jnp.asarray(next_obs)
    next_done = jnp.zeros(args.num_envs)
    sum_rewards = jnp.zeros(args.num_envs)
    sum_rewards_hist: list[float] = []

    for iteration in range(1, args.num_iterations + 1):
        start_time = time.time()

        # -- Rollout --
        obs_list, act_list, logp_list, val_list, rew_list, done_list = [], [], [], [], [], []

        for _ in range(args.num_steps):
            global_step += args.num_envs
            obs_list.append(next_obs)
            done_list.append(next_done)

            rng, action_key = jax.random.split(rng)
            action, logprob, value = policy_step(agent, next_obs, action_key)
            act_list.append(action)
            logp_list.append(logprob)
            val_list.append(value)

            next_obs, reward, terminations, truncations, _ = envs.step(action)
            next_obs = jnp.asarray(next_obs)
            reward = jnp.asarray(reward)
            rew_list.append(reward)

            # Track episode returns; reset done envs before accumulating
            sum_rewards = jnp.where(next_done.astype(bool), 0.0, sum_rewards)
            next_done = jnp.asarray(terminations | truncations).astype(jnp.float32)
            sum_rewards = sum_rewards + reward

            if wandb_enabled and bool(jnp.any(next_done)):
                done_indices = np.where(np.array(next_done, dtype=bool))[0]
                for idx in done_indices:
                    r = float(sum_rewards[idx])
                    wandb.log({"train/reward": r}, step=global_step)
                    sum_rewards_hist.append(r)

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

        # -- PPO update epochs --
        last_metrics = None
        for _ in range(args.update_epochs):
            rng, epoch_key = jax.random.split(rng)
            metrics = update_epoch(agent, optimizer, flat_data, epoch_key)
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
            wandb.log(
                {
                    "losses/value_loss": float(jnp.mean(v_arr)),
                    "losses/policy_loss": float(jnp.mean(pg_arr)),
                    "losses/entropy": float(jnp.mean(ent_arr)),
                    "losses/approx_kl": float(jnp.mean(kl_arr)),
                    "losses/clipfrac": float(
                        jnp.mean(jnp.abs(ratio_arr - 1.0) > args.clip_coef)
                    ),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(global_step / (end_time - start_time)),
                },
                step=global_step,
            )
        print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")

    train_end_time = time.time()
    print(
        f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds."
    )
    if model_path is not None:
        with open(model_path, "wb") as f:
            pickle.dump(nnx.state(agent, nnx.Param), f)
        print(f"Model saved to {model_path}")
    envs.close()
    return sum_rewards_hist


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
